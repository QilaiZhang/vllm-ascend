# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# mypy: ignore-errors
"""Inference-only Qwen3Next model."""
from collections.abc import Iterable
from typing import Optional

import torch
from einops import rearrange
from torch import nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from vllm.attention import AttentionBackend, AttentionMetadata
from vllm.compilation.decorators import support_torch_compile
from vllm.config import (CacheConfig, ModelConfig, SpeculativeConfig,
                         VllmConfig, get_current_vllm_config)
from vllm.distributed import (divide, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fla.ops import RMSNormGated
from vllm.model_executor.layers.fla.ops.chunk import chunk_gated_delta_rule
from vllm.model_executor.layers.fla.ops.fused_recurrent import \
    fused_recurrent_gated_delta_rule
from vllm.model_executor.layers.fused_moe import FusedMoE
# yapf conflicts with isort for this block
# yapf: disable
from vllm.model_executor.layers.layernorm import \
    GemmaRMSNorm as Qwen3NextRMSNorm
# yapf: enable
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_mixer2 import \
    mamba_v2_sharded_weight_loader
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator, MambaStateShapeCalculator)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn, causal_conv1d_update)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, sharded_weight_loader)
from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP as Qwen3NextMLP
from vllm.model_executor.models.utils import (
    PPMissingLayer, extract_layer_index, is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory, make_layers, maybe_prefix)
from vllm.model_executor.utils import set_weight_attrs
from vllm.transformers_utils.configs import Qwen3NextConfig
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

from vllm_ascend.utils import vllm_version_is

from vllm.model_executor.models.qwen3_next import (  # isort: skip
    Qwen3NextAttention, Qwen3NextDecoderLayer, Qwen3NextForCausalLM,
    Qwen3NextGatedDeltaNet, Qwen3NextModel, Qwen3NextSparseMoeBlock,
    fused_gdn_gating)


class CustomQwen3NextGatedDeltaNet(Qwen3NextGatedDeltaNet, MambaBase):

    @property
    def mamba_type(self) -> str:
        return "linear_attention"

    def get_attn_backend(self) -> type["AttentionBackend"]:
        from vllm.v1.attention.backends.gdn_attn import GDNAttentionBackend
        return GDNAttentionBackend

    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            self.model_config.dtype, self.cache_config.mamba_cache_dtype)

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            self.tp_size, self.num_k_heads, self.num_v_heads, self.head_k_dim,
            self.head_v_dim, self.conv_kernel_size, self.num_spec)

    def __init__(
        self,
        config: Qwen3NextConfig,
        model_config: Optional[ModelConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        speculative_config: Optional[SpeculativeConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = extract_layer_index(prefix)
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]
        self.layer_norm_epsilon = config.rms_norm_eps
        self.prefix = prefix

        self.config = config
        self.model_config = model_config
        self.cache_config = cache_config
        self.quant_config = quant_config
        self.speculative_config = speculative_config
        self.num_spec = (self.speculative_config.num_speculative_tokens
                         if self.speculative_config else 0)
        self.prefix_caching_enabled = self.cache_config.enable_prefix_caching

        # QKV
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = ColumnParallelLinear(
            input_size=self.conv_kernel_size,
            output_size=self.conv_dim,
            bias=False,
            prefix=f"{prefix}.conv1d",
        )
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        # projection of the input hidden states
        self.projection_size_qkvz = self.key_dim * 2 + self.value_dim * 2
        self.projection_size_ba = self.num_v_heads * 2
        self.in_proj = MergedColumnParallelLinear(
            input_size=self.hidden_size,
            output_sizes=[self.projection_size_qkvz, self.projection_size_ba],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.in_proj",
        )

        query_key_settings = (self.key_dim, 0, False)
        value_settings = (self.value_dim, 0, False)

        delattr(self.conv1d.weight, "weight_loader")
        set_weight_attrs(
            self.conv1d.weight, {
                "weight_loader":
                mamba_v2_sharded_weight_loader([
                    query_key_settings,
                    query_key_settings,
                    value_settings,
                ], self.tp_size, self.tp_rank)
            })

        # selective projection used to make dt, B and C input dependent

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(
            torch.ones(self.num_v_heads // self.tp_size), )
        self.A_log = nn.Parameter(
            torch.empty(
                divide(self.num_v_heads, self.tp_size),
                dtype=torch.float32,
            ))

        set_weight_attrs(self.A_log,
                         {"weight_loader": sharded_weight_loader(0)})
        set_weight_attrs(self.dt_bias,
                         {"weight_loader": sharded_weight_loader(0)})

        self.norm = RMSNormGated(
            self.head_v_dim,
            eps=self.layer_norm_epsilon,
            norm_before_gate=True,
            device="npu",
        )

        self.out_proj = RowParallelLinear(self.value_dim,
                                          self.hidden_size,
                                          bias=False,
                                          input_is_parallel=True,
                                          quant_config=quant_config,
                                          prefix=f"{prefix}.out_proj")

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def _forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        forward_context = get_forward_context()
        attn_metadata: AttentionMetadata = forward_context.attn_metadata

        if attn_metadata is None:
            # V1 profile run
            return

        assert isinstance(attn_metadata, dict)
        attn_metadata = attn_metadata[self.prefix]
        assert isinstance(attn_metadata, GDNAttentionMetadata)
        has_initial_state = attn_metadata.has_initial_state
        spec_query_start_loc = attn_metadata.spec_query_start_loc
        non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        spec_sequence_masks = attn_metadata.spec_sequence_masks
        if vllm_version_is("0.11.0"):
            spec_token_masks = attn_metadata.spec_token_masks
        else:
            spec_token_indx = attn_metadata.spec_token_indx
            non_spec_token_indx = attn_metadata.non_spec_token_indx
        spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor  # noqa: E501
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
        self_kv_cache = self.kv_cache[forward_context.virtual_engine]

        state_indices_tensor_d = attn_metadata.state_indices_tensor_d
        state_indices_tensor_p = attn_metadata.state_indices_tensor_p
        block_idx_last_computed_token_d = attn_metadata.block_idx_last_computed_token_d
        block_idx_last_scheduled_token_d = attn_metadata.block_idx_last_scheduled_token_d
        block_idx_first_scheduled_token_p = attn_metadata.block_idx_first_scheduled_token_p
        block_idx_last_computed_token_p = attn_metadata.block_idx_last_computed_token_p
        block_idx_last_scheduled_token_p = attn_metadata.block_idx_last_scheduled_token_p

        conv_state = self_kv_cache[0].transpose(-1, -2)
        ssm_state = self_kv_cache[1]

        num_actual_tokens = (attn_metadata.num_prefill_tokens +
                             attn_metadata.num_decode_tokens +
                             attn_metadata.num_spec_decode_tokens)
        num_accepted_tokens = attn_metadata.num_accepted_tokens

        non_spec_state_indices_runtime = non_spec_state_indices_tensor
        state_indices_decode: torch.Tensor | None = None
        state_indices_prefill: torch.Tensor | None = None

        start_non_spec_prefill = attn_metadata.num_decodes
        end_non_spec_prefill = start_non_spec_prefill + attn_metadata.num_prefills

        if (
            self.prefix_caching_enabled
            and non_spec_state_indices_tensor is not None
            and non_spec_state_indices_tensor.numel() > 0
        ):
            non_spec_state_indices_runtime = non_spec_state_indices_tensor.clone()

            num_decodes = attn_metadata.num_decodes
            if (
                num_decodes > 0
                and state_indices_tensor_d is not None
                and block_idx_last_computed_token_d is not None
                and block_idx_last_scheduled_token_d is not None
            ):
                decode_slice = slice(0, num_decodes)
                base_decode_slots = non_spec_state_indices_tensor[decode_slice]
                gathered_last_computed = (
                    block_idx_last_computed_token_d[:num_decodes]
                    .clamp(min=0)
                    .to(torch.int32)
                )
                gathered_last_scheduled = (
                    block_idx_last_scheduled_token_d[:num_decodes]
                    .clamp(min=0)
                    .to(torch.int32)
                )
                slot_in = state_indices_tensor_d.gather(
                    1, gathered_last_computed.unsqueeze(1)
                ).squeeze(1)
                slot_out = state_indices_tensor_d.gather(
                    1, gathered_last_scheduled.unsqueeze(1)
                ).squeeze(1)
                valid_in = (block_idx_last_computed_token_d[:num_decodes] >= 0) & (slot_in >= 0)
                valid_out = (block_idx_last_scheduled_token_d[:num_decodes] >= 0) & (slot_out >= 0)
                slot_out_safe = torch.where(valid_out, slot_out, base_decode_slots)
                slot_in_safe = torch.where(valid_in, slot_in, slot_out_safe)
                if slot_out_safe.numel() > 0 and (slot_out_safe != slot_in_safe).any():
                    slot_out_copy = slot_out_safe.clamp(min=0).to(device=conv_state.device, dtype=torch.int32)
                    slot_in_copy = slot_in_safe.clamp(min=0).to(device=conv_state.device, dtype=torch.int32)
                    # conv_state.index_copy_(
                    #     0,
                    #     slot_out_copy,
                    #     conv_state.index_select(0, slot_in_copy)
                    # )
                    ssm_state.index_copy_(
                        0,
                        slot_out_copy,
                        ssm_state.index_select(0, slot_in_copy)
                    )
                updated_decode_slots = torch.where(
                    valid_out,
                    slot_out,
                    base_decode_slots,
                )
                non_spec_state_indices_runtime[decode_slice] = updated_decode_slots
                state_indices_decode = updated_decode_slots

            num_prefills = attn_metadata.num_prefills
            if (
                num_prefills > 0
                and state_indices_tensor_p is not None
                and block_idx_last_computed_token_p is not None
                and block_idx_last_scheduled_token_p is not None
            ):
                start = attn_metadata.num_decodes
                end = start + num_prefills
                base_prefill_slots = non_spec_state_indices_tensor[start:end]
                gathered_last_computed = (
                    block_idx_last_computed_token_p[:num_prefills]
                    .clamp(min=0)
                    .to(torch.int32)
                )
                gathered_last_scheduled = (
                    block_idx_last_scheduled_token_p[:num_prefills]
                    .clamp(min=0)
                    .to(torch.long)
                )
                slot_in = state_indices_tensor_p.gather(
                    1, gathered_last_computed.unsqueeze(1)
                ).squeeze(1)
                slot_out = state_indices_tensor_p.gather(
                    1, gathered_last_scheduled.unsqueeze(1)
                ).squeeze(1)
                valid_in = (block_idx_last_computed_token_p[:num_prefills] >= 0) & (
                    slot_in >= 0
                )
                valid_out = (block_idx_last_scheduled_token_p[:num_prefills] >= 0) & (
                    slot_out >= 0
                )
                slot_out_safe = torch.where(
                    valid_out,
                    slot_out,
                    base_prefill_slots,
                )
                slot_in_safe = torch.where(
                    valid_in,
                    slot_in,
                    slot_out_safe,
                )
                if slot_out_safe.numel() > 0 and (slot_out_safe != slot_in_safe).any():
                    slot_out_copy = slot_out_safe.clamp(min=0).to(
                        device=conv_state.device, dtype=torch.int32
                    )
                    slot_in_copy = slot_in_safe.clamp(min=0).to(
                        device=conv_state.device, dtype=torch.int32
                    )
                    conv_state.index_copy_(
                        0,
                        slot_out_copy,
                        conv_state.index_select(0, slot_in_copy)
                    )
                    ssm_state.index_copy_(
                        0,
                        slot_out_copy,
                        ssm_state.index_select(0, slot_in_copy)
                    )

                updated_prefill_slots = torch.where(
                    valid_out,
                    slot_out,
                    base_prefill_slots,
                )
                non_spec_state_indices_runtime[start:end] = updated_prefill_slots
                state_indices_prefill = updated_prefill_slots

        if state_indices_decode is None and non_spec_state_indices_tensor is not None:
            state_indices_decode = non_spec_state_indices_tensor[:attn_metadata.num_decodes]

        if state_indices_prefill is None and non_spec_state_indices_tensor is not None:
            state_indices_prefill = non_spec_state_indices_tensor[start_non_spec_prefill:end_non_spec_prefill]

        if attn_metadata.num_decodes > 0:
            assert state_indices_decode is not None

        if attn_metadata.num_prefills > 0:
            assert state_indices_prefill is not None    # No use
            assert non_spec_state_indices_runtime is not None

        # 1. Set up dimensions for reshapes later
        projected_states, _ = self.in_proj(hidden_states[:num_actual_tokens])
        if vllm_version_is("0.11.0"):
            if spec_token_masks is not None:
                spec_token_masks = spec_token_masks[:num_actual_tokens]
        projected_states_qkvz, projected_states_ba = torch.split(
            projected_states,
            [
                self.projection_size_qkvz // self.tp_size,
                self.projection_size_ba // self.tp_size
            ],
            dim=-1,
        )
        query, key, value, z, b, a = self.fix_query_key_value_ordering(
            projected_states_qkvz, projected_states_ba)
        query, key, value = map(lambda x: rearrange(x, 'l p d -> l (p d)'),
                                (query, key, value))
        mixed_qkv = torch.cat((query, key, value), dim=-1)

        # 2. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0),
                                               self.conv1d.weight.size(2))

        if spec_sequence_masks is not None:
            if (attn_metadata.num_prefills == 0
                    and attn_metadata.num_decodes == 0):
                mixed_qkv_spec = mixed_qkv
                mixed_qkv_non_spec = None
            else:
                if vllm_version_is("0.11.0"):
                    mixed_qkv_spec = mixed_qkv[spec_token_masks]
                    mixed_qkv_non_spec = mixed_qkv[~spec_token_masks]
                else:
                    mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
                    mixed_qkv_non_spec = mixed_qkv.index_select(
                        0, non_spec_token_indx)
        else:
            mixed_qkv_spec = None
            mixed_qkv_non_spec = mixed_qkv

        # 2.1: process the multi-query part
        if spec_sequence_masks is not None:
            mixed_qkv_spec = mixed_qkv_spec.view(
                attn_metadata.num_spec_decodes, -1, mixed_qkv_spec.size(-1))
            mixed_qkv_spec = rearrange(mixed_qkv_spec, 'b l d -> b d l')
            mixed_qkv_spec = causal_conv1d_update(
                mixed_qkv_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=spec_state_indices_tensor[:, 0]
                [:attn_metadata.num_spec_decodes],
                num_accepted_tokens=num_accepted_tokens,
                validate_data=False,
            )
            mixed_qkv_spec = rearrange(mixed_qkv_spec, 'b d l -> (b l) d')

        # 2.2: process the remaining part
        if attn_metadata.num_prefills > 0:
            # - "cache_indices" updates the conv_state cache in positions
            #   pointed to by "mamba_cache_params.state_indices_tensor"
            # x = mixed_qkv_non_spec.transpose(0, 1)
            # block_size = self.cache_config.block_size
            # for idx in range(has_initial_state.shape[0]):
            #     if idx < num_decodes:
            #         continue
            #     cur_x = x[:, non_spec_query_start_loc[idx]:non_spec_query_start_loc[idx+1]]
            #     start_idx = 0
            #     if has_initial_state[idx]:
            #         cur_x = torch.cat([conv_state[non_spec_state_indices_runtime[idx]], cur_x], dim=1)
            #         start_idx = conv_state.shape[-1]
            #     num_to_fill = block_idx_last_scheduled_token_p[idx-num_decodes] - block_idx_first_scheduled_token_p[idx-num_decodes]
            #     for i in range(num_to_fill):
            #         start = start_idx + (i+1) * block_size - 3
            #         end = start_idx + (i+1) * block_size
            #         conv_state[state_indices_tensor_p[idx-num_decodes, block_idx_first_scheduled_token_p+i].item()] = cur_x[:, start:end]
            mixed_qkv_non_spec = causal_conv1d_fn(
                mixed_qkv_non_spec.transpose(0, 1),
                conv_weights,
                self.conv1d.bias,
                activation=self.activation,
                conv_states=conv_state,
                has_initial_state=has_initial_state,
                cache_indices=non_spec_state_indices_runtime,
                query_start_loc=non_spec_query_start_loc,
            ).transpose(0, 1)
        elif attn_metadata.num_decodes > 0:
            mixed_qkv_non_spec = causal_conv1d_update(
                mixed_qkv_non_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=state_indices_tensor_d,
                current_last_idx=block_idx_last_computed_token_d,
                initial_state_idx=block_idx_last_computed_token_d,
                # validate_data=True,
            )
        else:
            mixed_qkv_non_spec = None

        query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(
            mixed_qkv_spec)
        query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(
            mixed_qkv_non_spec)

        beta = b.sigmoid()
        g = fused_gdn_gating(self.A_log, a, self.dt_bias)
        g, beta = map(lambda x: rearrange(x, 'l d -> 1 l d'), (g, beta))

        if spec_sequence_masks is not None:
            if (attn_metadata.num_prefills == 0
                    and attn_metadata.num_decodes == 0):
                g_spec = g
                beta_spec = beta
                g_non_spec = None
                beta_non_spec = None
            else:
                if vllm_version_is("0.11.0"):
                    g_spec = g[:, spec_token_masks]
                    beta_spec = beta[:, spec_token_masks]
                    g_non_spec = g[:, ~spec_token_masks]
                    beta_non_spec = beta[:, ~spec_token_masks]
                else:
                    g_spec = g.index_select(1, spec_token_indx)
                    beta_spec = beta.index_select(1, spec_token_indx)
                    g_non_spec = g.index_select(1, non_spec_token_indx)
                    beta_non_spec = beta.index_select(1, non_spec_token_indx)
        else:
            g_spec = None
            beta_spec = None
            g_non_spec = g
            beta_non_spec = beta

        # 3. Recurrent attention
        # 3.1: process the multi-query part
        if spec_sequence_masks is not None:
            core_attn_out_spec, last_recurrent_state = (
                fused_recurrent_gated_delta_rule(
                    q=query_spec,
                    k=key_spec,
                    v=value_spec,
                    g=g_spec,
                    beta=beta_spec,
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    cu_seqlens=spec_query_start_loc[:attn_metadata.
                                                    num_spec_decodes + 1],
                    ssm_state_indices=spec_state_indices_tensor,
                    num_accepted_tokens=num_accepted_tokens,
                    use_qk_l2norm_in_kernel=True,
                ))
        else:
            core_attn_out_spec, last_recurrent_state = None, None

        # 3.2: process the remaining part
        if attn_metadata.num_prefills > 0:
            # initial_state = ssm_state[non_spec_state_indices_tensor].contiguous()
            # initial_state[~has_initial_state, ...] = 0
            chunk_state_indices = non_spec_state_indices_runtime[:end_non_spec_prefill]
            initial_state = ssm_state.new_zeros(
                (chunk_state_indices.shape[0], *ssm_state.shape[1:])
            )
            if chunk_state_indices.numel() > 0:
                valid_chunk_slots = chunk_state_indices >= 0
                valid_chunk_positions = torch.nonzero(
                    valid_chunk_slots, as_tuple=False
                ).squeeze(-1)
                if valid_chunk_positions.numel() > 0:
                    initial_state.index_copy_(
                        0,
                        valid_chunk_positions,
                        ssm_state.index_select(
                            0,
                            chunk_state_indices.index_select(
                                0,
                                valid_chunk_positions
                            ).to(device=ssm_state.device, dtype=torch.int32)
                        )
                    )

                if has_initial_state is not None:
                    chunk_has_initial_state = has_initial_state[:end_non_spec_prefill]
                    initial_state[~chunk_has_initial_state, ...] = 0

            assert query_non_spec is not None
            assert key_non_spec is not None
            assert value_non_spec is not None
            assert g_non_spec is not None
            assert beta_non_spec is not None
            non_spec_query_start_loc = non_spec_query_start_loc[: end_non_spec_prefill + 1]

            batch_size = initial_state.shape[0]
            core_attn_out = []
            last_recurrent_state = []
            block_state_history = []

            for b_idx in range(batch_size):
                start, end = non_spec_query_start_loc[
                    b_idx], non_spec_query_start_loc[b_idx + 1]
                cur_q = query_non_spec[:, start:end, ...]
                cur_k = key_non_spec[:, start:end, ...]
                cur_v = value_non_spec[:, start:end, ...]
                cur_g = g_non_spec[:, start:end, ...]
                cur_b = beta_non_spec[:, start:end, ...]
                cur_state = initial_state[b_idx].unsqueeze(0)
                (
                    cur_core_attn_out_non_spec,
                    cur_last_recurrent_state,
                    cur_intermediate_states
                ) = chunk_gated_delta_rule(
                    query=cur_q,
                    key=cur_k,
                    value=cur_v,
                    g=cur_g,
                    beta=cur_b,
                    initial_state=cur_state,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                    return_intermediate_states=self.prefix_caching_enabled,
                )

                core_attn_out.append(cur_core_attn_out_non_spec)
                last_recurrent_state.append(cur_last_recurrent_state)
                block_state_history.extend(cur_intermediate_states)

            tar_dtype = core_attn_out[0].dtype
            tar_device = core_attn_out[0].device
            tar_shape = list(core_attn_out[0].shape)
            tar_shape[1] = non_spec_query_start_loc[-1]
            core_attn_out_non_spec = torch.empty(tar_shape,
                                                 dtype=tar_dtype,
                                                 device=tar_device)
            for b_idx in range(batch_size):
                cur_core_attn_out = core_attn_out[b_idx]
                start, end = non_spec_query_start_loc[
                    b_idx], non_spec_query_start_loc[b_idx + 1]
                core_attn_out_non_spec[:, start:end, ...] = cur_core_attn_out
            last_recurrent_state = torch.cat(last_recurrent_state, dim=0)
            block_state_history = torch.cat(block_state_history, dim=0) if block_state_history else None

            if chunk_state_indices.numel() > 0:
                valid_chunk_slots = chunk_state_indices >= 0
                valid_chunk_positions = torch.nonzero(
                    valid_chunk_slots, as_tuple=False
                ).squeeze(-1)
                if valid_chunk_positions.numel() > 0:
                    dest_slots = chunk_state_indices.index_select(
                        0, valid_chunk_positions
                    ).to(device=ssm_state.device, dtype=torch.int32)
                    ssm_state.index_copy_(
                        0,
                        dest_slots,
                        last_recurrent_state.index_select(0, valid_chunk_positions).to(
                            ssm_state.dtype
                        ),
                    )

            if self.prefix_caching_enabled and (
                block_state_history is not None
                and block_idx_first_scheduled_token_p is not None
                and block_idx_last_scheduled_token_p is not None
                and state_indices_tensor_p is not None
                and attn_metadata.last_chunk_indices_p is not None
                and attn_metadata.num_computed_tokens_p is not None
                and attn_metadata.chunk_size is not None
                and attn_metadata.block_size is not None
            ):
                block_history = block_state_history.to(ssm_state.dtype)
                total_chunks = block_history.shape[0]
                last_chunk_indices = attn_metadata.last_chunk_indices_p
                prefill_chunk_count = (
                    int(last_chunk_indices[-1].item()) + 1
                    if last_chunk_indices is not None and last_chunk_indices.numel() > 0
                    else 0
                )
                decode_chunk_count = max(total_chunks - prefill_chunk_count, 0)
                block_history_prefill = block_history[decode_chunk_count:]
                if block_history_prefill.shape[0] > 0:
                    chunk_size = attn_metadata.chunk_size
                    block_size = attn_metadata.block_size
                    chunk_stride = block_size // chunk_size
                    last_chunk_indices = attn_metadata.last_chunk_indices_p
                    last_chunk_indices_long = last_chunk_indices.to(torch.int32)
                    num_computed_tokens_p = attn_metadata.num_computed_tokens_p

                    for seq_idx in range(attn_metadata.num_prefills):
                        block_first = int(
                            block_idx_first_scheduled_token_p[seq_idx].item()
                        )
                        block_last = int(
                            block_idx_last_scheduled_token_p[seq_idx].item()
                        )
                        n_blocks_to_fill = block_last - block_first
                        if n_blocks_to_fill <= 0:
                            continue

                        cache_blocks = state_indices_tensor_p[
                            seq_idx, block_first:block_last
                        ].to(torch.int32)

                        first_chunk = (
                            0
                            if seq_idx == 0
                            else int(last_chunk_indices[seq_idx - 1].item()) + 1
                        )
                        first_aligned_chunk = first_chunk + chunk_stride - 1
                        num_unaligned_tokens = int(
                            num_computed_tokens_p[seq_idx].item() % block_size
                        )
                        if num_unaligned_tokens > 0:
                            first_aligned_chunk -= num_unaligned_tokens // chunk_size
                        chunk_stop = (
                            first_aligned_chunk + n_blocks_to_fill * chunk_stride
                        )
                        cached_states = block_history_prefill[
                            first_aligned_chunk:chunk_stop:chunk_stride
                        ]
                        ssm_state[cache_blocks] = cached_states

                    # final_slots = state_indices_tensor_p.gather(
                    #     1, block_idx_last_scheduled_token_p.unsqueeze(1)
                    # ).squeeze(1)
                    # valid_final = final_slots >= 0
                    # valid_final_positions = torch.nonzero(
                    #     valid_final, as_tuple=False
                    # ).squeeze(-1)
                    # if valid_final_positions.numel() > 0:
                    #     final_slot_ids = final_slots.index_select(
                    #         0, valid_final_positions
                    #     ).to(device=ssm_state.device, dtype=torch.int32)
                    #     final_states = block_history_prefill.index_select(
                    #         0,
                    #         last_chunk_indices_long.index_select(
                    #             0, valid_final_positions
                    #         ),
                    #     )
                    #     ssm_state.index_copy_(0, final_slot_ids, final_states)

                

        elif attn_metadata.num_decodes > 0:
            core_attn_out_non_spec, last_recurrent_state = (
                fused_recurrent_gated_delta_rule(
                    q=query_non_spec,
                    k=key_non_spec,
                    v=value_non_spec,
                    g=g_non_spec,
                    beta=beta_non_spec,
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    cu_seqlens=non_spec_query_start_loc[:attn_metadata.
                                                        num_decodes + 1],
                    ssm_state_indices=state_indices_decode,
                    use_qk_l2norm_in_kernel=True,
                ))
        else:
            core_attn_out_non_spec, last_recurrent_state = None, None

        # Merge core attention output
        if (spec_sequence_masks is not None
                and core_attn_out_non_spec is not None):
            core_attn_out = torch.empty(
                (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),
                dtype=core_attn_out_non_spec.dtype,
                device=core_attn_out_non_spec.device,
            )
            if vllm_version_is("0.11.0"):
                core_attn_out[:, spec_token_masks] = core_attn_out_spec
                core_attn_out[:, ~spec_token_masks] = core_attn_out_non_spec
            else:
                core_attn_out.index_copy_(1, spec_token_indx,
                                          core_attn_out_spec)
                core_attn_out.index_copy_(1, non_spec_token_indx,
                                          core_attn_out_non_spec)
        elif spec_sequence_masks is not None:
            core_attn_out = core_attn_out_spec
        else:
            core_attn_out = core_attn_out_non_spec

        z_shape_og = z.shape
        # reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = rearrange(core_attn_out, '... h d -> ... (h d)')

        output[:num_actual_tokens], _ = self.out_proj(core_attn_out)


class CustomQwen3NextDecoderLayer(Qwen3NextDecoderLayer):

    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        speculative_config = vllm_config.speculative_config

        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)

        if self.layer_type == "linear_attention":
            self.linear_attn = CustomQwen3NextGatedDeltaNet(
                config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                speculative_config=speculative_config,
                prefix=f'{prefix}.linear_attn')
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3NextAttention(
                config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f'{prefix}.self_attn',
            )
        else:
            raise ValueError(f"Invalid layer_type {self.layer_type}")

        mlp_only_layers = ([] if not hasattr(config, "mlp_only_layers") else
                           config.mlp_only_layers)
        if (self.layer_idx not in mlp_only_layers) and (
                config.num_experts > 0 and
            (self.layer_idx + 1) % config.decoder_sparse_step == 0):
            self.mlp = Qwen3NextSparseMoeBlock(vllm_config=vllm_config,
                                               prefix=f"{prefix}.mlp")
        else:
            self.mlp = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
            )

        self.input_layernorm = Qwen3NextRMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3NextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)

        self.layer_scale = getattr(config, "layer_scale", False)
        if self.layer_scale:
            self.attn_layer_scale = torch.nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    config.hidden_size,
                    dtype=config.torch_dtype,
                ), )
            self.ffn_layer_scale = torch.nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    config.hidden_size,
                    dtype=config.torch_dtype,
                ), )


@support_torch_compile
class CustomQwen3NextModel(Qwen3NextModel):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        config: Qwen3NextConfig = vllm_config.model_config.hf_config
        parallel_config = vllm_config.parallel_config
        lora_config = vllm_config.lora_config
        eplb_config = parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts

        self.config = config
        lora_vocab = ((lora_config.lora_extra_vocab_size *
                       (lora_config.max_loras or 1)) if lora_config else 0)
        self.vocab_size = config.vocab_size + lora_vocab

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )

        def get_layer(prefix: str):
            return CustomQwen3NextDecoderLayer(
                vllm_config,
                layer_type=config.layer_types[extract_layer_index(prefix)],
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers")
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

        self.norm = Qwen3NextRMSNorm(config.hidden_size,
                                     eps=config.rms_norm_eps)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            ("in_proj", "in_proj_qkvz", 0),
            ("in_proj", "in_proj_ba", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            if name.startswith("mtp."):
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue

                if "mlp.experts" in name:
                    continue

                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue
                # name = apply_attn_prefix(name, params_dict)
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Skip loading extra bias for GPTQ models.
                    if ((name.endswith(".bias") or name.endswith("_bias"))
                            and name not in params_dict):
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param,
                                  loaded_weight,
                                  name,
                                  shard_id=shard_id,
                                  expert_id=expert_id)
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class CustomQwen3NextForCausalLM(Qwen3NextForCausalLM):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        # cache_config = vllm_config.cache_config
        lora_config = vllm_config.lora_config
        scheduler_config = vllm_config.scheduler_config
        # assert not cache_config.enable_prefix_caching, \
        #     "Qwen3Next currently does not support prefix caching"
        self.quant_config = vllm_config.quant_config
        self.config = config
        self.scheduler_config = scheduler_config
        self.model = CustomQwen3NextModel(vllm_config=vllm_config,
                                          prefix=maybe_prefix(prefix, "model"))
        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE
            # We need bigger padding if using lora for kernel
            # compatibility
            if not lora_config else lora_config.lora_vocab_padding_size,
        )
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

        # Set MoE hyperparameters
        self.expert_weights = []

        self.moe_layers: list[FusedMoE] = []
        example_layer = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue

            assert isinstance(layer, Qwen3NextDecoderLayer)
            if isinstance(layer.mlp, Qwen3NextSparseMoeBlock):
                example_layer = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        if example_layer is None:
            raise RuntimeError("No Qwen3Next layer found in the model.layers.")

        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_logical_experts = example_layer.n_logical_experts
        self.num_physical_experts = example_layer.n_physical_experts
        self.num_local_physical_experts = example_layer.n_local_physical_experts
        self.num_routed_experts = example_layer.n_routed_experts
        self.num_redundant_experts = example_layer.n_redundant_experts
