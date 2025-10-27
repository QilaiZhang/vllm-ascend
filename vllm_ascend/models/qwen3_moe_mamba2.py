"""Inference-only Qwen3MoeMamba model."""
from collections.abc import Iterable
from typing import Optional, Union

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from vllm.attention.backends.abstract import AttentionBackend
from vllm.config import (CacheConfig, ModelConfig, VllmConfig,
                         get_current_vllm_config)
from vllm.distributed import (divide, get_pp_group,
                              get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_mixer2 import (
    Mixer2RMSNormGated, mamba_v2_sharded_weight_loader)
from vllm.model_executor.layers.mamba.mamba_utils import \
    MambaStateDtypeCalculator
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    composed_weight_loader, default_weight_loader, sharded_weight_loader)
from vllm.model_executor.models.interfaces import (IsHybrid, SupportsLoRA,
                                                   SupportsPP)
from vllm.model_executor.models.qwen3_moe import (Qwen3MoeDecoderLayer,
                                                  Qwen3MoeSparseMoeBlock)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader, extract_layer_index, is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory, make_layers, maybe_prefix)
from vllm.model_executor.utils import set_weight_attrs
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.qwen3_moe_mamba2 import MambaConfig

from vllm_ascend.models.qwen3_moe_mamba2_utils import (batch_hidden_states,
                                                       causal_conv1d_fn,
                                                       causal_conv1d_update,
                                                       repeat_kv,
                                                       ssd_chunk_scan_combined,
                                                       unbatch_hidden_states)

logger = init_logger(__name__)


def qwen3_mamba2_state_shape(
    tp_world_size: int,
    intermediate_size: int,
    xb_size: int,
    num_heads: int,
    head_dim: int,
    state_size: int,
    conv_kernel: int,
    use_v1: bool = True,
) -> tuple[tuple[int, int], tuple[int, int, int]]:

    # heads and n_groups are TP-ed
    conv_dim = intermediate_size + 2 * xb_size

    # contiguous along 'dim' axis
    conv_state_shape = (conv_kernel - 1, divide(conv_dim, tp_world_size))
    if not use_v1:
        conv_state_shape = conv_state_shape[1], conv_state_shape[0]

    # These are not TP-ed as they depend on A, dt_bias, D
    # - they are typically small
    #   e.g., (h_heads, head_dim, state_size) = (128, 64, 128)
    temporal_state_shape = (divide(num_heads,
                                   tp_world_size), head_dim, state_size)
    return conv_state_shape, temporal_state_shape


class MambaMixer2Hybrid(nn.Module, MambaBase):

    def __init__(
        self,
        config: MambaConfig,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.n_groups = config.ssm_cfg['ngroups']
        if self.n_groups % self.tp_size != 0:
            self.n_groups = self.tp_size

        self.hidden_size = config.hidden_size
        self.xb_size = config.d_xb
        self.ssm_state_size = config.d_state
        self.intermediate_size = config.d_inner
        self.num_heads = config.ssm_cfg['ngroups']
        self.head_dim = self.intermediate_size // self.n_groups
        self.repeat_group = self.intermediate_size // self.xb_size
        self.activation = config.hidden_act
        self.prefix = prefix

        self.conv_dim = self.intermediate_size + self.xb_size * 2
        self.conv_kernel_size = config.d_conv
        self.conv1d = ColumnParallelLinear(
            input_size=self.conv_kernel_size,
            output_size=self.conv_dim,
            bias=config.use_conv_bias,
            prefix=f"{prefix}.conv1d",
            quant_config=quant_config,
        )
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        self.in_proj = ColumnParallelLinear(
            input_size=self.hidden_size,
            output_size=self.intermediate_size + self.conv_dim +
            self.num_heads,
            bias=config.use_bias,
            prefix=f"{prefix}.in_proj",
            quant_config=quant_config)

        self.groups_time_state_size = self.n_groups * self.ssm_state_size
        group_shard_settings = (
            self.groups_time_state_size,  # expected model size
            (self.n_groups - config.ssm_cfg['ngroups']) *
            self.ssm_state_size,  # extra dims assigned
            config.ssm_cfg['ngroups'] == 1,  # if there was only one group
        )
        intermediate_settings = (self.intermediate_size, 0, False)
        head_setings = (self.num_heads, 0, False)
        xb_settings = (self.xb_size, 0, False)

        delattr(self.conv1d.bias, "weight_loader")
        set_weight_attrs(
            self.conv1d.bias, {
                "weight_loader":
                mamba_v2_sharded_weight_loader(
                    [
                        xb_settings,
                        xb_settings,
                        group_shard_settings,
                    ],
                    self.tp_size,
                    self.tp_rank,
                )
            })

        delattr(self.conv1d.weight, "weight_loader")
        set_weight_attrs(
            self.conv1d.weight, {
                "weight_loader":
                mamba_v2_sharded_weight_loader([
                    xb_settings,
                    xb_settings,
                    group_shard_settings,
                ], self.tp_size, self.tp_rank)
            })

        if quant_config is None:
            # - quant layers do not have a weight loader
            delattr(self.in_proj.weight, "weight_loader")
            set_weight_attrs(
                self.in_proj.weight,
                {
                    "weight_loader":
                    mamba_v2_sharded_weight_loader(
                        [
                            intermediate_settings,  # for gate
                            xb_settings,
                            xb_settings,
                            group_shard_settings,
                            head_setings,  # for dt
                        ],
                        self.tp_size,
                        self.tp_rank)
                })

        self.A_log = nn.Parameter(
            torch.empty(
                divide(self.num_heads, self.tp_size),
                dtype=torch.float32,
            ))
        self.A = self.A_log
        self.D = nn.Parameter(torch.ones(self.num_heads // self.tp_size))
        self.dt_bias = nn.Parameter(torch.ones(self.num_heads // self.tp_size))

        a_weight_loader = composed_weight_loader(
            sharded_weight_loader(0), lambda x: -torch.exp(x.float()))
        set_weight_attrs(self.D, {"weight_loader": sharded_weight_loader(0)})
        set_weight_attrs(self.A, {"weight_loader": a_weight_loader})
        set_weight_attrs(self.dt_bias,
                         {"weight_loader": sharded_weight_loader(0)})

        self.out_proj = RowParallelLinear(self.intermediate_size,
                                          self.hidden_size,
                                          bias=config.use_bias,
                                          input_is_parallel=True,
                                          prefix=f"{prefix}.out_proj",
                                          quant_config=quant_config)

        self.norm = Mixer2RMSNormGated(self.intermediate_size,
                                       self.n_groups,
                                       eps=config.rms_norm_eps)

        self.model_config = model_config
        self.cache_config = cache_config

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        self.kv_cache = [(torch.tensor([]), torch.tensor([]))]

    def forward(self, hidden_states: torch.Tensor):

        # 0. attn_metadata
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        if attn_metadata is None:
            return hidden_states

        attn_metadata = attn_metadata[self.prefix]
        self_kv_cache = self.kv_cache[forward_context.virtual_engine]
        conv_state = self_kv_cache[0]
        ssm_state = self_kv_cache[1]
        state_indices_tensor = attn_metadata.state_indices_tensor
        pad_size = hidden_states.shape[0] - torch.sum(attn_metadata.seq_lens)

        # 1. Gated MLP's linear projection
        projected_states, _ = self.in_proj(hidden_states)
        gate, hidden_states_B_C, dt = torch.split(
            projected_states,
            [
                self.intermediate_size // self.tp_size,
                self.conv_dim // self.tp_size,
                self.num_heads // self.tp_size,
            ],
            dim=-1,
        )

        # 2. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0),
                                               self.conv1d.weight.size(2))

        if attn_metadata.num_prefills > 0:
            hidden_states_batched = batch_hidden_states(
                hidden_states_B_C, attn_metadata.seq_lens)
            hidden_states_B_C_batched, final_states_out = causal_conv1d_fn(
                x=hidden_states_batched.transpose(1, 2),
                weight=conv_weights,
                bias=self.conv1d.bias,
                initial_states=None,
                activation=self.activation,
                final_states_out=None,
                return_final_states=True,
                seq_len=attn_metadata.seq_lens)
            conv_state[state_indices_tensor] = final_states_out.transpose(1, 2)
            hidden_states_B_C = hidden_states_B_C_batched.transpose(1, 2)
        else:
            hidden_states_B_C, final_states_out = causal_conv1d_update(
                x=hidden_states_B_C,
                conv_state=conv_state[state_indices_tensor],
                weight=conv_weights,
                bias=self.conv1d.bias,
                activation=self.activation,
                seq_len=attn_metadata.seq_lens)
            conv_state[state_indices_tensor] = final_states_out.transpose(1, 2)

        # - get hidden_states, B and C after depthwise convolution.
        hidden_states, B, C = torch.split(
            hidden_states_B_C,
            [
                self.xb_size // self.tp_size,
                self.xb_size // self.tp_size,
                self.groups_time_state_size // self.tp_size,
            ],
            dim=-1,
        )

        # 3. State Space Model sequence transformation
        if attn_metadata.num_prefills > 0:
            hidden_states = rearrange(
                hidden_states,
                "b l (xb_group dstate) -> b xb_group l dstate",
                dstate=self.ssm_state_size)
            hidden_states = repeat_kv(hidden_states, self.repeat_group)
            hidden_states = rearrange(hidden_states, "b g l p -> b l g p")

            B = rearrange(B,
                          "b l (xb_group dstate) -> b xb_group l dstate",
                          dstate=self.ssm_state_size)
            B = repeat_kv(B, self.repeat_group)
            B = rearrange(B, "b g l n -> b l g n")

            C = rearrange(C, "b l (g n) -> b l g n", g=B.shape[-2])

            scan_output, states = ssd_chunk_scan_combined(
                x=hidden_states,
                dt=batch_hidden_states(dt, attn_metadata.seq_lens),
                A=self.A,
                B=B,
                C=C,
                D=self.D,
                z=None,
                dt_bias=self.dt_bias,
                dt_softplus=True,
                chunk_size=256)
            # update ssm states
            ssm_state[state_indices_tensor] = states
            hidden_states = unbatch_hidden_states(
                scan_output.to(hidden_states.dtype), attn_metadata.seq_lens,
                pad_size)
        else:
            # minic the GQA
            hidden_states = rearrange(
                hidden_states,
                "b (xb_group dstate) -> b xb_group dstate",
                dstate=self.ssm_state_size)
            x = torch.repeat_interleave(hidden_states,
                                        dim=1,
                                        repeats=self.repeat_group)

            B = rearrange(B,
                          "b (xb_group dstate) -> b xb_group dstate",
                          dstate=self.ssm_state_size)
            B = torch.repeat_interleave(B, dim=1, repeats=self.repeat_group)

            A = repeat(self.A,
                       "h -> h p n",
                       p=self.head_dim,
                       n=self.ssm_state_size).to(dtype=torch.float32)
            dt = repeat(dt, "b h -> b h p", p=self.head_dim)
            dt_bias = repeat(self.dt_bias, "h -> h p", p=self.head_dim)
            D = repeat(self.D, "h -> h p", p=self.head_dim)
            C = rearrange(C, "b (g n) -> b g n", g=B.shape[-2])

            state = ssm_state[state_indices_tensor]
            batch, nheads, _, _ = state.shape
            ngroups = B.shape[1]

            if dt_bias is not None:
                dt = dt + dt_bias
            dt = F.softplus(dt)
            dA = torch.exp(rearrange(dt, "b h d -> b h d 1") *
                           A)  # (batch, nheads, dim, dstate)
            B = repeat(B, "b g n -> b (g h) n",
                       h=nheads // ngroups)  # (batch, nheads, dstate)
            C = repeat(C, "b g n -> b (g h) n",
                       h=nheads // ngroups)  # (batch, nheads, dstate)
            dB = rearrange(dt, "b h d -> b h d 1") * rearrange(
                B, "b h n -> b h 1 n")  # (batch, nheads, dim, dstate)
            out = torch.einsum("bhdn,bhn->bhd", state.to(C.dtype), C)
            if D is not None:
                out += (x * D).to(out.dtype)
            hidden_states = out.view(batch, -1).to(x.dtype)
            state_new = state * dA + dB * rearrange(x, "b h d -> b h d 1")
            ssm_state[state_indices_tensor] = state_new.to(
                C.dtype)  # (batch, nheads, dim, dstate)

        # 4. gated MLP
        hidden_states = self.norm(hidden_states, gate)

        # 5. Final linear projection
        out, _ = self.out_proj(hidden_states)

        return out

    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
        assert self.model_config is not None
        assert self.cache_config is not None
        return MambaStateDtypeCalculator.mamba2_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
            self.cache_config.mamba_ssm_cache_dtype,
        )

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return qwen3_mamba2_state_shape(
            intermediate_size=self.intermediate_size,
            tp_world_size=get_tensor_model_parallel_world_size(),
            xb_size=self.xb_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            state_size=self.ssm_state_size,
            conv_kernel=self.conv_kernel_size,
        )

    @property
    def mamba_type(self) -> str:
        return "mamba2"

    def get_attn_backend(self) -> type["AttentionBackend"]:
        from vllm.v1.attention.backends.mamba2_attn import \
            Mamba2AttentionBackend
        return Mamba2AttentionBackend


class Mamba2DecoderLayer(nn.Module):

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        config = MambaConfig(**model_config.hf_config.mamba_config)
        if config.d_inner is None:
            config.d_inner = config.ssm_cfg['expand'] * config.d_model

        self.mamba = MambaMixer2Hybrid(config=config,
                                       model_config=model_config,
                                       cache_config=cache_config,
                                       quant_config=quant_config,
                                       prefix=f"{prefix}.mamba2")
        self.mlp = Qwen3MoeSparseMoeBlock(vllm_config, prefix=f"{prefix}.mlp")
        self.input_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.d_model,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)

        hidden_states = self.mamba(hidden_states)

        residual = hidden_states + residual
        hidden_states = self.post_attention_layernorm(residual)
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class Qwen3MoeMamba2Model(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens")

        def get_layer(prefix):
            idx = extract_layer_index(prefix)
            if idx in config.mamba_config['attn_layers']:
                return Qwen3MoeDecoderLayer(vllm_config=vllm_config,
                                            prefix=prefix)
            else:
                return Mamba2DecoderLayer(vllm_config, prefix)

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            get_layer,
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            if isinstance(layer, Qwen3MoeDecoderLayer):
                hidden_states, residual = layer(positions=positions,
                                                hidden_states=hidden_states,
                                                residual=residual)
            else:
                hidden_states, residual = layer(
                    hidden_states=hidden_states,
                    residual=residual,
                )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Skip loading extra parameters for GPTQ/modelopt models.
        ignore_suffixes = (".bias", "_bias", ".k_scale", "_k_scale",
                           ".v_scale", "_v_scale", ".weight_scale",
                           "_weight_scale", ".input_scale", "_input_scale")

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()
        for name, loaded_weight in weights:
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue

                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)

                # Skip loading extra parameters for GPTQ/modelopt models.
                if name.endswith(ignore_suffixes) and name not in params_dict:
                    continue

                # Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue
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
                    # Skip loading extra parameters for GPTQ/modelopt models.
                    if name.endswith(
                            ignore_suffixes) and name not in params_dict:
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
                    # Skip loading extra parameters for GPTQ/modelopt models.
                    if name.endswith(
                            ignore_suffixes) and name not in params_dict:
                        continue
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Remapping the name of FP8 kv-scale.
                    if name.endswith("kv_scale"):
                        remapped_kv_scale_name = name.replace(
                            ".kv_scale", ".attn.kv_scale")
                        if remapped_kv_scale_name not in params_dict:
                            logger.warning_once(
                                "Found kv scale in the checkpoint (e.g. %s), but not found the expected name in the model (e.g. %s). kv-scale is not loaded.",  # noqa: E501
                                name,
                                remapped_kv_scale_name,
                            )
                            continue
                        else:
                            name = remapped_kv_scale_name
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Qwen3MoeMamba2ForCausalLM(nn.Module, SupportsPP, SupportsLoRA, IsHybrid):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.vllm_config = vllm_config
        self.config = config
        self.quant_config = quant_config
        self.model_config = vllm_config.model_config
        self.mamba_config = MambaConfig(**config.mamba_config)
        assert not vllm_config.cache_config.enable_prefix_caching, \
            "Mamba does not support prefix caching"

        self.model = Qwen3MoeMamba2Model(vllm_config=vllm_config,
                                         prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(config.vocab_size,
                                      config.hidden_size,
                                      quant_config=quant_config)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                intermediate_tensors: Optional[IntermediateTensors] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                **kwargs) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(input_ids, positions, intermediate_tensors,
                                   inputs_embeds)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:

        return MambaStateDtypeCalculator.mamba2_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: "VllmConfig",
        use_v1: bool = True,
    ) -> tuple[tuple[int, int], tuple[int, int, int]]:
        """Calculate shapes for Mamba's convolutional and state caches.

        Args:
            vllm_config: vLLM config
            use_v1: Get shapes for V1 (or V0)

        Returns:
            Tuple containing:
            - conv_state_shape: Shape for convolutional state cache
            - temporal_state_shape: Shape for state space model cache
        """
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_config
        mamba_config = MambaConfig(**hf_config.mamba_config)

        return qwen3_mamba2_state_shape(
            intermediate_size=mamba_config.d_inner,
            tp_world_size=parallel_config.tensor_parallel_size,
            xb_size=mamba_config.d_xb,
            num_heads=mamba_config.ssm_cfg['ngroups'],
            head_dim=mamba_config.d_inner // mamba_config.ssm_cfg['ngroups'],
            state_size=mamba_config.d_state,
            conv_kernel=mamba_config.d_conv,
            use_v1=use_v1,
        )
