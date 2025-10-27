import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.nn.utils.rnn import pad_sequence
from vllm.model_executor.model_loader.weight_utils import LoaderFunction


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :,
                                  None, :, :].expand(batch,
                                                     num_key_value_heads,
                                                     n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen,
                                 head_dim)


def extra_groups_for_head_shards(ngroups: int, tp_size: int):
    """Compute the increase in group numbers to account for
    replication in order to accompany the head shards."""

    # in the case ngoups % tp_size == 0, this will be zero
    if ngroups % tp_size == 0:
        return 0

    # for n_groups == 1, this is exactly tp_size - n_groups
    return tp_size - ngroups


def mamba_v2_sharded_weight_loader(
    shard_spec: list[tuple[int, int, float]],
    tp_size: int,
    tp_rank: int,
) -> LoaderFunction:
    """Create a weight loader for mamba v2. This ensures that the projections 
    are correctly sharded so that they can be split into x, B, C. It also 
    ensures that all the groups corresponding to a head shard is placed 
    together with it.
    """

    def loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:

        # - track boundary of (sharded) param, and loaded_weight, respectively
        boundary, loaded_boundary = 0, 0

        # - iterate over the shard specs
        for full_dim, extra, duplicate_groups in shard_spec:
            # - full dim is the model dim (before TP).
            # - extra > 0, means there is expected overall increase
            #   of dimensions. This is so because of replication.
            # - ratio is used map the tp_rank to the actual shard
            #   rank. This is useful when there is replication of
            #   groups to accompany head shards.

            # - size of the loaded shard
            shard_size = full_dim // tp_size

            # - compute the rank into the loaded shard.
            # - if there is replication, different TP shards will
            #   take from the same rank.
            # NOTE: currently we only support duplication
            # in the case where num_groups == 1
            rank = 0 if duplicate_groups else tp_rank

            # - leftmost boundary index into loaded weight.
            loaded_skip = rank * shard_size
            loaded_start_idx = loaded_boundary + loaded_skip

            # - take these many dims from the loaded weight.
            take = min(shard_size, full_dim - extra - loaded_skip)

            # - always shard on dim 0
            # - the ignore is for a mundane mypy error as it does not
            #   seem to handle slices well.
            # https://github.com/python/mypy/issues/2410
            param.data[
                boundary:(boundary + take),
                ...  # type: ignore[misc]
            ] = loaded_weight[loaded_start_idx:(loaded_start_idx +
                                                take)  # type: ignore[misc]
                              ]  # type: ignore[misc]

            # move indexing boundaries
            boundary += shard_size
            loaded_boundary += full_dim - extra

    return loader


def unbatch_hidden_states(batched_hidden, seq_lens, pad_size=0):
    """
    Args:
        batched_hidden: [batch, max(seq_lens), dim]
        seq_lens: [s1, s2, ..., s_batch]

    Returns:
        hidden_states: [seqlen, dim]
    """
    batch_size = batched_hidden.shape[0]
    sequences = []

    for i in range(batch_size):
        seq_len = seq_lens[i]
        sequence = batched_hidden[i, :seq_len]
        sequences.append(sequence)

    hidden_states = torch.cat(sequences, dim=0)
    out = hidden_states.view(hidden_states.shape[0], -1)

    if pad_size > 0:
        out = F.pad(out, (0, 0, 0, pad_size))

    return out


def batch_hidden_states(hidden_states, seq_lens):
    """
    Args:
        batched_hidden: [seqlen, dim]
        seq_lens: [s1, s2, ..., s_batch]

    Returns:
        hidden_states: [batch, max(seq_lens), dim]
    """
    sequences = []
    start_idx = 0
    for seq_len in seq_lens:
        sequences.append(hidden_states[start_idx:start_idx + seq_len])
        start_idx += seq_len

    # 对序列进行padding
    padded_sequences = pad_sequence(sequences,
                                    batch_first=True,
                                    padding_value=0)

    return padded_sequences


def pad_for_causal_conv(x, lengths, width):
    batch, _, _ = x.shape
    final_states = []

    for i in range(batch):
        L_i = lengths[i]
        pad_length = max(width - 1 - L_i, 0)

        if L_i >= width - 1:
            # 截断最后 width - 1 个状态
            padded = x[i, :, -(width - 1):]  # [dim, width - 1]
        else:
            # 左边填充 pad_length 个零
            padded = F.pad(x[i], (pad_length, 0))  # [dim, width - 1]

        final_states.append(padded.unsqueeze(0))  # [1, dim, width - 1]

    final_states = torch.cat(final_states, dim=0)  # [batch, dim, width - 1]
    return final_states


def causal_conv1d_update(x,
                         conv_state,
                         weight,
                         bias=None,
                         activation=None,
                         cache_seqlens=None,
                         seq_len=None):
    """
    x: (batch, dim) or (batch, dim, seqlen)
    conv_state: (batch, dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    cache_seqlens: (batch,), dtype int32.
        If not None, the conv_state is treated as a circular buffer.
        The conv_state will be updated by copying x to the conv_state starting at the index
        @cache_seqlens % state_len before performing the convolution.

    out: (batch, dim) or (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    width = weight.shape[1]
    conv_state = conv_state.transpose(1, 2)
    state_len = conv_state.shape[-1]
    assert conv_state.shape == (batch, dim, state_len)
    assert weight.shape == (dim, width)
    if cache_seqlens is None:
        x_new = torch.cat([conv_state, x], dim=-1).to(
            weight.dtype)  # (batch, dim, state_len + seqlen)
        conv_state = x_new[:, :, -state_len:]
    else:
        width_idx = torch.arange(
            -(width - 1), 0, dtype=torch.long,
            device=x.device).unsqueeze(0) + cache_seqlens.unsqueeze(1)
        width_idx = torch.remainder(width_idx, state_len).unsqueeze(1).expand(
            -1, dim, -1)
        x_new = torch.cat([conv_state.gather(2, width_idx), x],
                          dim=-1).to(weight.dtype)
        copy_idx = torch.arange(
            seqlen, dtype=torch.long,
            device=x.device).unsqueeze(0) + cache_seqlens.unsqueeze(1)
        copy_idx = torch.remainder(copy_idx,
                                   state_len).unsqueeze(1).expand(-1, dim, -1)
        conv_state.scatter_(2, copy_idx, x)
    out = F.conv1d(x_new, weight.unsqueeze(1), bias, padding=0,
                   groups=dim)[:, :, -seqlen:]
    if unsqueeze:
        out = out.squeeze(-1)
    return (out if activation is None else F.silu(out)).to(
        dtype=dtype_in), conv_state


def causal_conv1d_fn(x,
                     weight,
                     bias=None,
                     initial_states=None,
                     return_final_states=False,
                     final_states_out=None,
                     activation=None,
                     seq_len=None):
    """
    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    initial_states: (batch, dim, width - 1)
    final_states_out: (batch, dim, width - 1)

    out: (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    x = x.to(weight.dtype)
    seqlen = x.shape[-1]
    dim, width = weight.shape
    if initial_states is None:
        out = F.conv1d(x,
                       weight.unsqueeze(1),
                       bias,
                       padding=width - 1,
                       groups=dim)
    else:
        x = torch.cat([initial_states, x], dim=-1)
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=0, groups=dim)
    out = out[..., :seqlen]

    final_states = pad_for_causal_conv(x, seq_len, width)
    if final_states_out is not None:
        final_states_out.copy_(final_states)
    else:
        final_states_out = final_states
    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    return out if not return_final_states else (out, final_states_out)


def chunk_state(B, x, dt, dA_cumsum):
    """
    Argument:
        B: (batch, seqlen, ngroups, headdim)
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, nheads, nchunks, chunk_size)
        dA_cumsum: (batch, nheads, nchunks, chunk_size)
    Return:
        states: (batch, nchunks, nheads, headdim, dstate)
    """
    # Check constraints.
    batch, seqlen, nheads, headdim = x.shape
    dstate = B.shape[-1]
    _, _, nchunks, chunk_size = dt.shape
    assert seqlen <= nchunks * chunk_size
    assert x.shape == (batch, seqlen, nheads, headdim)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    ngroups = B.shape[2]
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    B = repeat(B, "b l g d -> b l (g h) d", h=nheads // ngroups)
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
    if seqlen < nchunks * chunk_size:
        x = F.pad(x, (0, 0, 0, 0, 0, nchunks * chunk_size - seqlen))
        B = F.pad(B, (0, 0, 0, 0, 0, nchunks * chunk_size - seqlen))
    x = rearrange(x, "b (c l) h p -> b c l h p", l=chunk_size)
    B = rearrange(B, "b (c l) ... -> b c l ...", l=chunk_size)
    decay_states = torch.exp((dA_cumsum[:, :, :, -1:] - dA_cumsum))
    return torch.einsum("bclhn,bhcl,bhcl,bclhp->bchpn", B.to(x.dtype),
                        decay_states.to(x.dtype), dt.to(x.dtype), x)


def chunk_scan(B, C, x, dt, dA_cumsum, prev_states, D=None, z=None):
    """
    Argument:
        B: (batch, seqlen, ngroups, dstate)
        C: (batch, seqlen, ngroups, dstate)
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, nheads, nchunks, chunk_size)
        dA_cumsum: (batch, nheads, nchunks, chunk_size)
        prev_states: (batch, nchunks, nheads, headdim, dstate)
        D: (nheads, headdim) or (nheads,)
        z: (batch, seqlen, nheads, headdim)
    Return:
        out: (batch, seqlen, nheads, headdim)
    """
    batch, seqlen, nheads, headdim = x.shape
    _, _, ngroups, dstate = B.shape
    assert B.shape == (batch, seqlen, ngroups, dstate)
    _, _, nchunks, chunk_size = dt.shape
    if seqlen != nchunks * chunk_size:
        a = 1
    assert seqlen == nchunks * chunk_size
    assert C.shape == B.shape
    B = repeat(B, "b l g d -> b l (g h) d", h=nheads // ngroups)
    C = repeat(C, "b l g d -> b l (g h) d", h=nheads // ngroups)
    CB = torch.einsum("bclhn,bcshn->bchls",
                      rearrange(C, "b (c l) h n -> b c l h n", c=nchunks),
                      rearrange(B, "b (c s) h n -> b c s h n", c=nchunks))
    # (batch, nheads, nchunks, chunksize, chunksize)
    dt_segment_sum = dA_cumsum[:, :, :, :, None] - dA_cumsum[:, :, :, None, :]
    decay = torch.exp(dt_segment_sum)
    scores_decay = CB * rearrange(decay, "b h c l s -> b c h l s")
    causal_mask = torch.tril(torch.ones(chunk_size,
                                        chunk_size,
                                        device=x.device,
                                        dtype=bool),
                             diagonal=0)
    scores_decay = scores_decay.masked_fill(~causal_mask, 0)
    out = torch.einsum('bchls,bhcs,bcshp->bclhp', scores_decay.to(x.dtype),
                       dt.to(x.dtype),
                       rearrange(x, "b (c s) h p -> b c s h p", c=nchunks))
    state_decay_out = torch.exp(rearrange(dA_cumsum, "b h c l -> b c l h 1"))
    out_prev = torch.einsum(
        'bclhn,bchpn->bclhp',
        rearrange(C, "b (c l) h n -> b c l h n", c=nchunks),
        prev_states.to(C.dtype)) * state_decay_out
    out = out + out_prev
    out = rearrange(out, "b c l h p -> b (c l) h p")
    if D is not None:
        if D.dim() == 1:
            D = rearrange(D, "h -> h 1")
        out = out + x * D
    return out if z is None else out * F.silu(z)


def state_passing(states, dA_chunk_cumsum, initial_states=None):
    """
    Argument:
        states: (batch, nchunks, nheads, dim)
        dA_chunk_cumsum: (batch, nheads, nchunks)
        initial_states: (batch, nheads, dim)
    Return:
        out: (batch, nchunks, nheads, dim)
        final_states: (batch, nheads, dim)
    """
    if initial_states is None:
        initial_states = torch.zeros_like(states[:, 0])
    states = torch.cat([rearrange(initial_states, "b h d -> b 1 h d"), states],
                       dim=1)
    dA_chunk_cumsum = F.pad(dA_chunk_cumsum, (1, 0))
    dA_chunk_cumsum = torch.cumsum(dA_chunk_cumsum, dim=-1)
    nchunks = dA_chunk_cumsum.shape[-1]
    # (batch, nheads, nchunks, nchunks)
    dt_chunk_segment_sum = dA_chunk_cumsum[:, :, :,
                                           None] - dA_chunk_cumsum[:, :,
                                                                   None, :]
    # (batch, nheads, nchunks, nchunks)
    decay_chunk = torch.exp(dt_chunk_segment_sum)
    causal_mask = torch.tril(torch.ones(nchunks,
                                        nchunks,
                                        device=states.device,
                                        dtype=bool),
                             diagonal=0)
    decay_chunk = decay_chunk.masked_fill(~causal_mask, 0)
    out = torch.einsum("bhzc,bchd->bzhd", decay_chunk.to(dtype=states.dtype),
                       states)
    return out[:, :-1], out[:, -1]


def ssd_chunk_scan_combined(x,
                            dt,
                            A,
                            B,
                            C,
                            chunk_size=256,
                            D=None,
                            z=None,
                            dt_bias=None,
                            dt_softplus=False):
    """
    Argument:
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, seqlen, nheads)
        A: (nheads)
        B: (batch, seqlen, ngroups, dstate)
        C: (batch, seqlen, ngroups, dstate)
        D: (nheads, headdim) or (nheads,)
        z: (batch, seqlen, nheads, headdim)
        dt_bias: (nheads,)
    Return:
        out: (batch, seqlen, nheads, headdim)
    """

    batch, seqlen, nheads, headdim = x.shape
    import os
    vllm_debug_chunk = os.environ.get("VLLM_DEBUG_CHUNK", "0")
    if vllm_debug_chunk == '1':
        origin_seqlen = seqlen
    else:
        chunk_size = 2048

    # adaptive chunk_size
    chunk_size = seqlen

    dstate = B.shape[-1]
    if seqlen < chunk_size:
        chunk_size = seqlen
    if seqlen % chunk_size != 0:
        dt = F.pad(dt, (0, 0, 0, chunk_size - seqlen % chunk_size))
        if vllm_debug_chunk == '1':
            pad_len = chunk_size - seqlen % chunk_size
            # dt = F.pad(dt, (0, 0, 0, pad_len))
            B = F.pad(B, (0, 0, 0, 0, 0, pad_len))
            C = F.pad(C, (0, 0, 0, 0, 0, pad_len))
            x = F.pad(x, (0, 0, 0, 0, 0, pad_len))
            seqlen += pad_len
    dt = rearrange(dt, "b (c l) h -> b h c l", l=chunk_size)
    dt = dt.float()  # We want high precision for this before cumsum
    if dt_bias is not None:
        dt = dt + rearrange(dt_bias, "h -> h 1 1")
    if dt_softplus:
        dt = F.softplus(dt)
    dA = dt * rearrange(A, "h -> h 1 1")
    dA_cumsum = torch.cumsum(dA, dim=-1)
    # 1. Compute the state for each chunk
    states = chunk_state(B, x, dt, dA_cumsum)
    states_dtype = states.dtype
    if states.dtype not in [torch.float32, torch.float64]:
        states = states.to(torch.float32)
    # 2. Pass the state to all the chunks by weighted cumsum.
    # state_passing_ref is much less numerically stable
    states, final_states = state_passing(
        rearrange(states, "... p n -> ... (p n)"), dA_cumsum[:, :, :, -1])
    states, final_states = [
        rearrange(t, "... (p n) -> ... p n", n=dstate)
        for t in [states, final_states]
    ]
    states = states.to(states_dtype)
    final_states = final_states.to(states_dtype)
    # 3. Compute the output for each chunk
    out = chunk_scan(B, C, x, dt, dA_cumsum, states, D=D, z=z)
    if vllm_debug_chunk == '1' and origin_seqlen != seqlen:
        out = out[:, :origin_seqlen]
    return out, final_states
