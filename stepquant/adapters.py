"""Eager HF adapters. Original projections, convolutions, gates, norms and KV caches stay native.

Only the recurrent kernel is replaced. Each attention's post-hook packs its cache;
its pre-hook reconstructs it on the next call. No persistent FP32 state shadow.
Single unpadded request, greedy decode only; not thread-safe or torch.compile compatible.
"""
import inspect
import types
import torch
from .core import delta_step
from .calibration import LayerStatistics
from .quantization import QuantizationPlan, StateCodec, PackedState


def _store_fp32(self, recurrent_states, **kwargs):
    self.recurrent_states = recurrent_states.float()
    self.is_recurrent_states_initialized = True
    return self.recurrent_states


def _kda_gate(g, a_log, head_dim, g_bias=None):
    # Checkpoint uses the older flattened FLA gate API. Compute the same gate
    # explicitly so the adapter is independent of that API's later signature.
    g = g.float()
    if g_bias is not None:
        g = g + g_bias.float()
    g = g.reshape(*g.shape[:-1], -1, head_dim)
    return -a_log.float().reshape(1, 1, -1, 1).exp() * torch.nn.functional.softplus(g)


class RecurrentKernel:
    def __init__(self, architecture, observer=None):
        self.architecture = architecture
        self.observer = observer

    @torch.no_grad()
    def __call__(self, q=None, k=None, v=None, g=None, beta=None, initial_state=None,
                 output_final_state=False, use_qk_l2norm_in_kernel=False, scale=None,
                 cu_seqlens=None, **kwargs):
        if kwargs:
            raise TypeError(f"unsupported recurrent kernel options: {sorted(kwargs)}")
        if q.shape[0] != 1:
            raise ValueError("HF reference adapter supports one unpadded request at a time")
        if cu_seqlens is not None and (cu_seqlens.numel() != 2 or int(cu_seqlens[-1]) != q.shape[1]):
            raise ValueError("variable-length batches are not supported")
        dtype = q.dtype
        q, k, v = q.float(), k.float(), v.float()
        if use_qk_l2norm_in_kernel:
            q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
            k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
        q = q * (q.shape[-1] ** -0.5 if scale is None else scale)
        batch, seq, heads, keys = k.shape
        state = torch.zeros(batch, heads, keys, v.shape[-1], device=q.device) if initial_state is None else initial_state.float()
        outputs = []
        for t in range(seq):
            y, state = delta_step(state, q[:, t], k[:, t], v[:, t], g[:, t], beta[:, t])
            outputs.append(y)
            if self.observer is not None:
                self.observer.observe(q[:, t], k[:, t], v[:, t], g[:, t], beta[:, t], state)
        return torch.stack(outputs, 1).to(dtype), state if output_final_state else None


class ModelAdapter:
    def __init__(self, model, artifact=None, collect=False, sample_every=8, max_snapshots=64):
        if artifact is not None and artifact.get('format_version') != 2:
            raise ValueError('unsupported STEPQuant plan format; regenerate calibration')
        self.model = model
        self.handles, self.originals, self.statistics, self.codecs = [], [], {}, {}
        self.last_bytes = {}
        self.layer_names = []
        for name, module in model.named_modules():
            cls = type(module).__name__
            if cls not in ('Qwen3_5GatedDeltaNet', 'KimiDeltaAttention'):
                continue
            architecture = 'gdn' if cls == 'Qwen3_5GatedDeltaNet' else 'kda'
            self.layer_names.append(name)
            observer = LayerStatistics(architecture, sample_every, max_snapshots) if collect else None
            if observer:
                self.statistics[name] = observer
            codec = None
            if artifact is not None:
                if name not in artifact['plans']:
                    raise ValueError(f"calibration has no plan for {name}")
                plan = QuantizationPlan(**artifact['plans'][name])
                if plan.architecture != architecture:
                    raise ValueError("checkpoint and calibration architecture mismatch")
                codec = StateCodec(plan.to(next(module.parameters()).device))
                self.codecs[name] = codec
            kernel = RecurrentKernel(architecture, observer)
            if architecture == 'gdn':
                for attr in ('chunk_gated_delta_rule', 'recurrent_gated_delta_rule'):
                    self.originals.append((module, attr, getattr(module, attr)))
                    setattr(module, attr, kernel)
            else:
                # Clone only this instance's function globals. Never patch process-global FLA.
                attr = '_old_forward' if hasattr(module, '_hf_hook') else 'forward'
                fn = getattr(module, attr).__func__
                namespace = dict(fn.__globals__, chunk_kda=kernel, fused_recurrent_kda=kernel, fused_kda_gate=_kda_gate)
                cloned = types.FunctionType(fn.__code__, namespace, fn.__name__, fn.__defaults__, fn.__closure__)
                cloned.__kwdefaults__ = fn.__kwdefaults__
                self.originals.append((module, attr, getattr(module, attr)))
                setattr(module, attr, types.MethodType(cloned, module))
            signature = inspect.signature(module.forward)
            def before(mod, args, kwargs, architecture=architecture, codec=codec, signature=signature):
                bound = signature.bind_partial(*args, **kwargs).arguments
                cache = bound.get('cache_params')
                hidden = bound.get('hidden_states')
                mask = bound.get('attention_mask')
                if hidden.shape[0] != 1 or (mask is not None and mask.ndim == 2 and not mask.bool().all()):
                    raise ValueError("STEPQuant HF path requires one unpadded request")
                if cache is None:
                    return
                idx = mod.layer_idx
                if architecture == 'gdn':
                    layer = cache.layers[idx]
                    layer.update_recurrent_state = types.MethodType(_store_fp32, layer)
                    state = layer.recurrent_states
                else:
                    state = cache.recurrent_states[idx]
                if codec is not None and state is not None and hidden.shape[1] != 1:
                    raise ValueError("compressed cached continuation must decode one token at a time")
                if isinstance(state, PackedState):
                    state = codec.decode(state)
                    if architecture == 'gdn':
                        layer.recurrent_states = state
                    else:
                        cache.recurrent_states[idx] = state
            def after(mod, args, kwargs, output, architecture=architecture, codec=codec, name=name, signature=signature):
                if codec is None:
                    return
                cache = signature.bind_partial(*args, **kwargs).arguments.get('cache_params')
                if cache is None:
                    return
                idx = mod.layer_idx
                state = cache.layers[idx].recurrent_states if architecture == 'gdn' else cache.recurrent_states[idx]
                packed = codec.encode(state)
                self.last_bytes[name] = packed.nbytes
                if architecture == 'gdn':
                    cache.layers[idx].recurrent_states = packed
                else:
                    cache.recurrent_states[idx] = packed
            self.handles.append(module.register_forward_pre_hook(before, with_kwargs=True))
            self.handles.append(module.register_forward_hook(after, with_kwargs=True))
        if not self.layer_names:
            raise ValueError("no supported GDN/KDA modules found")
        if artifact is not None and set(artifact['plans']) != set(self.layer_names):
            raise ValueError("calibration layer set does not match model")

    def export_statistics(self):
        return {name: stat.export() for name, stat in self.statistics.items()}

    def close(self):
        for handle in self.handles:
            handle.remove()
        for module, attr, original in reversed(self.originals):
            setattr(module, attr, original)
        self.handles.clear()
        self.originals.clear()
