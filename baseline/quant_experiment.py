"""Numerical-only W8A8 Linear experiments; NOT native INT8 inference.

Conv, attention matrix products, norms, nonlinearities and bias remain FP32.
Bias is added after output dequantization identically in Q1-Q4.
"""
import re
import torch
from torch import nn
from torch.nn import functional as F

DEFAULT_PATTERN = r'layers\.[123]\.blocks\.\d+\.(attn\.(qkv|proj)|mlp\.fc[12])'


def selected(model, pattern):
    modules = {name: m for name, m in model.named_modules()
               if isinstance(m, nn.Linear) and re.fullmatch(pattern, name)}
    if not modules:
        raise ValueError('Pattern selected no Linear modules')
    return modules


def calibrate(model, tensors, pattern=DEFAULT_PATTERN):
    scales = {name: dict(input_absmax=0., output_absmax=0.) for name in selected(model, pattern)}
    handles = []
    def hook(name):
        def observe(m, args, out):
            y = out if m.bias is None else out-m.bias
            if not torch.isfinite(args[0]).all() or not torch.isfinite(y).all():
                raise ValueError('Non-finite calibration values: ' + name)
            s = scales[name]
            s['input_absmax'] = max(s['input_absmax'], args[0].abs().max().item())
            s['output_absmax'] = max(s['output_absmax'], y.abs().max().item())
        return observe
    try:
        for name, m in selected(model, pattern).items():
            handles.append(m.register_forward_hook(hook(name)))
        with torch.inference_mode():
            for x in tensors:
                model(x)
    finally:
        for h in handles:
            h.remove()
    for s in scales.values():
        # All-zero tensors use unit scale rather than a near-zero arbitrary scale.
        s['sx'] = s['input_absmax']/127 if s['input_absmax'] else 1.
        s['sy'] = s['output_absmax']/127 if s['output_absmax'] else 1.
    return scales


class SimulatedLinear(nn.Module):
    def __init__(self, original, scales, variant):
        super().__init__()
        if variant not in ('Q1', 'Q2', 'Q3', 'Q4'):
            raise ValueError(variant)
        self.variant = variant
        self.sx, self.sy = float(scales['sx']), float(scales['sy'])
        if self.sx <= 0 or self.sy <= 0:
            raise ValueError('Scales must be positive')
        w = original.weight.detach().double()
        if not torch.isfinite(w).all():
            raise ValueError('Invalid weights')
        sw = w.abs().amax(dim=1) / 127 if variant == 'Q2' else w.abs().max().reshape(1) / 127
        sw = torch.where(sw == 0, torch.ones_like(sw), sw)
        self.register_buffer('weight', original.weight.detach().clone())
        self.register_buffer('bias', None if original.bias is None else original.bias.detach().clone())
        self.register_buffer('qw', (w/sw[:, None]).round().clamp(-127, 127))
        self.register_buffer('multiplier', sw*self.sx/self.sy)
        q16 = (self.multiplier*65536).round().long()
        if variant in ('Q3', 'Q4') and ((q16 < 1).any() or (q16 > 65535).any()):
            raise ValueError('Multiplier rounds to zero or exceeds Q0.16; recalibrate as a new variant')
        self.register_buffer('q16', q16)
        if w.shape[1]*16384 > 2147483647:
            raise ValueError('INT32 accumulation bound exceeded')
        self.stats = dict(input_clipped=0, input_elements=0, output_clipped=0,
                          output_elements=0, tile_clipped=0, tile_elements=0,
                          merge_clipped=0, merge_elements=0, error_elements=0,
                          abs_error_sum=0., squared_error_sum=0., max_abs_error=0.)

    def clip(self, x, prefix, lo=-128, hi=127):
        self.stats[prefix+'_clipped'] += int(((x < lo) | (x > hi)).sum())
        self.stats[prefix+'_elements'] += x.numel()
        return x.clamp(lo, hi)

    def forward(self, x):
        shape = x.shape
        qx = self.clip((x.double()/self.sx).round(), 'input', -127, 127).reshape(-1, shape[-1])
        # FP64 exactly represents these bounded integer products/sums. This is
        # accelerated numerical emulation, not a claim of integer kernel speed.
        if self.variant == 'Q4':
            y = torch.zeros(qx.shape[0], self.qw.shape[0], dtype=torch.int64)
            for start in range(0, qx.shape[1], 16):
                acc = F.linear(qx[:, start:start+16], self.qw[:, start:start+16]).long()
                part = torch.div(acc*self.q16, 65536, rounding_mode='floor')
                part = self.clip(part, 'tile')
                y = self.clip(y+part, 'merge')
            y = self.clip(y, 'output')
        else:
            acc = F.linear(qx, self.qw)
            if self.variant == 'Q3':
                y = torch.div(acc.long()*self.q16, 65536, rounding_mode='floor')
            else:
                y = (acc*self.multiplier).round()  # ties to even
            y = self.clip(y, 'output')
        out = (y.double()*self.sy).float().reshape(*shape[:-1], self.qw.shape[0])
        if self.bias is not None:
            out = out+self.bias
        ref = F.linear(x, self.weight, self.bias)
        err = (out-ref).double()
        self.stats['error_elements'] += err.numel()
        self.stats['abs_error_sum'] += err.abs().sum().item()
        self.stats['squared_error_sum'] += err.square().sum().item()
        self.stats['max_abs_error'] = max(self.stats['max_abs_error'], err.abs().max().item())
        return out

    def report(self):
        s = dict(self.stats)
        n = s['error_elements']
        s['local_mae'] = s['abs_error_sum']/n if n else None
        s['local_rmse'] = (s['squared_error_sum']/n)**.5 if n else None
        for prefix in ('input', 'output', 'tile', 'merge'):
            count = s[prefix+'_elements']
            s[prefix+'_clipping_rate'] = s[prefix+'_clipped']/count if count else None
        s['multiplier'] = self.multiplier.tolist()
        s['represented_q16_multiplier'] = (self.q16.double()/65536).tolist() if self.variant in ('Q3', 'Q4') else None
        s['error_scope'] = 'local error against FP32 Linear on this SAME propagated input; not full-path FP32 tensor error'
        return s


def apply_quantization(model, calibration, variant):
    modules = dict(model.named_modules())
    for name, scales in calibration['scales'].items():
        original = modules.get(name)
        if not isinstance(original, nn.Linear):
            raise ValueError('Calibration module missing or not Linear: ' + name)
        parent, _, child = name.rpartition('.')
        try:
            replacement = SimulatedLinear(original, scales, variant)
        except ValueError as e:
            raise ValueError(name + ': ' + str(e)) from e
        setattr(modules[parent], child, replacement)
    return model
