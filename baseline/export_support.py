"""Equivalent static window partition for batch-1 224 export.

Avoid six-dimensional window reshapes/transposes unsupported by some converters.
Always compare against the unmodified reference before accepting an export.
"""
import copy
import types
import torch


def static_windows_forward(self, x):
    h, w = self.input_resolution
    ws = self.window_size
    b, _, c = x.shape
    residual = x
    grid = x.reshape(b, h, w, c)
    # Keep batch=1 for each window; converters may not support concatenation
    # along batch. Windows are independent at inference. Export may duplicate
    # shared weights; measure resulting artifact size and runtime overhead.
    strips = []
    for i in range(0, h, ws):
        row = []
        for j in range(0, w, ws):
            window = grid[:, i:i+ws, j:j+ws, :].reshape(b, ws*ws, c)
            row.append(self.attn(window).reshape(b, ws, ws, c))
        strips.append(torch.cat(row, dim=2))
    x = torch.cat(strips, dim=1).reshape(b, h*w, c)
    x = residual+self.drop_path(x)
    x = self.local_conv(x.transpose(1, 2).reshape(b, c, h, w))
    x = x.reshape(b, c, h*w).transpose(1, 2)
    return x+self.drop_path(self.mlp(x))


def export_model(model):
    result = copy.deepcopy(model)
    for layer in result.layers[1:]:
        for block in layer.blocks:
            h, w = block.input_resolution
            ws = block.window_size
            if h == ws and w == ws:
                continue
            if h % ws or w % ws:
                raise ValueError('Static export supports only divisible window dimensions')
            block.forward = types.MethodType(static_windows_forward, block)
    return result.eval()
