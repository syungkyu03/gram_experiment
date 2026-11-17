import math
import torch
import torch.nn.functional as F


def gram(z):
    z = F.normalize(z.float(), dim=-1)
    return z @ z.transpose(-1, -2)


def static_score(g):
    n = g.shape[-1]
    if n < 2:
        raise ValueError('At least two patch tokens are required')
    idx = torch.triu_indices(n, n, 1, device=g.device)
    values = g[:, idx[0], idx[1]]
    return values.topk(max(1, math.ceil(values.shape[1] * .1)), largest=False).values.mean(-1)


def scores(z, cls, zt, clst, mask=None):
    g, gt = gram(z), gram(zt)
    delta = gt - g
    batch, n, _ = delta.shape
    side = math.isqrt(n)
    if side * side != n:
        raise ValueError('Expected a square patch grid')
    idx = torch.triu_indices(n, n, 1, device=z.device)
    d = delta[:, idx[0], idx[1]]
    coords = torch.stack(torch.meshgrid(torch.arange(side, device=z.device), torch.arange(side, device=z.device), indexing='ij'), -1).reshape(-1, 2)
    distance = (coords[idx[0]] - coords[idx[1]]).float().norm(dim=-1)
    near = distance <= side / 4
    far = distance > side / 2
    out = {'gram_l1': d.abs().mean(-1), 'gram_signed': d.mean(-1),
           'gram_near': d[:, near].abs().mean(-1), 'gram_far': d[:, far].abs().mean(-1),
           'patch_change': (z - zt).norm(dim=-1).mean(-1),
           'cls_change': (cls - clst).norm(dim=-1)}
    center = lambda x: x - x.mean(-1, keepdim=True) - x.mean(-2, keepdim=True) + x.mean((-1, -2), keepdim=True)
    a, b = center(g), center(gt)
    out['cka_change'] = 1 - (a * b).sum((-1, -2)) / (a.square().sum((-1, -2)).sqrt() * b.square().sum((-1, -2)).sqrt()).clamp_min(1e-12)
    if mask is not None:
        # Any perturbed pixel marks the corresponding patch as affected.
        affected = F.adaptive_max_pool2d(mask[:, None].float(), (side, side)).flatten(1).bool()
        valid = (~affected[:, idx[0]]) & (~affected[:, idx[1]])
        out['gram_bb'] = (d.abs() * valid).sum(-1) / valid.sum(-1).clamp_min(1)
        out['gram_bb'] = out['gram_bb'].masked_fill(valid.sum(-1) == 0, float('nan'))
    return out
