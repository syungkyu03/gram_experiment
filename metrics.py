import math
import torch
import torch.nn.functional as F


def gram(z):
    z = F.normalize(z.float(), dim=-1)
    return z @ z.transpose(-1, -2)


def ring_scores(z, zt, top, left, side_a=8, margin=4):
    side = math.isqrt(z.shape[0])
    if side * side != z.shape[0]:
        raise ValueError('Expected square patch grid')
    y, x = torch.meshgrid(torch.arange(side, device=z.device), torch.arange(side, device=z.device), indexing='ij')
    dy = torch.maximum(torch.maximum(top - y, y - (top + side_a - 1)), torch.zeros_like(y))
    dx = torch.maximum(torch.maximum(left - x, x - (left + side_a - 1)), torch.zeros_like(x))
    dist = torch.maximum(dy, dx).flatten()
    reference = dist > margin
    diff = gram(zt) - gram(z)
    patch = (zt - z).norm(dim=-1)
    result = []
    for distance in range(1, int(dist.max()) + 1):
        ring = dist == distance
        eligible = ring[:, None] & reference[None, :]
        eligible.fill_diagonal_(False)
        if not eligible.any():
            raise ValueError('No valid reference edges for ring')
        result.append((distance, int(ring.sum()), float(patch[ring].mean()), float(diff.abs()[eligible].mean())))
    b = dist > 0
    pairs = b[:, None] & b[None, :]
    pairs.fill_diagonal_(False)
    return result, float(diff.abs()[pairs].mean())


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
           'cls_change': (cls - clst).norm(dim=-1),
           'cls_cosine_change': 1 - F.cosine_similarity(cls, clst, dim=-1)}
    center = lambda x: x - x.mean(-1, keepdim=True) - x.mean(-2, keepdim=True) + x.mean((-1, -2), keepdim=True)
    a, b = center(g), center(gt)
    out['cka_change'] = 1 - (a * b).sum((-1, -2)) / (a.square().sum((-1, -2)).sqrt() * b.square().sum((-1, -2)).sqrt()).clamp_min(1e-12)
    if mask is not None:
        # Any perturbed pixel marks the corresponding patch as affected.
        affected = F.adaptive_max_pool2d(mask[:, None].float(), (side, side)).flatten(1).bool()
        patch_diff = (z - zt).norm(dim=-1)
        for name, valid in [('patch_change_a', affected), ('patch_change_b', ~affected)]:
            count = valid.sum(-1)
            out[name] = ((patch_diff * valid).sum(-1) / count.clamp_min(1)).masked_fill(count == 0, float('nan'))
        first, second = affected[:, idx[0]], affected[:, idx[1]]
        for name, valid in [('gram_aa', first & second), ('gram_ab', first ^ second), ('gram_bb', ~first & ~second)]:
            count = valid.sum(-1)
            out[name] = ((d.abs() * valid).sum(-1) / count.clamp_min(1)).masked_fill(count == 0, float('nan'))
    return out
