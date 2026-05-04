import math

import torch


def normalize_xywh(box_xywh: torch.Tensor, sizes_wh: torch.Tensor) -> torch.Tensor:
    """
    box_xywh: (..., 4) in absolute pixels
    sizes_wh: (..., 2) = [W, H]
    """
    w = sizes_wh[..., 0].unsqueeze(-1).clamp(min=1e-6)
    h = sizes_wh[..., 1].unsqueeze(-1).clamp(min=1e-6)
    out = box_xywh.clone().float()
    out[..., 0] = out[..., 0] / w.squeeze(-1)
    out[..., 1] = out[..., 1] / h.squeeze(-1)
    out[..., 2] = out[..., 2] / w.squeeze(-1)
    out[..., 3] = out[..., 3] / h.squeeze(-1)
    return out


def denormalize_xywh(box_xywh_norm: torch.Tensor, sizes_wh: torch.Tensor) -> torch.Tensor:
    w = sizes_wh[..., 0].unsqueeze(-1).clamp(min=1e-6)
    h = sizes_wh[..., 1].unsqueeze(-1).clamp(min=1e-6)
    out = box_xywh_norm.clone().float()
    out[..., 0] = out[..., 0] * w.squeeze(-1)
    out[..., 1] = out[..., 1] * h.squeeze(-1)
    out[..., 2] = out[..., 2] * w.squeeze(-1)
    out[..., 3] = out[..., 3] * h.squeeze(-1)
    return out


def xywh_to_xyxy(box_xywh: torch.Tensor) -> torch.Tensor:
    x, y, w, h = box_xywh.unbind(dim=-1)
    x2 = x + w
    y2 = y + h
    return torch.stack([x, y, x2, y2], dim=-1)


def iou_xywh(pred_xywh: torch.Tensor, true_xywh: torch.Tensor) -> torch.Tensor:
    pred = pred_xywh.clamp(min=0.0, max=1.0)
    true = true_xywh.clamp(min=0.0, max=1.0)
    p = xywh_to_xyxy(pred)
    t = xywh_to_xyxy(true)
    inter_x1 = torch.max(p[:, 0], t[:, 0])
    inter_y1 = torch.max(p[:, 1], t[:, 1])
    inter_x2 = torch.min(p[:, 2], t[:, 2])
    inter_y2 = torch.min(p[:, 3], t[:, 3])
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h
    p_area = (p[:, 2] - p[:, 0]).clamp(min=0) * (p[:, 3] - p[:, 1]).clamp(min=0)
    t_area = (t[:, 2] - t[:, 0]).clamp(min=0) * (t[:, 3] - t[:, 1]).clamp(min=0)
    union = p_area + t_area - inter
    return inter / (union + 1e-6)


def ciou_xywh(pred_xywh: torch.Tensor, true_xywh: torch.Tensor) -> torch.Tensor:
    pred = pred_xywh.clamp(min=0.0, max=1.0)
    true = true_xywh.clamp(min=0.0, max=1.0)

    p = xywh_to_xyxy(pred)
    t = xywh_to_xyxy(true)

    iou = iou_xywh(pred, true)

    px = (p[:, 0] + p[:, 2]) * 0.5
    py = (p[:, 1] + p[:, 3]) * 0.5
    tx = (t[:, 0] + t[:, 2]) * 0.5
    ty = (t[:, 1] + t[:, 3]) * 0.5
    rho2 = (px - tx) ** 2 + (py - ty) ** 2

    c_x1 = torch.min(p[:, 0], t[:, 0])
    c_y1 = torch.min(p[:, 1], t[:, 1])
    c_x2 = torch.max(p[:, 2], t[:, 2])
    c_y2 = torch.max(p[:, 3], t[:, 3])
    c2 = (c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2 + 1e-6

    pw = (p[:, 2] - p[:, 0]).clamp(min=1e-6)
    ph = (p[:, 3] - p[:, 1]).clamp(min=1e-6)
    tw = (t[:, 2] - t[:, 0]).clamp(min=1e-6)
    th = (t[:, 3] - t[:, 1]).clamp(min=1e-6)
    v = (4 / (math.pi ** 2)) * (torch.atan(tw / th) - torch.atan(pw / ph)) ** 2
    alpha = v / (1 - iou + v + 1e-6)
    return iou - (rho2 / c2 + alpha * v)
