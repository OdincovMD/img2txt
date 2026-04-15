"""
Метрики: Precision@k, Recall@k.
"""

import torch


def precision_recall_at_k(
    logits: torch.Tensor,
    target: torch.Tensor,
    k: int = 10,
) -> tuple[float, float]:
    """
    Усреднённые по батчу Precision@k и Recall@k.

    logits: (B, num_labels)  — сырые логиты
    target: (B, num_labels)  — бинарные метки
    """
    probs = torch.sigmoid(logits)
    _, top_idx = torch.topk(probs, k=min(k, probs.size(1)), dim=1)
    pred = torch.zeros_like(probs)
    pred.scatter_(1, top_idx, 1.0)

    intersection = (pred * target).sum(dim=1)
    num_true = target.sum(dim=1).clamp(min=1e-6)

    precision = (intersection / k).clamp(max=1.0)
    recall = (intersection / num_true).clamp(max=1.0)
    recall = torch.where(target.sum(dim=1) > 0, recall, torch.zeros_like(recall))

    return precision.mean().item(), recall.mean().item()
