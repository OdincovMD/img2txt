"""
Метрики для модели отбора важных признаков: Precision@10, Recall@10.
"""

import torch


def precision_recall_at_k(
    logits: torch.Tensor,
    target: torch.Tensor,
    k: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    logits: (batch, num_labels), target: (batch, num_labels) бинарные.
    Для каждого сэмпла: топ-k по logits считаем предсказаниями;
    Precision@k = |pred ∩ true| / k, Recall@k = |pred ∩ true| / |true|.
    Если у сэмпла 0 истинных меток, Recall не определён — возвращаем 0 для этого сэмпла.
    Возвращает (precision_batch, recall_batch) — тензоры длины batch.
    """
    batch = logits.size(0)
    probs = torch.sigmoid(logits)
    _, top_idx = torch.topk(probs, k=min(k, probs.size(1)), dim=1)
    pred_one_hot = torch.zeros_like(probs, device=probs.device)
    pred_one_hot.scatter_(1, top_idx, 1.0)

    intersection = (pred_one_hot * target).sum(dim=1)
    num_true = target.sum(dim=1).clamp(min=1e-6)
    precision = (intersection / k).clamp(max=1.0)
    recall = (intersection / num_true).clamp(max=1.0)
    recall = torch.where(target.sum(dim=1) > 0, recall, torch.zeros_like(recall))
    return precision, recall


def precision_recall_at_k_batch(
    logits: torch.Tensor,
    target: torch.Tensor,
    k: int = 10,
) -> tuple[float, float]:
    """
    Усреднённые по батчу Precision@k и Recall@k.
    """
    p, r = precision_recall_at_k(logits, target, k=k)
    return p.mean().item(), r.mean().item()
