import torch
import torch.nn.functional as F

def dice_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        multiclass: bool = False,
        from_logits: bool = False,
        smooth: float = 1.0,
        eps: float = 1e-7,
) -> torch.Tensor:
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)  # [B,1,H,W]
    if target.dim() == 3:
        target = target.unsqueeze(1)

    if multiclass:
        if from_logits:
            pred = F.softmax(pred, dim=1)
        else:
            pred = torch.clamp(pred, 0, 1)
        B, C, H, W = pred.shape
        if target.shape == (B, 1, H, W):  # 索引形式
            target_idx = target.squeeze(1).long()
            target = F.one_hot(target_idx, num_classes=C).permute(0, 3, 1, 2).float()
        elif target.shape == (B, C, H, W):
            target = target.float()
        else:
            raise ValueError("For multiclass, target must be [B,1,H,W] or [B,C,H,W]")

        dims = (0, 2, 3)
        inter = (pred * target).sum(dim=dims)
        card  = (pred + target).sum(dim=dims)
        dice  = (2. * inter + smooth) / (card + smooth + eps)
        return 1. - dice.mean()

    # binary
    if from_logits:
        pred = torch.sigmoid(pred)
    else:
        pred = torch.clamp(pred, 0, 1)

    target = target.float()
    if target.shape != pred.shape:
        if target.dim() == 3 and pred.dim() == 4 and pred.size(1) == 1:
            target = target.unsqueeze(1)
        else:
            raise ValueError("Shapes for binary dice must match")

    dims = (0, 2, 3)
    inter = (pred * target).sum(dim=dims)
    card  = (pred + target).sum(dim=dims)
    dice  = (2. * inter + smooth) / (card + smooth + eps)
    return 1. - dice.mean()


@torch.no_grad()
def dice_coeff(
    input: torch.Tensor,
    target: torch.Tensor,
    reduce_batch_first: bool = True,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    if input.dim() == 3:
        input = input.unsqueeze(1)
    if target.dim() == 3:
        target = target.unsqueeze(1)
    input  = input.float()
    target = target.float()

    if reduce_batch_first:
        B = input.size(0)
        dices = []
        for b in range(B):
            x, y = input[b], target[b]
            inter = (x * y).sum()
            sets  = x.sum() + y.sum()
            if sets.item() == 0:
                sets = 2 * inter
            dices.append((2.0 * inter + epsilon) / (sets + epsilon))
        return torch.stack(dices).mean()
    else:
        inter = (input * target).sum()
        sets  = input.sum() + target.sum()
        if sets.item() == 0:
            sets = 2 * inter
        return (2.0 * inter + epsilon) / (sets + epsilon)
