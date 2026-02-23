import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    
    Formula: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: 平衡因子，用于处理类别不平衡 (default: 0.25)
        gamma: 聚焦参数，gamma越大，对简单样本的权重降低越多 (default: 2.0)
        reduction: 'mean', 'sum' or 'none'
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, C, H, W] 或 [B, 1, H, W]，logits（未经过sigmoid）
            targets: [B, C, H, W] 或 [B, 1, H, W]，二值标签 (0 or 1)
        """
        # 将logits转换为概率
        probs = torch.sigmoid(inputs)
        
        # 计算二值交叉熵
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # 计算pt (正确类别的预测概率)
        # 对于正样本：pt = p, 对于负样本：pt = 1-p
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # 计算alpha_t
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Focal Loss = -alpha_t * (1 - p_t)^gamma * log(p_t)
        # 等价于: alpha_t * (1 - p_t)^gamma * BCE
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        focal_loss = focal_weight * bce
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedLoss(nn.Module):
    """
    组合损失：Focal Loss + Dice Loss
    
    Args:
        focal_weight: Focal Loss的权重 (default: 1.0)
        dice_weight: Dice Loss的权重 (default: 1.0)
        focal_alpha: Focal Loss的alpha参数 (default: 0.25)
        focal_gamma: Focal Loss的gamma参数 (default: 2.0)
    """
    def __init__(self, focal_weight=1.0, dice_weight=1.0, focal_alpha=0.25, focal_gamma=2.0):
        super(CombinedLoss, self).__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    
    def dice_loss(self, inputs, targets, smooth=1e-6):
        """
        计算Dice Loss
        
        Args:
            inputs: [B, C, H, W]，logits
            targets: [B, C, H, W]，二值标签
        """
        probs = torch.sigmoid(inputs)
        
        # 展平
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        # 计算Dice系数
        intersection = (probs_flat * targets_flat).sum()
        dice = (2. * intersection + smooth) / (probs_flat.sum() + targets_flat.sum() + smooth)
        
        return 1 - dice
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, C, H, W]，logits
            targets: [B, C, H, W]，二值标签
        """
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        
        return self.focal_weight * focal + self.dice_weight * dice


