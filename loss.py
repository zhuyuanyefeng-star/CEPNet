import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. 原始的损失函数 (保留以防你想要切回旧版本对比)
# ==========================================
class SoftIoULoss(nn.Module):
    def __init__(self):
        super(SoftIoULoss, self).__init__()

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        smooth = 1
        intersection = pred * target
        intersection_sum = torch.sum(intersection, dim=(1, 2, 3))
        pred_sum = torch.sum(pred, dim=(1, 2, 3))
        target_sum = torch.sum(target, dim=(1, 2, 3))
        loss = (intersection_sum + smooth) / \
               (pred_sum + target_sum - intersection_sum + smooth)
        loss = 1 - torch.mean(loss)
        return loss

def criterion(inputs, target):
    if isinstance(inputs, list):
        losses = [F.binary_cross_entropy_with_logits(inputs[i], target) for i in range(len(inputs))]
        total_loss = sum(losses)
    else:
        total_loss = F.binary_cross_entropy_with_logits(inputs, target)
    return total_loss

def criterion_bce_softiou(inputs, target, iou_weight=0.15):
    if isinstance(inputs, list):
        bce_loss = sum(F.binary_cross_entropy_with_logits(x, target) for x in inputs)
        main_pred = inputs[0]
    else:
        bce_loss = F.binary_cross_entropy_with_logits(inputs, target)
        main_pred = inputs
    return bce_loss + iou_weight * SoftIoULoss()(main_pred, target)

# ==========================================
# 2. 创新的局部对比度焦点损失函数 (LC-Focal-IoU Loss)
# ==========================================
class LCFocalIoULoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, smooth=1.0):
        super(LCFocalIoULoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        # 最大池化核，用于提取局部邻域亮度先验
        self.max_pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, pred, target, input_img=None):
        pred_sigmoid = torch.sigmoid(pred)

        # 1. Focal Loss 计算
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        alpha_t = target * self.alpha + (1 - target) * (1 - self.alpha)
        focal_loss_map = alpha_t * (1 - pt) ** self.gamma * bce_loss

        # 2. 局部对比度引导 (核心创新)
        if input_img is not None:
            if input_img.shape[1] == 3:
                input_img = torch.mean(input_img, dim=1, keepdim=True)
            local_max = self.max_pool(input_img)
            # 差异越小说明越接近局部极值，赋予更大惩罚权重
            contrast_weight = torch.exp(-torch.abs(input_img - local_max))
            focal_loss_map = focal_loss_map * (1.0 + contrast_weight)

        focal_loss = focal_loss_map.mean()

        # 3. Soft-IoU Loss 计算
        intersection = (pred_sigmoid * target).sum(dim=(2, 3))
        union = pred_sigmoid.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        iou_loss = 1.0 - iou.mean()

        # 4. 损失融合 (Focal 负责背景，IoU 负责形状)
        total_loss = focal_loss + 2 * iou_loss
        return total_loss

def criterion_LC(inputs, target, input_img=None):
    loss_fn = LCFocalIoULoss()
    if isinstance(inputs, list):
        losses = [loss_fn(inputs[i], target, input_img) for i in range(len(inputs))]
        total_loss = sum(losses)
    else:
        total_loss = loss_fn(inputs, target, input_img)
    return total_loss
