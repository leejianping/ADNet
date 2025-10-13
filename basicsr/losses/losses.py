import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np

#from basicsr.losses.loss_util import weighted_loss
from losses.loss_util import weighted_loss

#======================================
#BinaryLoss, sobel_edge_loss, gan_loss
#=========================================


class BinaryLoss(nn.Module):
    """
    Binary Cross-Entropy loss for QR code binary outputs.
    """
    def __init__(self):
        super().__init__()
        #self.bce = nn.BCEWithLogitsLoss()
        self.bce = nn.BCELoss()

    def forward(self, output, target):
        """
        Compute BCE loss between output and target.
        Args:
            output (torch.Tensor): Predicted logits [batch_size, 1, H, W].
            target (torch.Tensor): Ground truth binary image [batch_size, 1, H, W].
        Returns:
            torch.Tensor: Loss value.
        """
        return self.bce(output, target)


        
class SobelEdgeLoss(nn.Module):
    """
    Sobel edge loss to emphasize sharp edges in QR codes.
    """
    def __init__(self):
        super().__init__()
        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_x.weight = nn.Parameter(sobel_x, requires_grad=False)
        self.sobel_y.weight = nn.Parameter(sobel_y, requires_grad=False)
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, output, target):
        """
        Compute Sobel edge loss between output and target.
        Args:
            output (torch.Tensor): Predicted image [batch_size, 1, H, W].
            target (torch.Tensor): Ground truth image [batch_size, 1, H, W].
        Returns:
            torch.Tensor: Loss value.
        """
        edge_x_out = self.sobel_x(output)
        edge_y_out = self.sobel_y(output)
        edge_x_tgt = self.sobel_x(target)
        edge_y_tgt = self.sobel_y(target)
        loss_x = self.mse(edge_x_out, edge_x_tgt)
        loss_y = self.mse(edge_y_out, edge_y_tgt)
        return loss_x + loss_y
    
        
class AdaptiveBinaryLoss(nn.Module):
    
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold
        self.bce_loss = nn.BCELoss(reduction='none')

    def forward(self, output, target):
       
        output = torch.clamp(output, 0, 1)
        
       
        pixel_loss = self.bce_loss(output, target)
        
       
        edge_weight = torch.abs(target - self.threshold) * 2 + 0.5
        
        
        weighted_loss = pixel_loss * edge_weight
        
        return weighted_loss.mean() 
        
class EnhancedEdgeLoss(nn.Module):
    
    
    def __init__(self, sobel_weight=1.0, lap_weight=0.5):
        super().__init__()
        self.sobel_weight = sobel_weight
        self.lap_weight = lap_weight
        
        # Sobel����
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        # ������˹��
        laplacian = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32)
        
        # ע��Ϊbuffer���������ݶȼ���
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
        self.register_buffer('laplacian', laplacian.view(1, 1, 3, 3))
        
    def forward(self, output, target):
        # ��ȡͨ���������ƾ�����
        channels = output.size(1)
        if channels > 1:
            sobel_x_kernel = self.sobel_x.repeat(channels, 1, 1, 1)
            sobel_y_kernel = self.sobel_y.repeat(channels, 1, 1, 1) 
            lap_kernel = self.laplacian.repeat(channels, 1, 1, 1)
            groups = channels
        else:
            sobel_x_kernel = self.sobel_x
            sobel_y_kernel = self.sobel_y
            lap_kernel = self.laplacian
            groups = 1
        
        # Sobel��Ե���
        sx_out = F.conv2d(output, sobel_x_kernel, padding=1, groups=channels)
        sy_out = F.conv2d(output, sobel_y_kernel, padding=1, groups=channels)
        sx_tgt = F.conv2d(target, sobel_x_kernel, padding=1, groups=channels)
        sy_tgt = F.conv2d(target, sobel_y_kernel, padding=1, groups=channels)
        
        # ���ڶ�ά�룬ʹ���ݶȷ�ֵ����Ч
        mag_out = torch.sqrt(sx_out**2 + sy_out**2 + 1e-6)
        mag_tgt = torch.sqrt(sx_tgt**2 + sy_tgt**2 + 1e-6)
        sobel_loss = F.l1_loss(mag_out, mag_tgt)
        
        # ������˹ϸ����ʧ
        lap_out = F.conv2d(output, lap_kernel, padding=1, groups=channels)
        lap_tgt = F.conv2d(target, lap_kernel, padding=1, groups=channels)
        lap_loss = F.l1_loss(lap_out, lap_tgt)
        
        return self.sobel_weight * sobel_loss + self.lap_weight * lap_loss
        
class QRLoss(nn.Module):
    
    def __init__(self, w_l1=1.0, w_edge=0.5, w_binary=0.1):
       
        super(QRLoss, self).__init__()
        #print(f"Initializing QRLoss with weights: L1={w_l1}, Edge={w_edge}, Binarization={w_binary}")
        self.l1_loss = L1Loss()
        self.edge_loss = EnhancedEdgeLoss()
        self.binary_loss = AdaptiveBinaryLoss()
        self.l1_weight = w_l1
        self.edge_weight = w_edge
        self.bianry_weight = w_binary

    def forward(self, pred, target):
       
        
        loss_l1 = self.l1_weight * self.l1_loss(pred, target)
       
        
       
        loss_edge = self.edge_weight * self.edge_loss(pred, target)
        
       
        loss_binary = self.bianry_weight * self.binary_loss(pred, target)
        
        
        total_loss = loss_l1 + loss_edge + loss_binary
        
        
        
        return total_loss







_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


# @weighted_loss
# def charbonnier_loss(pred, target, eps=1e-12):
#     return torch.sqrt((pred - target)**2 + eps)


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)

class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

