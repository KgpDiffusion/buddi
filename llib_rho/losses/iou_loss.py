import torch 
import torch.nn as nn

class IOULoss(nn.Module):
    def __init__(
        self,
        weighted: bool = False,
        d1_aggregation: str = 'mean',
        **kwargs
    ):
        super().__init__()
       
        self.weighted = weighted

    def compute_IOU(self,pred_mask:torch.Tensor,gt_mask:torch.Tensor):
        """ Compute the intersection over union between predicted and ground truth mask.
            Args:
                pred_mask: Predicted mask. Shape = (H,W,3)
                gt_mask: Ground truth mask. Shape = (H,W,3)

            Returns:
                iou: Intersection over union"""
        
        assert pred_mask.shape == gt_mask.shape, f"Shape of predicted and ground truth mask is not same {pred_mask.shape} {gt_mask.shape}"
        intersection = torch.sum((pred_mask)*(gt_mask))
        union = torch.sum((pred_mask)) + torch.sum((gt_mask)) - intersection
        epsilon = 1e-6
        iou = intersection / (union+epsilon)
        assert iou>=0 and iou<=1, f"IOU value is not in range {iou}"
        return iou
    
    def compute_soft_centroid(self,mask:torch.Tensor):
        """ Compute the centroid index from soft mask.
            Args:
                mask: Soft mask. Shape = (H,W,3)

            Returns:
                centroid: Soft Centroid index"""

        # compute centroid
        H,W = mask.shape

        centroid = torch.meshgrid(torch.arange(H),torch.arange(W))
        centroid_x = torch.sum(torch.Tensor(centroid[0]).to(mask.device).float()*mask)/torch.sum(mask)
        centroid_y = torch.sum(torch.Tensor(centroid[1]).to(mask.device).float()*mask)/torch.sum(mask)

        return centroid_x, centroid_y

    def forward(self, pred_mask:torch.Tensor=None, gt_mask:torch.Tensor=None, weight=None):
        """ Compute the backpropable intersection over union loss between predicted and ground truth mask.
            Args:
                pred_mask: Predicted mask. Shape = (H,W,3)
                gt_mask: Ground truth mask. Shape = (H,W,3)

            Returns:
                iou_loss: Intersection over union loss"""
                
        iou = self.compute_IOU(pred_mask,gt_mask)
        x_pred, y_pred = self.compute_soft_centroid(pred_mask)
        x_gt, y_gt = self.compute_soft_centroid(gt_mask)
        distance_n = torch.sqrt((x_pred - x_gt)**2 + (y_pred - y_gt)**2)/max(pred_mask.shape)
        w = 25
        loss = (1- iou) + w*distance_n

        # weight distance by weight
        if self.weighted:
            assert weight is not None
            loss = loss * weight

        return loss