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
        intersection = torch.logical_and(pred_mask, gt_mask)
        union = torch.logical_or(pred_mask, gt_mask)
        epsilon = 1e-6
        iou = torch.sum(intersection) / (torch.sum(union)+epsilon)
        assert iou>=0 and iou<=1, f"IOU value is not in range {iou}"
        return iou
    
    def forward(self, pred_mask:torch.Tensor=None, gt_mask:torch.Tensor=None, weight=None):
        """ Compute the backpropable intersection over union loss between predicted and ground truth mask.
            Args:
                pred_mask: Predicted mask. Shape = (H,W,3)
                gt_mask: Ground truth mask. Shape = (H,W,3)

            Returns:
                iou_loss: Intersection over union loss"""
        
        iou = self.compute_IOU(pred_mask,gt_mask)
        loss = 1- iou

        # weight distance by weight
        if self.weighted:
            assert weight is not None
            loss = loss * weight

        return loss