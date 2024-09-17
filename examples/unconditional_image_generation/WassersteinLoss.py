import torch
from torch.autograd import Function

class WassersteinLoss(Function):
   @staticmethod
   def forward(ctx, pred, target):
       # Sort the distributions
       sorted_pred, _ = torch.sort(pred)
       sorted_target, _ = torch.sort(target)
       
       # Pad the shorter distribution if necessary
       if len(sorted_pred) < len(sorted_target):
           sorted_pred = torch.nn.functional.pad(sorted_pred, (0, len(sorted_target) - len(sorted_pred)))
       elif len(sorted_target) < len(sorted_pred):
           sorted_target = torch.nn.functional.pad(sorted_target, (0, len(sorted_pred) - len(sorted_target)))
       
       # Compute Wasserstein distance
       loss = torch.mean(torch.abs(sorted_pred - sorted_target))
       
       ctx.save_for_backward(pred, target)
       return loss

   @staticmethod
   def backward(ctx, grad_output):
       pred, target = ctx.saved_tensors
       grad_pred = torch.zeros_like(pred)
       grad_target = None  # We don't need gradients w.r.t. target
       
       # Compute gradients
       grad_pred[pred > target] = 1
       grad_pred[pred < target] = -1
       
       return grad_pred * grad_output, grad_target