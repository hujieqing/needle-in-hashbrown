import torch
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F


class DistanceLoss(BCEWithLogitsLoss):
	__constants__ = ['lambda1', 'lambda2']

	def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None, lambda1=1.0, lambda2=0.0, device=torch.device('cpu')):
		super(DistanceLoss, self).__init__(weight, size_average, reduce, reduction, pos_weight)

		self.lambda1 = lambda1
		self.lambda2 = lambda2
		self.device = device

	def forward(self, input, target, input2, target2):
		result = self.lambda1 * F.binary_cross_entropy_with_logits(input.to(self.device), target, self.weight, pos_weight=self.pos_weight, reduction=self.reduction)
		result += self.lambda2 * F.mse_loss(input2.to(self.device), target2, reduction=self.reduction)
		return result

