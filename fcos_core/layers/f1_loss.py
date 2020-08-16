import torch
from torch import nn
import torch.nn.functional as F

class F1_Loss(nn.Module):
	def __init__(self, num_classes, epsilon=1e-7):
		super().__init__()
		self.epsilon = epsilon
		self.num_classes = num_classes

	def forward(self, pred, target):
		assert pred.dim() == 2
		assert target.dim() == 1

		pred = torch.softmax(pred, dim=1)
		target = F.one_hot(target, self.num_classes).to(torch.float32)

		tp = (target * pred).sum(dim=0).to(torch.float32)
		tn = ((1 - target) * (1 - pred)).sum(dim=0).to(torch.float32)
		fp = ((1 - target) * pred).sum(dim=0).to(torch.float32)
		fn = (target * (1 - pred)).sum(dim=0).to(torch.float32)

		precision = tp / (tp + fp + self.epsilon)
		recall = tp / (tp + fn + self.epsilon)

		f1 = 2* (precision*recall) / (precision + recall + self.epsilon)
		f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)
		f1 =  1 - f1.mean()

		return f1