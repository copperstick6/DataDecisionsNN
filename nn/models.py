import torch
from torch import nn



class ConvNetModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.linear_layer_1 = nn.Linear(8, 64)
		self.sigmoid = nn.Sigmoid()
		self.linear_layer_2 = nn.Linear(64,64)
		self.relu = nn.ReLU(True)
		self.linear_layer_3 = nn.Linear(64,9)
		nn.init.constant_(self.linear_layer_2.weight, 0)
		nn.init.constant_(self.linear_layer_2.bias, 0)

	def forward(self, x):
		return self.linear_layer_3(self.sigmoid(self.linear_layer_2(self.relu(self.linear_layer_1(x)))))
