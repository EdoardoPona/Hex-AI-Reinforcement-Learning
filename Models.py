import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.autograd import Variable


class ConvNetTwoSides(nn.Module):
	""" a model that can in theory play both sides, will pick a side depending on the turn
	vector """
	def __init__(self):
		super(ConvNetTwoSides, self).__init__()
		self.conv0 = nn.Conv2d(1, 5, 3, stride=(1, 1))
		self.conv1 = nn.Conv2d(5, 8, 3, stride=(1, 1))

		self.dense00 = nn.Linear(8*4*4, 128)  
		self.dense01 = nn.Linear(10, 128)
		self.dense1 = nn.Linear(128, 64)

	def forward(self, x, turn_vector):
		h0 = F.relu(self.conv1(self.conv0(x)).view((-1, 8*4*4)))
		h1 = F.relu(self.dense00(h0) + self.dense01(turn_vector))
		return self.dense1(h1)


class ConvNetOneSide(nn.Module):
	""" will always play the same side """
	def __init__(self):
		super(ConvNetOneSide, self).__init__()
		self.conv0 = nn.Conv2d(1, 5, 3, stride=(1, 1))
		self.conv1 = nn.Conv2d(5, 8, 3, stride=(1, 1))
		self.dense0 = nn.Linear(8*4*4, 64)

	def forward(self, x):
		h0 = F.relu(self.conv1(F.relu(self.conv0(x)))).view(-1, 8*4*4)
		return self.dense0(h0)


class PolicyOneSide(nn.Module):
	""" wrapper for the model for training while playing (rewards and logprobs) """
	def __init__(self):
		super(PolicyOneSide, self).__init__()
		self.model = ConvNetOneSide()
		self.rewards = []
		self.log_probs = []

	def filter(self, x, available_moves):
		""" getting only the valid moves """
		return torch.cat([x[:, i] for i in available_moves], dim=0).unsqueeze(0)

	def forward(self, x, available_moves):
		return F.softmax(self.filter(self.model(x), available_moves), dim=1)


class PolicyTwoSides(nn.Module):

	def __init__(self):
		super(PolicyTwoSides, self).__init__()
		self.model = ConvNetTwoSides()

	def filter(self, x, available_moves):
		""" getting only the valid moves """
		return torch.cat([x[:, i] for i in available_moves], dim=0).unsqueeze(0)

	def forward(self, x, available_moves, turn_vector):
		return F.softmax(self.filter(self.model(x, turn_vector), available_moves), dim=1)
