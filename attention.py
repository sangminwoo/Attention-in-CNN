import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelGate(nn.Module):

	def __init__(self, gate_channels, reduction_ratio=4):
		super(ChannelGate, self).__init__()
		self.gate_channels = gate_channels
		self.mlp = nn.Sequential(
			nn.Linear(gate_channels, gate_channels // reduction_ratio),
			nn.ReLU(),
			nn.Linear(gate_channels // reduction_ratio, gate_channels)
			)

	def forward(self, x):
		avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))).squeeze()
		channel_att_raw = self.mlp(avg_pool)
		channel_att = torch.sigmoid(channel_att_raw).unsqueeze(2).unsqueeze(3)
		return x * channel_att

class SpatialGate(nn.Module):

	def __init__(self, gate_channels):
		super(SpatialGate, self).__init__()
		self.spatial = nn.Sequential(
			nn.Conv2d(gate_channels, 1, kernel_size=7, padding=3),
			nn.Sigmoid()
			)

	def forward(self, x):
		scale = self.spatial(x)
		return x * scale

class AttentionModule(nn.Module):

	def __init__(self, gate_channels, reduction_ratio=16):
		super(AttentionModule, self).__init__()
		self.ChannelGate = ChannelGate(gate_channels, reduction_ratio)
		self.SpatialGate = SpatialGate(gate_channels)

	def forward(self, x):
		out = self.ChannelGate(x) + self.SpatialGate(x)
		return out

class SELayer(nn.Module):
	def __init__(self, gate_channels, reduction_ratio=16):
		super(SELayer, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.fc = nn.Sequential(
				nn.Linear(gate_channels, gate_channels // reduction_ratio),
				nn.ReLU(inplace=True),
				nn.Linear(channel // reduction, channel),
				nn.Sigmoid()
			)

	def forward(self, x):
		b, c, _, _ = x.size()
		y = self.avg_pool(x).view(b, c)
		y = self.fc(y).view(b, c, 1, 1)
		return x * y