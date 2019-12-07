import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from attention import AttentionModule, SELayer

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(inplanes, outplanes, stride=1):
	"""3x3 convolution with padding"""
	return nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, stride=1, downsample=None, use_att=False, att_mode='ours'):
		super(BasicBlock, self).__init__()
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = nn.BatchNorm2d(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = nn.BatchNorm2d(planes)
		self.downsample = downsample
		self.stride = stride
		self.use_att = use_att

		if use_att:
			assert att_mode in ['ours', 'se']
			if att_mode == 'ours':
				self.att = AttentionModule(planes)
			elif att_mode == 'se':
				self.att = SELayer(planes)

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		
		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample:
			residual = self.downsample(x)

		if self.use_att:
			out = self.att(out)

		out += residual
		out = self.relu(out)

		return out

class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, stride=1, downsample=None, use_att=False, att_mode='ours'):
		super(Bottleneck, self).__init__()
		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(planes * self.expansion)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride
		self.use_att = use_att

		if use_att:
			assert att_mode in ['ours', 'se']
			if att_mode == 'ours':
				self.att = AttentionModule(planes * self.expansion)
			elif att_mode == 'se':
				self.att = SELayer(planes * self.expansion)

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample:
			residual = self.downsample(x)

		if self.use_att:
			out = self.att(out)

		out += residual
		out += self.relu(out)

		return out

class ResNet(nn.Module):

	def __init__(self, block, layers, num_classes=1000, use_att=False, att_mode='ours'):
		self.inplanes = 64
		super(ResNet, self).__init__()
		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

		self.layer1 = self._make_layer(block, 64, layers[0], stride=1, use_att=False, att_mode=att_mode)
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2, use_att=False, att_mode=att_mode)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2, use_att=False, att_mode=att_mode)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2, use_att=False, att_mode=att_mode)
		self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)

		self.final_fc = nn.Linear(512 * block.expansion, num_classes)

		self.use_att = use_att
		if use_att:
			self.att = AttentionModule(512*block.expansion)
		self.num_classes = num_classes
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def _make_layer(self, block, planes, blocks, stride=1, use_att=False, att_mode='ours'):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion,
						kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(planes * block.expansion),
				)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample, use_att=use_att, att_mode=att_mode))
		self.inplanes = planes * block.expansion

		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes, use_att=use_att, att_mode=att_mode))

		return nn.Sequential(*layers)

	def forward(self, x):
		imgs = x
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x4 = self.layer4(x)

		if self.use_att:
			x4 = self.att(x4)

		x = self.avgpool(x4)
		x = x.view(x.size(0), -1)
		return x, x4

def resnet(arch='resnet50', pretrained=False, **kwargs):
	assert arch in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

	if arch == 'resnet18':
		layers = [2, 2, 2, 2]
	elif arch == 'resnet34':
		layers = [3, 4, 6, 3]
	elif arch == 'resnet50':
		layers = [3, 4, 6, 3]
	elif arch == 'resnet101':
		layers = [3, 4, 23, 3]
	elif arch == 'resnet152':
		layers = [3, 8, 36, 3]

	model = ResNet(BasicBlock, layers, **kwargs)

	if pretrained:
		pretrained_dict = model_zoo.load_url(model_urls[arch])
		model_dict = model.state_dict()
		pretrained_dict = {k : v for k, v in pretrained_dict.items() if k in model_dict}
		model_dict.update(pretrained_dict)
		model.load_state_dict(model_dict)
		print(f'pre-trained model {arch} loaded successfully')

	return model