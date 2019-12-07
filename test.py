import os
import torch
import torch.nn as nn

import ResNet


model = ResNet.resnet(arch='resnet50', pretrained=False, num_classes=100,
		use_att=True, att_mode='ours')
#model = nn.DataParallel(model)

resume = 'save/model_best_att.pth.tar'

if os.path.isfile(resume):
	print(f'=> loading checkpoint {resume}')
	checkpoint = torch.load(resume)
	best_acc5 = checkpoint['best_acc5']
	model.load_state_dict(checkpoint['state_dict'], strict=False)
	print(f"=> loaded checkpoint {resume} (epoch {checkpoint['epoch']})")
	print(f'=> best accuracy {best_acc5}')
else:
	print(f'=> no checkpoint found at {resume}')

#print(model)

for modules in model._modules:
	print(modules)

print(model._modules['att'])

print(resnet._modules)