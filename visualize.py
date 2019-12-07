import os
import PIL
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.utils import make_grid, save_image

import ResNet
from utils import visualize_cam, Normalize
from gradcam import GradCAM, GradCAMpp

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--type', type=str, default='resnet', help='alexnet, vgg, resnet, densenet, squeezenet')
	parser.add_argument('--root_dir', type=str, default='gradcam/data')
	parser.add_argument('--save_dir', type=str, default='gradcam/save')
	parser.add_argument('--use_att', action='store_true', help='use attention module')
	parser.add_argument('--att_mode', type=str, default='ours', help='attention module mode: se, ours')
	parser.add_argument('--resume', type=str, default='save/model_best.pth.tar')
	args = parser.parse_args()
	return args

def get_model_dict(model, type):
	assert type in ['alexnet', 'vgg', 'resnet', 'densenet', 'squeezenet']

	if type == 'alexnet':
		alexnet = models.alexnet(pretrained=True)
		alexnet.eval()
		model_dict = dict(type='alexnet', arch=alexnet, layer_name='features_11', input_size=(224, 224))

	elif type == 'vgg':
		vgg = models.vgg16(pretrained=True)
		vgg.eval()
		model_dict = dict(type='vgg', arch=vgg, layer_name='features_29', input_size=(224, 224))

	elif type == 'resnet':
		resnet = models.resnet50(pretrained=True)
		resnet.eval()
		model_dict = dict(type='resnet', arch=resnet, layer_name='layer4', input_size=(224, 224))

	elif type == 'densenet':
		densenet = models.densenet161(pretrained=True)
		densenet.eval()
		model_dict = dict(type='densenet', arch=densenet, layer_name='features_norm5', input_size=(224, 224))

	elif type == 'squeezenet':
		squeezenet = models.squeezenet1_1(pretrained=True)
		squeezenet.eval()
		model_dict = dict(type='squeezenet', arch=squeezenet, layer_name='features_12_expand3x3_activation', input_size=(224, 224))

	return model_dict

def main():
	args = get_args()
	root_dir = args.root_dir
	imgs = list(os.walk(root_dir))[0][2]

	save_dir = args.save_dir
	num_classes = 100 # CIFAR100
	model = ResNet.resnet(arch='resnet50', pretrained=False, num_classes=num_classes,
		use_att=args.use_att, att_mode=args.att_mode)
	#model = nn.DataParallel(model)
	#print(model)

	if args.resume:
		if os.path.isfile(args.resume):
			print(f'=> loading checkpoint {args.resume}')
			checkpoint = torch.load(args.resume)
			best_acc5 = checkpoint['best_acc5']
			model.load_state_dict(checkpoint['state_dict'], strict=False)
			print(f"=> loaded checkpoint {args.resume} (epoch {checkpoint['epoch']})")
			print(f'=> best accuracy {best_acc5}')
		else:
			print(f'=> no checkpoint found at {args.resume}')

	model_dict = get_model_dict(model, args.type)
	normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

	for img_name in imgs:
		img_path = os.path.join(root_dir, img_name)
		pil_img = PIL.Image.open(img_path)
	
		torch_img = torch.from_numpy(np.asarray(pil_img))
		torch_img = torch_img.permute(2, 0, 1).unsqueeze(0)
		torch_img = torch_img.float().div(255)
		torch_img = F.interpolate(torch_img, size=(224, 224), mode='bilinear', align_corners=False)

		normalized_torch_img = normalizer(torch_img)

		gradcam = GradCAM(model_dict, True)
		gradcam_pp = GradCAMpp(model_dict, True)

		mask, _ = gradcam(normalized_torch_img)
		heatmap, result = visualize_cam(mask, torch_img)

		mask_pp, _ = gradcam_pp(normalized_torch_img)
		heatmap_pp, result_pp = visualize_cam(mask_pp, torch_img)
		
		images = torch.stack([torch_img.squeeze().cpu(), heatmap, heatmap_pp, result, result_pp], 0)

		images = make_grid(images, nrow=1)

		if args.use_att:
			save_dir = os.path.join(args.save_dir, 'att')
		else:
			save_dir = os.path.join(args.save_dir, 'no_att')

		os.makedirs(save_dir, exist_ok=True)
		output_name = img_name
		output_path = os.path.join(save_dir, output_name)

		save_image(images, output_path)

if __name__=='__main__':
	main()