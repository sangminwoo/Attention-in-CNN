import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils import AverageMeter, accuracy
from ResNet import resnet

class AttentionInCNN:
	def __init__(self, args):
		self.args = args
		self.device = args.device
		num_classes = 100 # CIFAR-100
		pretrain = args.pretrain and self.args.resume is None

		self.model = self.select_model(args.arch, pretrained=pretrain,
			num_classes=num_classes, use_att=self.args.use_att, att_mode=self.args.att_mode)
		self.criterion = nn.CrossEntropyLoss().to(self.device)
		self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr,
			momentum=args.momentum, weight_decay=args.weight_decay)

		self.start_epoch = 0
		self.epoch_count = 0

		if self.args.resume:
			self.resume()

	def select_model(self, arch, pretrained, **kwargs):
		assert arch in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
		
		model = resnet(arch=arch, pretrained=pretrained, **kwargs)
		model = nn.DataParallel(model.to(self.device))

		print(model)
		print(f'Number of model parameters: {sum([p.data.nelement() for p in model.parameters()])}')
		return model

	def resume(self):
		if os.path.isfile(self.args.resume):
			print(f'=> loading checkpoint {self.args.resume}')
			checkpoint = torch.load(self.args.resume)
			self.start_epoch = checkpoint['epoch']
			best_acc5 = checkpoint['best_acc5']
			self.model.load_state_dict(checkpoint['state_dict'])
			self.optimizer.load_state_dict(checkpoint['optimizer'])
			print(f"=> loaded checkpoint {self.args.resume} (epoch {checkpoint['epoch']})")
			print(f'=> best accuracy {best_acc5}')
		else:
			print(f'=> no checkpoint found at {self.args.resume}')

	def train(self, train_loader):
		batch_time = AverageMeter()
		data_time = AverageMeter()
		losses = AverageMeter()
		top1 = AverageMeter()
		top5 = AverageMeter()

		self.model.train()

		end = time.time()
		for i, (input, target) in enumerate(train_loader):
			data_time.update(time.time() - end)

			if self.args.gpu is not None:
				input = input.to(self.device)

			target = torch.from_numpy(np.asarray(target)).to(self.device)
			output = self.model(input)

			loss = self.criterion(output[0], target)
			acc = accuracy(output[0], target)

			losses.update(loss.item(), input.size(0))
			top1.update(acc[0].item(), input.size(0))
			top5.update(acc[1].item(), input.size(0))

			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

			batch_time.update(time.time() - end)
			end = time.time()

			if i % 10 == 0:
				print(f'Epoch: [{self.epoch_count}][{i}/{len(train_loader)}]\t'
					  f'Batch {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					  f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
					  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
					  f'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
					  f'Acc@5 {top5.val:.3f} ({top5.avg:.3f})')
		self.epoch_count += 1

	def validate(self, valid_loader):
		batch_time = AverageMeter()
		losses = AverageMeter()
		top1 = AverageMeter()
		top5 = AverageMeter()

		self.model.eval()

		with torch.no_grad():
			end = time.time()
			for i, (input, target) in enumerate(valid_loader):
				if self.args.gpu is not None:
					input = input.to(self.device)

				target = torch.from_numpy(np.asarray(target)).to(self.device)
				output = self.model(input)

				loss = self.criterion(output[0], target)
				acc = accuracy(output[0], target)

				losses.update(loss.item(), input.size(0))
				top1.update(acc[0].item(), input.size(0))
				top5.update(acc[1].item(), input.size(0))

				batch_time.update(time.time() - end)
				end = time.time()

				if i % 10 == 0:
					print(f'Valid: [{i}/{len(valid_loader)}]\t'
						  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
						  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
						  f'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
						  f'Acc@5 {top5.val:.3f} ({top5.avg:.3f})')

			print(f' * Acc@1 {top1.avg:.3f}\t *Acc@5 {top5.avg:.3f}')

		return top1.avg, top5.avg, loss

	def test(self, test_loader):
		batch_time = AverageMeter()
		losses = AverageMeter()
		top1 = AverageMeter()
		top5 = AverageMeter()

		self.model.eval()

		with torch.no_grad():
			end = time.time()
			for i, (input, target) in enumerate(test_loader):
				if self.args.gpu is not None:
					input = input.to(self.device)

				target = torch.from_numpy(np.asarray(target)).to(self.device)
				output = self.model(input)

				loss = self.criterion(output[0], target)
				acc = accuracy(output[0], target)

				losses.update(loss.item(), input.size(0))
				top1.update(acc[0].item(), input.size(0))
				top5.update(acc[1].item(), input.size(0))

				batch_time.update(time.time() - end)
				end = time.time()

				if i % 10 == 0:
					print(f'Test: [{i}/{len(test_loader)}]\t'
						  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
						  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
						  f'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
						  f'Acc@5 {top5.val:.3f} ({top5.avg:.3f})')

			print(f' * Acc@1 {top1.avg:.3f}\t *Acc@5 {top5.avg:.3f}')

def build_model(args):
	return AttentionInCNN(args)