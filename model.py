import os
import random
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils import AverageMeter, accuracy
import resnet

class AttentionInCNN:
	def __init__(self, *args):
		self.args = args

		if args.seed is not None:
			random.seed(args.seed)
			np.random.seed(args.seed)
			torch.manual_seed(args.seed)
			cudnn.deterministic = True
			#cudnn.benchmark = False

		self.args.num_classes = 100 # CIFAR-100
		self.device = self.select_device(self.args.gpu)
		self.args.pretrain = not self.args.no_pretrain if not self.args.resume else self.args.no_pretrain
		self.model = self.select_model(self.args.arch, pretrained=self.args.pretrain,
			num_classes=self.args.num_classes, use_att=self.args.use_att, att_mode=self.args.att_mode)
	   	self.criterion = nn.CrossEntropyLoss().to(self.device)
	   	self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr,
	   		momentum=self.args.momentum, weight_deacy=self.args.weight_deacy)
	   	self.start_epoch = 0

	def select_device(self, gpu):
	    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
		if torch.cuda.is_available():
		   	self.device = torch.device('cuda')
		  	print(f'using {torch.cuda.device_count()} CUDA devices')
		else:
		   	self.device = torch.device('cpu')

	def select_model(self, arch, **kwargs):
		assert arch in {'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'}
	    if arch == 'resnet18':
	    	model = resnet.resnet18(*kwargs.items())
		elif arch == 'resnet34':
	    	model = resnet.resnet34(*kwargs.items())
	    elif arch == 'resnet50':
	    	model = resnet.resnet50(*kwargs.items())
	    elif arch == 'resnet101':
	    	model = resnet.resnet101(*kwargs.items())
	    elif arch == 'resnet152':
	    	model = resnet.resnet152(*kwargs.items())
	    model = nn.DataParallel(model.to(self.device))

	    print(model)
	    print(f'Number of model parameters: {sum([p.data.nelement() for p in model.parameters()])}')
	    return model

	def data_loader(self, phase):
		assert phase in {'train', 'val', 'test'}
		MEAN = [0.485, 0.456, 0.406]
   		STD = [0.229, 0.224, 0.225]

		if phase == 'train':
			img_transform = transforms.Compose([
									transforms.RandomResizedCrop(224),
									transforms.RandomHorizontalFlip(),
									transforms.ToTensor(),
									transforms.Normalize(mean=MEAN, std=STD)])
		else: # val, test
			img_transform = transforms.Compose([
									transforms.Resize(224),
									transforms.CenterCrop(224),
									transforms.ToTensor(),
									transforms.Normalize(mean=MEAN, std=STD)])

		dataset = datasets.CIFAR100(self.args.root_dir, train=True if phase=='train' else False, transform=img_transform, download=True)
		loader = DataLoader(dataset, batch_size=self.args.batch,
			shuffle=True if phase=='train' else False, num_workers=self.args.workers, pin_memory=True)
		return loader

	def resume(self):
   		if os.path.isfile(self.args.resume):
            print(f'=> loading checkpoint {self.args.resume}')
            checkpoint = torch.load(self.args.resume)
            self.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print(f'=> loaded checkpoint {self.args.resume} (epoch {checkpoint['epoch']})')
            print(f'=> best accuracy {best_acc1}')
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

    		#target = torch.from_numpy(np.asarray(target)).to(self.device)
    		target = torch.tensor(target).to(self.device)
    		output = model(input)

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
    			print(f'Epoch: [{self.epoch}][{i}/{len(train_loader)}]\t\
    				Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t\
                	Data {data_time.val:.3f} ({data_time.avg:.3f})\t\
                	Loss {loss.val:.4f} ({loss.avg:.4f})\t\
                	Acc@1 {top1.val:.3f} ({top1.avg:.3f})\
                	Acc@5 {top5.val:.3f} ({top5.avg:.3f})')

    def validate(self, data_loader):
    	batch_time = AverageMeter()
    	losses = AverageMeter()
    	top1 = AverageMeter()
    	top5 = AverageMeter()

    	self.model.eval()

    	with torch.no_grad():
	    	end = time.time()
	    	for i, (input, target) in enumerate(data_loader):
	    		if self.args.gpu is not None:
	    			input = input.to(self.device)

	    		#target = torch.from_numpy(np.asarray(target)).to(self.device)
	    		target = torch.tensor(target).to(self.device)
	    		output = model(input)

	    		loss = self.criterion(output[0], target)
	    		acc = accuracy(output[0], target)

	    		losses.update(loss.item(), input.size(0))
	    		top1.update(acc[0].item(), input.size(0))
	    		top5.update(acc[1].item(), input.size(0))

	    		batch_time.update(time.time() - end)
	    		end = time.time()

	    		if i % 10 == 0:
	    			print(f'Test: [{i}/{len(val_loader)}]\t\
	    				Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t\
	                	Loss {loss.val:.4f} ({loss.avg:.4f})\t\
	                	Acc@1 {top1.val:.3f} ({top1.avg:.3f})\
	                	Acc@5 {top5.val:.3f} ({top5.avg:.3f})')

    		print(f' * Acc@1 {top1.avg:.3f}\t *Acc@5 {top5.avg:.3f}')

    	return top1.avg, top5.avg

def build_model(args):
	return AttentionInCNN(args)