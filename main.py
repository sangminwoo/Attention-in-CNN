import argparse
import os
import random

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.backends.cudnn as cudnn

from model import build_model
from data import CIFARLoader
from utils import save_checkpoint

def get_arguments():
	parser = argparse.ArgumentParser(description='Attention in Neural Networks')
	parser.add_argument('-b', '--batch', type=int, default=16)
	parser.add_argument('--gpu', type=str, help='0; 0,1; 0,3; etc', required=True)
	parser.add_argument('--root_dir', type=str, default='data')
	parser.add_argument('--save_dir', type=str, default='save')
	parser.add_argument('--resume', type=str, default=None)
	parser.add_argument('--lr', type=float, default=2e-4)
	parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
	parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
	parser.add_argument('--arch', type=str, default='resnet50', help='resnet18, 34, 50, 101, 152')
	parser.add_argument('--use_att', action='store_true', help='use attention module')
	parser.add_argument('--att_mode', type=str, default='ours', help='attention module mode: se, ours')
	parser.add_argument('--pretrain', default=True, help='load pretrained model. If False, training from scratch')
	parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
	parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
	parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
	parser.add_argument('--seed', default=1234, type=int, help='seed for initializing training. ')
	args = parser.parse_args()
	return args

def train(args):
	model = build_model(args)
	data_loader = CIFARLoader(args)
	train_loader, val_loader = data_loader.get_train_valid_loader(shuffle=True, valid_size=0.1)

	best_acc1 = 0; 	best_acc5 = 0
	
	scheduler = ReduceLROnPlateau(optimizer=model.optimizer, mode='min', patience=1, verbose=True)
	
	for epoch in range(model.start_epoch, args.epochs):

		model.train(train_loader)

		acc1, acc5, loss = model.validate(val_loader)
		is_best = acc5 > best_acc5
		best_acc5 = max(acc5, best_acc5)

		scheduler.step(loss)

		save_checkpoint({
			'epoch': epoch + 1,
			'arch' : args.arch,
			'state_dict': model.model.state_dict(),
			'best_acc1': best_acc1,
			'best_acc5': best_acc5,
			'optimizer': model.optimizer.state_dict(),
			}, is_best, args.save_dir)

def test(args):
	model = build_model(args)
	data_loader = CIFARLoader(args)
	test_loader = data_loader.get_test_loader()
	model.test(test_loader)

	#model.test(test_loader)

def set_device(gpu):
	os.environ['CUDA_VISIBLE_DEVICES'] = gpu
	
	if torch.cuda.is_available():
		device = torch.device('cuda')
		print(f'using {torch.cuda.device_count()} CUDA devices')
	else:
		device = torch.device('cpu')
		print(f'using CPU')
	
	return device

def main():
	args = get_arguments()

	if args.seed is not None:
		random.seed(args.seed)
		#np.random.seed(args.seed)
		torch.manual_seed(args.seed)
		cudnn.deterministic = True
		#cudnn.benchmark = False

	args.device = set_device(args.gpu)
	
	if not os.path.exists(args.save_dir):
		os.makedirs(args.save_dir)

	if args.evaluate:
		if not args.resume:
			args.resume = os.path.join(args.save_dir, 'model_best.pth.tar')
		test(args)
		return

	train(args)

if __name__ == '__main__':
	main()