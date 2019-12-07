import torch
import numpy as np

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

class CIFARLoader:
	def __init__(self, args):
		self.args = args

	def get_dataset(self, phase):
		assert phase in {'train', 'val', 'test'}
		MEAN = [0.485, 0.456, 0.406] #MEAN = [0.4914, 0.4822, 0.4465]
		STD = [0.229, 0.224, 0.225] #STD = [0.2023, 0.1994, 0.2010]
		
		if phase == 'train':
			img_transform = transforms.Compose([
									transforms.RandomResizedCrop(224),
									transforms.RandomHorizontalFlip(),
									transforms.ToTensor(),
									transforms.Normalize(mean=MEAN, std=STD)])
		elif phase == 'val':
			img_transform = transforms.Compose([
									transforms.Resize(224),
									transforms.CenterCrop(224),
									transforms.ToTensor(),
									transforms.Normalize(mean=MEAN, std=STD)])
		elif phase == 'test':
			img_transform = transforms.Compose([
									transforms.Resize(224),
									transforms.CenterCrop(224),
									transforms.ToTensor(),
									transforms.Normalize(mean=MEAN, std=STD)])

		dataset = datasets.CIFAR100(self.args.root_dir, train=False if phase=='test' else True,
									transform=img_transform, download=True)
		return dataset

	def get_train_valid_loader(self, shuffle=True, valid_size=0.1):
		train_dataset = self.get_dataset(phase='train')
		valid_dataset = self.get_dataset(phase='val')

		num_train = len(train_dataset)
		indices = list(range(num_train))
		split = int(np.floor(valid_size * num_train))

		if shuffle:
			np.random.shuffle(indices)

		train_idx, valid_idx = indices[split:], indices[:split]
		train_sampler = SubsetRandomSampler(train_idx)
		valid_sampler = SubsetRandomSampler(valid_idx)

		train_loader = DataLoader(train_dataset, batch_size=self.args.batch, sampler=train_sampler,
			num_workers=self.args.workers, pin_memory=True)

		valid_loader = DataLoader(valid_dataset, batch_size=self.args.batch, sampler=valid_sampler,
			num_workers=self.args.workers, pin_memory=True)

		return train_loader, valid_loader

	def get_test_loader(self):
		test_dataset = self.get_dataset(phase='test')

		test_loader = DataLoader(test_dataset, batch_size=self.args.batch, shuffle=False,
        	num_workers=self.args.workers, pin_memory=True)

		return test_loader