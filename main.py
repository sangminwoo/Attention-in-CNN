import argparse
import os

from utils import save_checkpoint

def get_arguments():
	parser = argparse.ArgumentParser(description='Attention in Neural Networks')
	parser.add_argument('-b', '--batch', type=int, default=16)
	parser.add_argument('--gpu', type=str, help='0; 0,1; 0,3; etc', required=True)
	parser.add_argument('--root_dir', type=str, default='data/')
    parser.add_argument('--save_dir', type=str, default='save/')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--arch', type=str, default='resnet50', help='resnet18, 34, 50, 101, 152')
    parser.add_argument('--att_mode', type=str, default='channel', help='attention module mode: att_c(channel), att_s(spatial), att_cs(channel+spatial)')
    parser.add_argument('--use_att', action='store_true', help='use attention module')
    parser.add_argument('--no_pretrain', action='store_false', help='training from scratch')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    #parser.add_argument('--adjust-freq', type=int, default=40, help='learning rate adjustment frequency (default: 40)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--seed', default=1234, type=int, help='seed for initializing training. ')
    args = parser.parse_args()
    return args

def train(args):
    model = build_model(args)
    train_loader = model.data_loader(phase='train')
    val_loader = model.data_loader(phase='val')

    best_acc1 = 0
	for epoch in range(model.start_epoch, args.epochs):
    	#opt = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    	model.train(args, train_loader)

    	acc1, acc5 = model.validate(args, val_loader)
    	is_best = acc5 > best_acc5
    	best_acc5 = max(acc5, best_acc5)

    	save_checkpoint({
    		'epoch': epoch + 1,
    		'arch' : args.arch,
    		'state_dict': model.model.state_dict(),
    		'best_acc1': best_acc1,
    		'best_acc5': best_acc5,
    		'optimizer': model.optimizer.state_dict(),
    		}, is_best, args.save_dir)

def test(args, model):
    test_loader = model.data_loader(phase='test')
	model.test(self.args)

def main():
	args = get_arguments()

	if args.evaluate:
    	if not os.path.exists(self.args.save):
    		os.path.makedir(self.args.save)
		test(args)
		return

	train(args)

if __name__ == '__main__':
	main()