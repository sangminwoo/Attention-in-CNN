Attention-in-CNN
================================

- Backbone: ResNet
- Dataset: CIFAR100

Requirements
-----------
Install all the python dependencies using pip:
```
$ git clone https://github.com/sangminwoo/Attention-in-CNN.git
$ cd Attention-in-CNN
$ pip install -r requirements.txt
```
* PyTorch is not inside. Please go to [official website](https://pytorch.org/get-started/locally/).

Training
---------
Without Attention
```
$ python main.py --gpu GPUNUM --arch ARCHITECTURE
```

With Attention
```
$ python main.py --gpu GPUNUM --arch ARCHITECTURE --use_att --att_mode ATT
```
- **GPUNUM**: 0; 0,1; 0,3; 0,1,2; whatever
- **ARCHITECTURE**: resnet18(default), resnet34, resnet50, resnet101, resnet152 
- **ATT**: ours(default), se  

You can find more configurations in *main.py*.

Test
----------
Without Attention
```
$ python main.py --gpu GPUNUM --evaluate --resume RESUME
```

With Attention
```
$ python main.py --gpu GPUNUM --evaluate --resume RESUME --use_att
```
- **GPUNUM**: 0; 0,1; 0,3; 0,1,2; whatever
- **RESUME**: e.g., save/model_best.pth.tar (If you have changed save path, you should change resume path as well.)  

Visualization(Grad-CAM)
----------
Without Attention
```
$ python visualize.py 
```

With Attention
```
$ python visualize.py --use_att
```
