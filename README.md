# Network Recasting
Source code of the paper
> [Network Recasting: A Universal Method for Network Architecture Transformation](https://arxiv.org/abs/1809.05262)\
> Joonsang Yu, Sungbum Kang, Kiyoung Choi.\
> AAAI-19
> _arXiv:1809.05262_.


## Requirements
```
Python >= 3.5.5, PyTorch == 0.3.1, torchvision == 0.2.0
```
NOTE: We found that our algorithm cannot be converged in NVIDIA TensorCores becasue its precision is lower than conventional float 32. We recommand install PyTorch with CUDNN 7.0.05 version to turn off TensorCores.

## Datasets

We use CIFAR and ImageNet (ILSVRC2012) dataset. CIFAR dataset is automatically downloaded by torchvision, but ImageNet have to be manually downloaded to run recasting.
To use ImageNet dataset, enviroment variable setting is needed.
```
export IMAGENET='your path'
```

## Network recasting
To carry out recasting result, run

```
cd script && ./run_recast_resnet56_into_convnet_cifar10.sh		# ResNet56 -> ConvNet in CIFAR-10
cd script && ./run_recast_densenet100_into_resnet_cifar100.sh	# DenseNet100 -> ResNet in CIFAR-100
```

We provide several sample examples and those have same naming convention with above examples. So, we think that you can easily understand the experimental setting.
We also provide "py" and "ipynb" for every sample experiment, and our temporal experimental result is included in ipynb files. Futher detail of experiment is included in those files.


## Citation
```
@article{yu2019network,
	title={Network Recasting: A Universal Method for Network Architecture Transformation},
	author={Yu, Joonsang and Kang, Sungbum and Choi, Kiyoung},
	journal={The Thirty-Third AAAI Conference on Artificial Intelligence (AAAI-19)},
	year={2019}
}
```
