# PGD-pytorch
**A pytorch implementation of "[Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083)"**

## Summary
This code is a pytorch implementation of **PGD attack**   
In this code, I used above methods to fool [Inception v3](https://arxiv.org/abs/1512.00567).   
'[Giant Panda](http://www.image-net.org/)' used for an example.   
You can add other pictures with a folder with the label name in the 'data/imagenet'.    

## Requirements
* python==3.6   
* numpy==1.14.2   
* pytorch==1.0.1   

## Important results not in the code
- Capacity(size of network) plays an important role in adversarial training. (p.9-10)
	- For only natural examples training, it increases the robustness against one-step perturbations.
	- For PGD adversarial training, small capacity networks fails.
	- As capacity increases, the model can fit the adversairal examples increasingly well.
	- More capacity and strong adversaries decrease transferability. (Section B)
- FGSM adversaries don't increase robustness for large epsilon(=8). (p.9-10)
	- The network overfit to FGSM adversarial examples.
- Adversarial training with PGD shows good enough defense results.(p.12-13)

## Notice
- This Repository won't be updated.
- Please check [the package of adversarial attacks in pytorch](https://github.com/Harry24k/adversairal-attacks-pytorch)

## Experiments

### How to run experiments for CIFAR10

#### No mixup Preactresnet18
```
python pgd.py --dataset cifar10 --data_dir data/cifar10/ --root_dir experiments/ --labels_per_class 5000 --arch preactresnet18 --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 2000 --schedule 500 1000 1500 --gammas 0.1 0.1 0.1 --train vanilla 
```

#### Mixup Preactresnet18

```
python pgd.py --dataset cifar10 --data_dir data/cifar10/ --root_dir experiments/ --labels_per_class 5000 --arch preactresnet18 --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 2000 --schedule 500 1000 1500 --gammas 0.1 0.1 0.1 --train mixup --mixup_alpha 1.0
```

#### Manifold mixup Preactresnet18
```
python pgd.py --dataset cifar10 --data_dir data/cifar10/ --root_dir experiments/ --labels_per_class 5000 --arch preactresnet18 --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 2000 --schedule 500 1000 1500 --gammas 0.1 0.1 0.1 --train mixup_hidden --mixup_alpha 2.0
```

#### No mixup Preactresnet34
```
python pgd.py --dataset cifar10 --data_dir data/cifar10/ --root_dir experiments/ --labels_per_class 5000 --arch preactresnet34 --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 2000 --schedule 500 1000 1500 --gammas 0.1 0.1 0.1 --train vanilla 
```

#### Mixup Preactresnet34

```
python pgd.py --dataset cifar10 --data_dir data/cifar10/ --root_dir experiments/ --labels_per_class 5000 --arch preactresnet34 --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 2000 --schedule 500 1000 1500 --gammas 0.1 0.1 0.1 --train mixup --mixup_alpha 1.0
```

#### Manifold mixup Preactresnet34
```
python pgd.py --dataset cifar10 --data_dir data/cifar10/ --root_dir experiments/ --labels_per_class 5000 --arch preactresnet34 --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 2000 --schedule 500 1000 1500 --gammas 0.1 0.1 0.1 --train mixup_hidden --mixup_alpha 2.0 
```

#### No mixup WRN-28-10
```
python pgd.py --dataset cifar10 --data_dir data/cifar10/ --root_dir experiments/ --labels_per_class 5000 --arch wrn28_10 --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 400 --schedule 200 300 --gammas 0.1 0.1 --train vanilla
```

#### Mixup WRN-28-10

```
python pgd.py --dataset cifar10 --data_dir data/cifar10/ --root_dir experiments/ --labels_per_class 5000 --arch wrn28_10 --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 400 --schedule 200 300 --gammas 0.1 0.1 --train mixup --mixup_alpha 1.0
```

#### Manifold mixup WRN-28-10
```
python pgd.py --dataset cifar10 --data_dir data/cifar10/ --root_dir experiments/ --labels_per_class 5000 --arch wrn28_10 --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 400 --schedule 200 300 --gammas 0.1 0.1 --train mixup_hidden --mixup_alpha 2.0
```

### How to run experiments for MNIST

#### No mixup LeNet5
```
python pgd.py --dataset mnist --data_dir data/mnist/ --root_dir experiments/ --labels_per_class 5000 --arch lenet5 --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 2000 --schedule 500 1000 1500 --gammas 0.1 0.1 0.1 --train vanilla
```

#### Mixup LeNet5

```
python pgd.py --dataset mnist --data_dir data/mnist/ --root_dir experiments/ --labels_per_class 5000 --arch lenet5 --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 2000 --schedule 500 1000 1500 --gammas 0.1 0.1 0.1 --train mixup --mixup_alpha 1
```

#### Manifold mixup LeNet5

```
python pgd.py --dataset mnist --data_dir data/mnist/ --root_dir experiments/ --labels_per_class 5000 --arch lenet5 --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 2000 --schedule 500 1000 1500 --gammas 0.1 0.1 0.1 --train mixup_hidden --mixup_alpha 2.0
```



### How to run experiments for CIFAR100

#### No mixup Preactresnet18
```
python pgd.py --dataset cifar100 --data_dir data/cifar100/ --root_dir experiments/ --labels_per_class 500 --arch preactresnet18 --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 2000 --schedule 500 1000 1500 --gammas 0.1 0.1 0.1 --train vanilla 
```

#### Mixup Preactresnet18

```
python pgd.py --dataset cifar100 --data_dir data/cifar100/ --root_dir experiments/ --labels_per_class 500 --arch preactresnet18 --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 2000 --schedule 500 1000 1500 --gammas 0.1 0.1 0.1 --train mixup --mixup_alpha 1.0
```

#### Manifold mixup Preactresnet18
```
python pgd.py --dataset cifar100 --data_dir data/cifar100/ --root_dir experiments/ --labels_per_class 500 --arch preactresnet18 --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 2000 --schedule 500 1000 1500 --gammas 0.1 0.1 0.1 --train mixup_hidden --mixup_alpha 2.0
```

#### No mixup Preactresnet34
```
python pgd.py --dataset cifar100 --data_dir data/cifar100/ --root_dir experiments/ --labels_per_class 500 --arch preactresnet34 --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 2000 --schedule 500 1000 1500 --gammas 0.1 0.1 0.1 --train vanilla
```

#### Mixup Preactresnet34
```
python pgd.py --dataset cifar100 --data_dir data/cifar100/ --root_dir experiments/ --labels_per_class 500 --arch preactresnet34 --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 2000 --schedule 500 1000 1500 --gammas 0.1 0.1 0.1 --train mixup --mixup_alpha 1.0
```

#### Manifold mixup Preactresnet34
```
python pgd.py --dataset cifar100 --data_dir data/cifar100/ --root_dir experiments/ --labels_per_class 500 --arch preactresnet34 --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 2000 --schedule 500 1000 1500 --gammas 0.1 0.1 0.1 --train mixup_hidden --mixup_alpha 2.0
```

#### No mixup WRN-28-10
```
python pgd.py --dataset cifar100 --data_dir data/cifar100/ --root_dir experiments/ --labels_per_class 500 --arch wrn28_10 --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 400 --schedule 200 300 --gammas 0.1 0.1 --train vanilla
```

#### Mixup WRN-28-10

```
python pgd.py --dataset cifar100 --data_dir data/cifar100/ --root_dir experiments/ --labels_per_class 500 --arch wrn28_10 --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 400 --schedule 200 300 --gammas 0.1 0.1 --train mixup --mixup_alpha 1.0
```

#### Manifold mixup WRN-28-10
```
python pgd.py --dataset cifar100 --data_dir data/cifar109/ --root_dir experiments/ --labels_per_class 500 --arch wrn28_10 --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 400 --schedule 200 300 --gammas 0.1 0.1 --train mixup_hidden --mixup_alpha 2.0
```


### How to run experiments for SVHN

#### No mixup Preactresnet18
```
python pgd.py --dataset svhn --data_dir data/svhn/ --root_dir experiments/ --labels_per_class 7325 --arch preactresnet18 --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 2000 --schedule 500 1000 1500 --gammas 0.1 0.1 0.1 --train vanilla
```

#### Mixup Preactresnet18

```
python pgd.py --dataset svhn --data_dir data/svhn/ --root_dir experiments/ --labels_per_class 7325 --arch preactresnet18 --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 2000 --schedule 500 1000 1500 --gammas 0.1 0.1 0.1 --train mixup --mixup_alpha 1.0
```

#### Manifold mixup Preactresnet18
```
python pgd.py --dataset svhn --data_dir data/svhn/ --root_dir experiments/ --labels_per_class 7325 --arch preactresnet18 --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 2000 --schedule 500 1000 1500 --gammas 0.1 0.1 0.1 --train mixup_hidden --mixup_alpha 2.0
```

#### No mixup Preactresnet34
```
python pgd.py --dataset svhn --data_dir data/svhn/ --root_dir experiments/ --labels_per_class 7325 --arch preactresnet34 --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 2000 --schedule 500 1000 1500 --gammas 0.1 0.1 0.1 --train vanilla
```

#### Mixup Preactresnet34

```
python pgd.py --dataset svhn --data_dir data/svhn/ --root_dir experiments/ --labels_per_class 7325 --arch preactresnet34 --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 2000 --schedule 500 1000 1500 --gammas 0.1 0.1 0.1 --train mixup --mixup_alpha 1.0
```

#### Manifold mixup Preactresnet34
```
python pgd.py --dataset svhn --data_dir data/svhn/ --root_dir experiments/ --labels_per_class 7325 --arch preactresnet34 --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 2000 --schedule 500 1000 1500 --gammas 0.1 0.1 0.1 --train mixup_hidden --mixup_alpha 2.0
```

#### No mixup WRN-28-10
```
python pgd.py --dataset svhn --data_dir data/svhn/ --root_dir experiments/ --labels_per_class 7325 --arch wrn28_10 --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 400 --schedule 200 300 --gammas 0.1 0.1 --train vanilla
```

#### Mixup WRN-28-10

```
python pgd.py --dataset svhn --data_dir data/svhn/ --root_dir experiments/ --labels_per_class 7325 --arch wrn28_10 --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 400 --schedule 200 300 --gammas 0.1 0.1 --train mixup --mixup_alpha 1.0
```

#### Manifold mixup WRN-28-10
```
python pgd.py --dataset svhn --data_dir data/svhn/ --root_dir experiments/ --labels_per_class 7325 --arch wrn28_10 --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 400 --schedule 200 300 --gammas 0.1 0.1 --train mixup_hidden --mixup_alpha 2.0
```

### How to run experiments for Tiny-Imagenet-200

1.Download the zipped data from https://tiny-imagenet.herokuapp.com/
2.If not already existing, create a subfolder "data" in root folder "manifold_mixup"
3.Extract the zipped data in folder manifold_mixup/data
4.Run the following script (This will arange the validation data in the format required by the pytorch loader)
```
python utils.py
```

5. Run the following commands
#### No mixup Preactresnet18
```
python pgd.py --dataset tiny-imagenet-200 --data_dir data/tiny-imagenet-200/ --root_dir experiments/ --labels_per_class 500 --arch preactresnet18 --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 2000 --schedule 1000 1500 --gammas 0.1 0.1 --train vanilla 
```

#### Mixup Preactresnet18

```
python pgd.py --dataset tiny-imagenet-200 --data_dir data/tiny-imagenet-200/ --root_dir experiments/ --labels_per_class 500 --arch preactresnet18 --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 2000 --schedule 1000 1500 --gammas 0.1 0.1 --train mixup --mixup_alpha 0.2
```

#### Manifold mixup Preactresnet18
```
python pgd.py --dataset tiny-imagenet-200 --data_dir data/tiny-imagenet-200/ --root_dir experiments/ --labels_per_class 500 --arch preactresnet18 --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 2000 --schedule 1000 1500 --gammas 0.1 0.1 --train mixup_hidden --mixup_alpha 0.2
```