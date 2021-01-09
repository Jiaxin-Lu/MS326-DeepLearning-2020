# Manifold_mixup Supervised

### Requirements
This code has been tested with
python 3.6.8
torch 1.0.0
torchvision 0.2.1

### Additional packages required

```
matplotlib==3.0.2
numpy==1.15.4
pandas==0.23.4
Pillow==5.4.1
scipy==1.1.0
seaborn==0.9.0
six==1.12.0
```

### Important :Running each of the following commands will automatically create a subdirectory containing the output of that particular experiment in the manifold_mixup/supervised/experiments directory

### How to run experiments for CIFAR10

#### No mixup Preactresnet18
```
python main.py --dataset cifar10 --data_dir data/cifar10/ --root_dir experiments/ --labels_per_class 5000 --arch preactresnet18 --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 2000 --schedule 500 1000 1500 --gammas 0.1 0.1 0.1 --train vanilla 
```

#### Mixup Preactresnet18

```
python main.py --dataset cifar10 --data_dir data/cifar10/ --root_dir experiments/ --labels_per_class 5000 --arch preactresnet18 --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 2000 --schedule 500 1000 1500 --gammas 0.1 0.1 0.1 --train mixup --mixup_alpha 1.0
```

#### Manifold mixup Preactresnet18
```
python main.py --dataset cifar10 --data_dir data/cifar10/ --root_dir experiments/ --labels_per_class 5000 --arch preactresnet18 --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 2000 --schedule 500 1000 1500 --gammas 0.1 0.1 0.1 --train mixup_hidden --mixup_alpha 2.0
```

#### No mixup Preactresnet34
```
python main.py --dataset cifar10 --data_dir data/cifar10/ --root_dir experiments/ --labels_per_class 5000 --arch preactresnet34 --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 2000 --schedule 500 1000 1500 --gammas 0.1 0.1 0.1 --train vanilla 
```

#### Mixup Preactresnet34

```
python main.py --dataset cifar10 --data_dir data/cifar10/ --root_dir experiments/ --labels_per_class 5000 --arch preactresnet34 --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 2000 --schedule 500 1000 1500 --gammas 0.1 0.1 0.1 --train mixup --mixup_alpha 1.0
```

#### Manifold mixup Preactresnet34
```
python main.py --dataset cifar10 --data_dir data/cifar10/ --root_dir experiments/ --labels_per_class 5000 --arch preactresnet34 --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 2000 --schedule 500 1000 1500 --gammas 0.1 0.1 0.1 --train mixup_hidden --mixup_alpha 2.0 
```

### How to run experiments for MNIST

#### No mixup LeNet5
```
python main.py --dataset mnist --data_dir data/mnist/ --root_dir experiments/ --labels_per_class 5000 --arch lenet5 --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 500 --schedule 300 400 --gammas 0.1 0.1 --train vanilla
```

#### Mixup LeNet5

```
python main.py --dataset mnist --data_dir data/mnist/ --root_dir experiments/ --labels_per_class 5000 --arch lenet5 --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 500 --schedule 300 400 --gammas 0.1 0.1 --train mixup --mixup_alpha 1
```

#### Manifold mixup LeNet5

```
python main.py --dataset mnist --data_dir data/mnist/ --root_dir experiments/ --labels_per_class 5000 --arch lenet5 --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 500 --schedule 300 400 --gammas 0.1 0.1 --train mixup_hidden --mixup_alpha 2.0
```
