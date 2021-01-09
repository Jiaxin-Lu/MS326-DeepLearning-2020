# Adversarial Attack and Defense Based on Data Mixup

Course project of MS326 2020

## Preparations

### How to run Python scripts in shell?

1. To import the "models" package, suppose the repository is at directory `REPO_DIR`. You should add `REPO_DIR` and `REPO_DIR/mixup` to the environmental variable `PYTHONPATH`, like
```
export PYTHONPATH=/home/MasterJH5574/MS326-Project-2020:/home/MasterJH5574/MS326-Project-2020/mixup
```
2. Install necessary packages via Conda or Pip, that is, Python version should be at least 3.6.2, and the latest version of the following packages is enough.

   ```
   torch
   torchvision
   matplotlib
   numpy
   pandas
   Pillow
   scipy
   seaborn
   six
   ```

3. You can run the scripts!

### How to run Python scripts in Pycharm?

You should mark the directories "MS326-Project-2020", "mixup" and "pgd" as "Sources Root" in Pycharm.

## Running Commands

### Experiment 1: Train and Test with Clean Data

The following commands should be run in directory `REPO_DIR/mixup`.

#### Vanilla LeNet5

```
python main.py --dataset mnist --data_dir data/mnist/ --root_dir experiments/ --labels_per_class 5000 --arch lenet5 --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 500 --schedule 300 400 --gammas 0.1 0.1 --train vanilla
```

#### Mixup LeNet5

```
python main.py --dataset mnist --data_dir data/mnist/ --root_dir experiments/ --labels_per_class 5000 --arch lenet5 --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 500 --schedule 300 400 --gammas 0.1 0.1 --train mixup --mixup_alpha 1
```

#### Manifold Mixup LeNet 5

```
python main.py --dataset mnist --data_dir data/mnist/ --root_dir experiments/ --labels_per_class 5000 --arch lenet5 --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 500 --schedule 300 400 --gammas 0.1 0.1 --train mixup_hidden --mixup_alpha 2.0
```

#### Vanilla PreAct-ResNet18

```
python main.py --dataset cifar10 --data_dir data/cifar10/ --root_dir experiments/ --labels_per_class 5000 --arch preactresnet18 --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 2000 --schedule 500 1000 1500 --gammas 0.1 0.1 0.1 --train vanilla
```

#### Mixup PreAct-ResNet18

```
python main.py --dataset cifar10 --data_dir data/cifar10/ --root_dir experiments/ --labels_per_class 5000 --arch preactresnet18 --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 2000 --schedule 500 1000 1500 --gammas 0.1 0.1 0.1 --train mixup --mixup_alpha 1.0
```

#### Manifold Mixup PreAct-ResNet18

```
python main.py --dataset cifar10 --data_dir data/cifar10/ --root_dir experiments/ --labels_per_class 5000 --arch preactresnet18 --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 2000 --schedule 500 1000 1500 --gammas 0.1 0.1 0.1 --train mixup_hidden --mixup_alpha 2.0
```

### Experiment 2: PGD Attack

This experiment is done with experiment 3. The `Adversarial attack before training` of the first epoch is exactly the accuracy after the PGD attack. One should refer to files

* `REPO_DIR/adversarial-training/mnist_vanilla_from_model/log.txt`
* `REPO_DIR/adversarial-training/mnist_mixup_from_model/log.txt`
* `REPO_DIR/adversarial-training/mnist_manifold_from_model/log.txt`
* `REPO_DIR/adversarial-training/cifar10_vanilla_from_model/log.txt`
* `REPO_DIR/adversarial-training/cifar10_mixup_from_model/log.txt`
* `REPO_DIR/adversarial-training/cifar10_manifold_from_model/log.txt`

### Experiment 3: Adversarial Training

In directory `REPO_DIR/adversarial-training`, run

#### Vanilla LeNet5, from Pretrained Model

```
python adv-main.py --dataset mnist --data_dir data/mnist/ --root_dir experiments/ --labels_per_class 5000 --arch lenet5 --learning_rate 0.0002 --momentum 0.9 --decay 0 --epochs 800 --schedule 400 600 --gammas 0.5 0.5 --train vanilla --pgd_eps 0.3 --pgd_alpha 0.01 --pgd_step_size 40
```

#### Vanilla LeNet5, from Newly Initialized Model

```
python adv-main.py --dataset mnist --data_dir data/mnist/ --root_dir experiments/ --labels_per_class 5000 --arch lenet5 --learning_rate 0.0002 --momentum 0.9 --decay 0 --epochs 800 --schedule 400 600 --gammas 0.5 0.5 --train vanilla --pgd_eps 0.3 --pgd_alpha 0.01 --pgd_step_size 40 --train_from_empty
```

#### Mixup LeNet5, from Pretrained Model

```
python adv-main.py --dataset mnist --data_dir data/mnist/ --root_dir experiments/ --labels_per_class 5000 --arch lenet5 --learning_rate 0.0002 --momentum 0.9 --decay 0 --epochs 800 --schedule 400 600 --gammas 0.5 0.5 --train mixup --mixup_alpha 1.0 --pgd_eps 0.3 --pgd_alpha 0.01 --pgd_step_size 40
```

#### Mixup LeNet5, from Newly Initialized Model

```
python adv-main.py --dataset mnist --data_dir data/mnist/ --root_dir experiments/ --labels_per_class 5000 --arch lenet5 --learning_rate 0.0002 --momentum 0.9 --decay 0 --epochs 800 --schedule 400 600 --gammas 0.5 0.5 --train mixup --mixup_alpha 1.0 --pgd_eps 0.3 --pgd_alpha 0.01 --pgd_step_size 40 --train_from_empty
```

#### Manifold Mixup LeNet5, from Pretrained Model

```
python adv-main.py --dataset mnist --data_dir data/mnist/ --root_dir experiments/ --labels_per_class 5000 --arch lenet5 --learning_rate 0.0002 --momentum 0.9 --decay 0 --epochs 800 --schedule 400 600 --gammas 0.5 0.5 --train mixup_hidden --mixup_alpha 2.0 --pgd_eps 0.3 --pgd_alpha 0.01 --pgd_step_size 40
```

#### Manifold Mixup LeNet5, from Newly Initialized Model

```
python adv-main.py --dataset mnist --data_dir data/mnist/ --root_dir experiments/ --labels_per_class 5000 --arch lenet5 --learning_rate 0.0002 --momentum 0.9 --decay 0 --epochs 800 --schedule 400 600 --gammas 0.5 0.5 --train mixup_hidden --mixup_alpha 2.0 --pgd_eps 0.3 --pgd_alpha 0.01 --pgd_step_size 40 --train_from_empty
```

#### Vanilla PreAct-ResNet18, from Pretrained Model

```
python adv-main.py --dataset cifar10 --data_dir data/cifar10/ --root_dir experiments/ --labels_per_class 5000 --arch preactresnet18 --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 500 --schedule 1000 1500 --gammas 0.1 0.1 --train vanilla
```

#### Mixup PreAct-ResNet18, from Pretrained Model

```
python adv-main.py --dataset cifar10 --data_dir data/cifar10/ --root_dir experiments/ --labels_per_class 5000 --arch preactresnet18 --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 500 --schedule 1000 1500 --gammas 0.1 0.1 --train mixup --mixup_alpha 1.0
```

#### Manifold Mixup PreAct-ResNte18, from Pretrained Model

```
python adv-main.py --dataset cifar10 --data_dir data/cifar10/ --root_dir experiments/ --labels_per_class 5000 --arch preactresnet18 --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 500 --schedule 1000 1500 --gammas 0.1 0.1 --train mixup_hidden --mixup_alpha 2.0
```

### Experiment 4.1: Soft Labels

In directory `REPO_DIR/adversarial-training`, run the following commands.

The `noise` value can be modified to other reasonable noise values.

#### Mixup LeNet5

```
python adv-main.py --dataset mnist --data_dir data/mnist/ --root_dir experiments/ --labels_per_class 5000 --arch lenet5 --learning_rate 0.0002 --momentum 0.9 --decay 0 --epochs 800 --schedule 400 600 --gammas 0.5 0.5 --train mixup --mixup_alpha 1.0 --pgd_eps 0.3 --pgd_alpha 0.01 --pgd_step_size 40 --noise 0.01
```

#### Mixup PreAct-ResNet18

```
python adv-main.py --dataset cifar10 --data_dir data/cifar10/ --root_dir experiments/ --labels_per_class 5000 --arch preactresnet18 --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 500 --schedule 1000 1500 --gammas 0.1 0.1 --train mixup --mixup_alpha 1.0 --noise 0.01
```

### Experiment 4.2: Partial PGD

In directory `REPO_DIR/adversarial-training`, run the following commands.

#### LeNet5

```
python adv-main.py --dataset mnist --data_dir data/mnist/ --root_dir experiments/ --labels_per_class 5000 --arch lenet5 --learning_rate 0.0002 --momentum 0.9 --decay 0 --epochs 800 --schedule 400 600 --gammas 0.5 0.5 --train partial_pgd --mixup_alpha 1.0 --pgd_eps 0.3 --pgd_alpha 0.01 --pgd_step_size 40
```

#### PreAct-ResNet18

```
python adv-main.py --dataset cifar10 --data_dir data/cifar10/ --root_dir experiments/ --labels_per_class 5000 --arch preactresnet18 --learning_rate 0.1 --momentum 0.9 --decay 0.0001 --epochs 500 --schedule 1000 1500 --gammas 0.1 0.1 --train partial_pgd --mixup_alpha 1.0
```

