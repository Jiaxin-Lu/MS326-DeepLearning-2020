from functools import reduce
from operator import __or__

import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms

from utils import *


class AdversarialDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index) -> T_co:
        return self.inputs[index], self.labels[index]


def per_image_standardization(x):
    y = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
    mean = y.mean(dim=1, keepdim=True).expand_as(y)
    std = y.std(dim=1, keepdim=True).expand_as(y)
    adjusted_std = torch.max(std, 1.0 / torch.sqrt(torch.cuda.FloatTensor([x.shape[1] * x.shape[2] * x.shape[3]])))
    y = (y - mean) / adjusted_std
    standarized_input = y.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
    return standarized_input


def load_raw_dataset(data_aug, dataset, data_target_dir):
    if dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    elif dataset == 'svhn':
        mean = [x / 255 for x in [127.5, 127.5, 127.5]]
        std = [x / 255 for x in [127.5, 127.5, 127.5]]
    elif dataset == 'tiny-imagenet-200':
        mean = [x / 255 for x in [127.5, 127.5, 127.5]]
        std = [x / 255 for x in [127.5, 127.5, 127.5]]
    elif dataset == 'mnist':
        pass
    else:
        assert False, "Unknown dataset : {}".format(dataset)

    if data_aug == 1:
        print('data aug')
        if dataset == 'svhn':
            train_transform = transforms.Compose(
                [transforms.RandomCrop(32, padding=2), transforms.ToTensor(),
                 transforms.Normalize(mean, std)])
            test_transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean, std)])
        elif dataset == 'mnist':
            hw_size = 32
            train_transform = transforms.Compose([
                transforms.Resize((hw_size, hw_size)),
                transforms.ToTensor()
            ])
            test_transform = transforms.Compose([
                transforms.Resize((hw_size, hw_size)),
                transforms.ToTensor()
            ])
        elif dataset == 'tiny-imagenet-200':
            train_transform = transforms.Compose(
                [transforms.RandomHorizontalFlip(),
                 transforms.RandomCrop(64, padding=4),
                 transforms.ToTensor(),
                 transforms.Normalize(mean, std)])
            test_transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean, std)])
        else:
            train_transform = transforms.Compose(
                [transforms.RandomHorizontalFlip(),
                 transforms.RandomCrop(32, padding=2),
                 transforms.ToTensor(),
                 transforms.Normalize(mean, std)])
            test_transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean, std)])
    else:
        print('no data aug')
        if dataset == 'mnist':
            hw_size = 32
            train_transform = transforms.Compose([
                transforms.Resize((hw_size, hw_size)),
                transforms.ToTensor()
            ])
            test_transform = transforms.Compose([
                transforms.Resize((hw_size, hw_size)),
                transforms.ToTensor()
            ])
        else:
            train_transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize(mean, std)])
            test_transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean, std)])
    if dataset == 'cifar10':
        train_data = datasets.CIFAR10(data_target_dir, train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR10(data_target_dir, train=False, transform=test_transform, download=True)
        num_classes = 10
    elif dataset == 'cifar100':
        train_data = datasets.CIFAR100(data_target_dir, train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR100(data_target_dir, train=False, transform=test_transform, download=True)
        num_classes = 100
    elif dataset == 'svhn':
        train_data = datasets.SVHN(data_target_dir, split='train', transform=train_transform, download=True)
        test_data = datasets.SVHN(data_target_dir, split='test', transform=test_transform, download=True)
        num_classes = 10
    elif dataset == 'mnist':
        train_data = datasets.MNIST(data_target_dir, train=True, transform=train_transform, download=True)
        test_data = datasets.MNIST(data_target_dir, train=False, transform=test_transform, download=True)
        num_classes = 10
    elif dataset == 'tiny-imagenet-200':
        train_root = os.path.join(data_target_dir, 'train')  # this is path to training images folder
        validation_root = os.path.join(data_target_dir, 'val/images')  # this is path to validation images folder
        train_data = datasets.ImageFolder(train_root, transform=train_transform)
        test_data = datasets.ImageFolder(validation_root, transform=test_transform)
        num_classes = 200
    else:
        assert False, "Dataset {} is unsupported.".format(dataset)

    return train_data, test_data, num_classes


def attack_single_batch_input(net, images, labels, num_iter, eps=0.3):
    alpha = 2.5 * eps / num_iter

    images = images.cuda()
    labels = labels.cuda()
    loss_function = nn.CrossEntropyLoss()

    ori_images = images.data
    for i in range(num_iter):
        images.requires_grad = True
        output = net(images)

        net.zero_grad()
        loss = loss_function(output, labels).cuda()
        loss.backward()

        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()

    return images


def attack_train_data(log, net, raw_train_data, batch_size, num_iter=100):
    net.eval()

    print_log("\nStart generating adversarial data...", log)
    c, h, w = raw_train_data[0][0].shape
    raw_train_input = []
    raw_train_label = []
    cnt = 0
    for image, label in raw_train_data:
        cnt += 1
        if cnt % 10000 == 0:
            print(cnt)
        raw_train_input.append(image.reshape(1, c, h, w))
        raw_train_label.append(label)
    raw_train_input = torch.cat(raw_train_input, dim=0)
    raw_train_label = torch.tensor(raw_train_label)

    adversarial_train_input = []
    for i in range(0, len(raw_train_data), batch_size):
        if i % 1000 == 0:
            print(i)
        images = raw_train_input[i:i + batch_size]
        labels = raw_train_label[i:i + batch_size]
        adversarial_batch_input = attack_single_batch_input(net, images, labels, num_iter)
        adversarial_train_input.append(adversarial_batch_input)
    adversarial_train_input = torch.cat(adversarial_train_input, dim=0)

    adversarial_dataset = AdversarialDataset(adversarial_train_input, raw_train_label)

    print_log("Generate adversarial data successfully!", log)
    return adversarial_dataset


def load_dataset(adversarial_dataset, test_data, num_classes, dataset, batch_size, workers, labels_per_class, log):
    def get_sampler(labels, labels_per_class=None):
        # Only choose digits in num_classes
        (indices,) = np.where(reduce(__or__, [labels == i for i in np.arange(num_classes)]))
        # Ensure uniform distribution of labels
        np.random.shuffle(indices)

        indices_train = np.hstack(
            [list(filter(lambda idx: labels[idx] == i, indices))[:labels_per_class] for i in range(num_classes)])
        indices_train = torch.from_numpy(indices_train)
        sampler_train = SubsetRandomSampler(indices_train)
        return sampler_train

    print_log("\nStart constructing adversarial dataloader...", log)
    if dataset == 'tiny-imagenet-200':
        train_sampler = None
    else:
        train_sampler = get_sampler(adversarial_dataset.labels, labels_per_class)

    if dataset == 'tiny-imagenet-200':
        adversarial_dataloader = torch.utils.data.DataLoader(adversarial_dataset, batch_size=batch_size, shuffle=True,
                                                             num_workers=workers, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                                  num_workers=workers, pin_memory=True)
    else:
        adversarial_dataloader = torch.utils.data.DataLoader(adversarial_dataset, batch_size=batch_size,
                                                             sampler=train_sampler, shuffle=False,
                                                             num_workers=workers, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                                  num_workers=workers, pin_memory=True)

    print_log("Constructing adversarial dataset successfully!", log)
    return adversarial_dataloader, test_loader
