from __future__ import division

import argparse

import matplotlib as mpl

import models
from data_process import *
from utils import *

mpl.use('Agg')

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Trains ResNeXt on CIFAR or ImageNet',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, default='cifar10',
                    choices=['cifar10', 'cifar100', 'imagenet', 'svhn', 'stl10', 'mnist', 'tiny-imagenet-200'],
                    help='Choose between Cifar10/100 and ImageNet.')
parser.add_argument('--data_dir', type=str, default='cifar10',
                    help='file where results are to be written')
parser.add_argument('--root_dir', type=str, default='experiments',
                    help='folder where results are to be stored')
parser.add_argument('--labels_per_class', type=int, default=5000, metavar='NL',
                    help='labels_per_class')
parser.add_argument('--valid_labels_per_class', type=int, default=0, metavar='NL',
                    help='validation labels_per_class')

parser.add_argument('--arch', metavar='ARCH', default='resnext29_8_64', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnext29_8_64)')
parser.add_argument('--initial_channels', type=int, default=64, choices=(16, 64))
# Optimization options
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
parser.add_argument('--train', type=str, default='vanilla', choices=['vanilla', 'mixup', 'mixup_hidden', 'cutout'])
parser.add_argument('--mixup_alpha', type=float, default=0.0, help='alpha parameter for mixup')

parser.add_argument('--dropout', action='store_true', default=False,
                    help='whether to use dropout or not in final layer')
# parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
parser.add_argument('--batch_size', type=int, default=100, help='Batch size.')
parser.add_argument('--learning_rate', type=float, default=0.1, help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--data_aug', type=int, default=1)
parser.add_argument('--adv_unpre', action='store_true', default=False,
                    help='the adversarial examples will be calculated on real input space (not preprocessed)')
parser.add_argument('--decay', type=float, default=0.0001, help='Weight decay (L2 penalty).')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1],
                    help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
# Checkpoints
parser.add_argument('--print_freq', default=100, type=int, metavar='N', help='print frequency (default: 200)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers (default: 2)')
# random seed
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--add_name', type=str, default='')
parser.add_argument('--job_id', type=str, default='')

args = parser.parse_args()

out_str = str(args)
print(out_str)

device = torch.device("cuda")


def experiment_name(dataset, arch, epochs, dropout, batch_size, lr, momentum, decay, data_aug, train,
                    mixup_alpha, job_id, add_name):
    exp_name = dataset
    exp_name += '_arch_' + str(arch)
    exp_name += '_train_' + str(train)
    exp_name += '_m_alpha_' + str(mixup_alpha)
    if dropout:
        exp_name += '_do_' + 'true'
    else:
        exp_name += '_do_' + 'False'
    exp_name += '_eph_' + str(epochs)
    exp_name += '_bs_' + str(batch_size)
    exp_name += '_lr_' + str(lr)
    exp_name += '_mom_' + str(momentum)
    exp_name += '_decay_' + str(decay)
    exp_name += '_data_aug_' + str(data_aug)
    if job_id is not None:
        exp_name += '_job_id_' + str(job_id)
    if add_name != '':
        exp_name += '_add_name_' + str(add_name)

    return exp_name


def main():
    exp_name = experiment_name(dataset=args.dataset,
                               arch=args.arch,
                               epochs=args.epochs,
                               dropout=args.dropout,
                               batch_size=args.batch_size,
                               lr=args.learning_rate,
                               momentum=args.momentum,
                               decay=args.decay,
                               data_aug=args.data_aug,
                               train=args.train,
                               mixup_alpha=args.mixup_alpha,
                               job_id=args.job_id,
                               add_name=args.add_name)
    exp_dir = args.root_dir + exp_name
    backup_model_dir = "../mixup/" + args.root_dir + "backup_" + exp_name

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    log = open(os.path.join(exp_dir, 'log.txt'.format(args.manualSeed)), 'w')

    print_log("exp_name = {}".format(exp_name), log)
    print_log("exp_dir = {}".format(exp_dir), log)
    print_log("backup_model_dir = {}".format(backup_model_dir), log)
    assert os.path.exists(backup_model_dir), "Experiment directory not found: " + backup_model_dir

    per_img_std = False
    if args.dataset == 'tiny-imagenet-200':
        stride = 2
    else:
        stride = 1

    raw_train_data, raw_test_data, num_classes = load_raw_dataset(args.data_aug, args.dataset, args.data_dir)
    if args.arch != "lenet5":
        net = models.__dict__[args.arch](num_classes, args.dropout, per_img_std, stride).cuda()
    else:
        net = models.lenet5(num_classes).cuda()
    optimizer = torch.optim.SGD(net.parameters(), args.learning_rate, momentum=args.momentum,
                                weight_decay=args.decay, nesterov=True)

    """
    Load model-best checkpoint.
    checkpoint = {"epoch", "arch", "state_dict", "recorder", "state_dict"}
    """
    print_log("\nStart loading model-best checkpoint...", log)
    checkpoint = torch.load(backup_model_dir + "/model_best.pth.tar")
    net.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    recorder = checkpoint["recorder"]
    print_log("Load model-best checkpoint successfully!\n", log)

    adv_dataset_dir = args.data_dir + "adv_data_" + exp_name + ".pth"
    if os.path.exists(adv_dataset_dir):
        print_log("Load adversarial data from {} directly...".format(adv_dataset_dir), log)
        adversarial_dataset = torch.load(adv_dataset_dir)
        print_log("Load adversarial data successfully!", log)
    else:
        adversarial_dataset = attack_train_data(net, raw_train_data, args.batch_size, num_iter=100)
        print_log("Save adversarial data to {}...".format(adv_dataset_dir), log)
        torch.save(adversarial_dataset, adv_dataset_dir)
        print_log("Save adversarial data successfully!", log)
    train_loader, test_loader = load_dataset(adversarial_dataset, raw_test_data, num_classes,
                                             args.dataset, args.batch_size, 2, args.labels_per_class)

    print_log("\nfinish", log)


if __name__ == '__main__':
    main()
