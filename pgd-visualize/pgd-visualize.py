from __future__ import division

import argparse
import sys
from collections import OrderedDict

import matplotlib as mpl

import models
from pgd_v_data_process import *
from utils import *

if sys.version_info[0] < 3:
    import cPickle as pickle
else:
    import _pickle as pickle
import shutil

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
parser.add_argument('--decay', type=float, default=0, help='Weight decay (L2 penalty).')
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
parser.add_argument('--train_from_empty', action='store_true', dest='train_from_empty', default=False)


parser.add_argument('--pgd_eps',type=float, default=8/255)
parser.add_argument('--pgd_alpha',type=float,default=2/255)
parser.add_argument('--pgd_step_size',type=int,default=7)

parser.add_argument('--print_origin_images', action='store_true', dest='print_origin_images', default=False)

args = parser.parse_args()

out_str = str(args)
print(out_str)

device = torch.device("cuda")


def experiment_name(dataset, arch, epochs, dropout, batch_size, lr, momentum, decay, data_aug, train,
                    mixup_alpha, job_id, add_name,eps,alpha,step_size,from_empty):
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
    exp_name += '_eps_' + str(eps)
    exp_name += '_alpha_' + str(alpha)
    exp_name += '_step_size_' + str(step_size)
    if from_empty:
        exp_name += '_from_empty'
    else:
        exp_name += '_from_model'
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
                               add_name=args.add_name,
                               eps=args.pgd_eps,
                               alpha=args.pgd_alpha,
                               step_size=args.pgd_step_size,
                               from_empty=args.train_from_empty)
    exp_dir = args.root_dir + exp_name

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    log = open(os.path.join(exp_dir, 'log.txt'.format(args.manualSeed)), 'w')

    print_log("exp_name = {}".format(exp_name), log)
    print_log("exp_dir = {}".format(exp_dir), log)

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
    if args.dataset=="mnist":
        backup_model_dir = "../mixup/" + args.root_dir + "backup_mnist_arch_lenet5_train_vanilla_m_alpha_0.0_do_False_eph_500_bs_100_lr_0.1_mom_0.9_decay_0.0001_data_aug_1_job_id_"
    elif args.dataset=='cifar10':
        backup_model_dir = "../mixup/" + args.root_dir + "backup_cifar10_arch_preactresnet18_train_vanilla_m_alpha_0.0_do_False_eph_2000_bs_100_lr_0.1_mom_0.9_decay_0.0001_data_aug_1_job_id_"

    print_log("backup_model_dir = {}".format(backup_model_dir), log)
    assert os.path.exists(backup_model_dir), "Experiment directory not found: " + backup_model_dir

    if not args.train_from_empty:
        basic_model = "model"
        print_log("\nStart loading model-best checkpoint...", log)
        checkpoint = torch.load(backup_model_dir + "/model_best.pth.tar")
        net.load_state_dict(checkpoint["state_dict"])
        print_log("Load model-best checkpoint successfully!", log)
    else:
        basic_model = "empty"

    adv_test_dataset = attack_test_data(0, log, net, raw_test_data, 200,
                                        args.pgd_step_size, args.pgd_eps, args.pgd_alpha)
    if args.dataset == "mnist":
        adv_test_dataset.inputs = adv_test_dataset.inputs.cpu()
    else:
        adv_test_dataset.inputs = adv_test_dataset.inputs.permute(0, 2, 3, 1).cpu().numpy()

    img_suffix = args.dataset + "_" + (basic_model if not args.print_origin_images else "origin")

    print(type(raw_test_data.data), raw_test_data.data.shape)
    print(type(adv_test_dataset.inputs), adv_test_dataset.inputs.shape)
    if args.dataset == "cifar10":
        train_mean = np.mean(raw_train_data.data, axis=(0, 1, 2))
        train_std = np.sqrt(np.mean((raw_train_data.data - train_mean) ** 2, axis=(0, 1, 2)))
        test_mean = np.mean(raw_test_data.data, axis=(0, 1, 2))
        test_std = np.sqrt(np.mean((raw_test_data.data - test_mean) ** 2, axis=(0, 1, 2)))
        print(train_mean, train_std)
        print(test_mean, test_std)
    else:
        train_mean = train_std = test_mean = test_std = None

    p = 0
    print_log("Start generating images...", log)
    for label in range(0, 10):
        while True:
            cond = adv_test_dataset[p][1] == label \
                if not args.print_origin_images \
                else raw_test_data[p][1] == label
            if cond:
                print("print {}".format(label))
                if args.dataset == "mnist":
                    arr = adv_test_dataset[p][0].reshape(32, 32).numpy() \
                        if not args.print_origin_images \
                        else raw_test_data[p][0].reshape(32, 32).numpy()
                    plt.imsave(exp_dir + "/" + str(label) + "_" + img_suffix + ".png", arr, cmap="gray")
                else:
                    if not args.print_origin_images:
                        arr = adv_test_dataset[p][0].reshape(32, 32, 3)
                    else:
                        arr = raw_test_data.data[p] / 255
                    arr = np.clip(arr, 0, 1)
                    plt.imsave(exp_dir + "/" + str(label) + "_" + img_suffix + ".png", arr)
                break
            p += 1

    print_log("\nfinish", log)
    log.close()


if __name__ == '__main__':
    main()
