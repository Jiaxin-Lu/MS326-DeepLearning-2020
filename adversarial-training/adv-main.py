from __future__ import division

import argparse
import sys
from collections import OrderedDict

import matplotlib as mpl

import models
from adv_data_process import *
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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def clear_directory(path_to_dir):
    for filename in os.listdir(path_to_dir):
        file_path = os.path.join(path_to_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def save_checkpoint(state_dict, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state_dict, filename)


def adjust_learning_rate(optimizer, epoch, gammas, schedule):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate
    assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
    for (gamma, step) in zip(gammas, schedule):
        if epoch >= step:
            lr = lr * gamma
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


bce_loss = nn.BCELoss().cuda()
softmax = nn.Softmax(dim=1).cuda()


def train(t, train_loader, model, optimizer, epoch, args, log):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        target = target.long()
        input, target = input.cuda(), target.cuda()
        data_time.update(time.time() - end)
        if args.train == 'mixup':
            input_var, target_var = Variable(input), Variable(target)
            output, reweighted_target = model(input_var, target_var, mixup=True, mixup_alpha=args.mixup_alpha)
            loss = bce_loss(softmax(output), reweighted_target)
        elif args.train == 'mixup_hidden':
            input_var, target_var = Variable(input), Variable(target)
            output, reweighted_target = model(input_var, target_var, mixup_hidden=True, mixup_alpha=args.mixup_alpha)
            loss = bce_loss(softmax(output), reweighted_target)
        elif args.train == 'vanilla':
            input_var, target_var = Variable(input), Variable(target)
            output, reweighted_target = model(input_var, target_var)
            loss = bce_loss(softmax(output), reweighted_target)
        else:
            assert False

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print_log('[iter {}] Epoch: [{:03d}][{:03d}/{:03d}]   '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                      'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
                t, epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5) + time_string(), log)

    print_log(('[iter {}] **Train** Prec@1 {top1.avg:.3f} '
               'Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}').format(t, top1=top1, top5=top5, error1=100 - top1.avg),
              log)
    return top1.avg, top5.avg, losses.avg


def validate(t, val_loader, model, mode, log):
    print_log("\n[iter {}] Start validation ({})...".format(t, mode), log)
    criterion = nn.CrossEntropyLoss().cuda()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (input_, target) in enumerate(val_loader):
        input_ = input_.cuda()
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            input_var = Variable(input_)
            target_var = Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input_.size(0))
        top1.update(prec1.item(), input_.size(0))
        top5.update(prec5.item(), input_.size(0))

    print_log(
        ('[iter {}] **{}** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
         + ' Error@1 {error1:.3f} Loss: {losses.avg:.3f} ').format(
            t, mode, top1=top1, top5=top5, error1=100 - top1.avg, losses=losses), log)

    return top1.avg, losses.avg


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

    backup_model_dir = "../mixup/" + args.root_dir + "backup_" + exp_name
    print_log("backup_model_dir = {}".format(backup_model_dir), log)
    assert os.path.exists(backup_model_dir), "Experiment directory not found: " + backup_model_dir

    print_log("\nStart loading model-best checkpoint...", log)
    checkpoint = torch.load(backup_model_dir + "/model_best.pth.tar")
    net.load_state_dict(checkpoint["state_dict"])
    print_log("Load model-best checkpoint successfully!", log)

    num_iter = 3
    for t in range(1, num_iter + 1):
        print_log("\n======== ITERATION {} ========".format(t), log)

        """
        Load model-best checkpoint.
        checkpoint = {"epoch", "arch", "state_dict", "recorder", "state_dict"}
        """

        optimizer = torch.optim.SGD(net.parameters(), args.learning_rate, momentum=args.momentum,
                                    weight_decay=args.decay, nesterov=True)
        recorder = RecorderMeter(args.epochs)
        best_acc = 0

        # adv_dataset_dir = exp_dir + "/iter" + str(t) + "/" + "adv_data_" + exp_name + ".pth"
        # if os.path.exists(adv_dataset_dir):
        #     print_log("\n Load adversarial data from {} directly...".format(adv_dataset_dir), log)
        #     adversarial_dataset = torch.load(adv_dataset_dir)
        #     print_log(" Load adversarial data successfully!", log)
        # else:
        #     adversarial_dataset = attack_train_data(log, net, raw_train_data, args.batch_size, num_iter=100)
        #     print_log("\nSave adversarial data to {}...".format(adv_dataset_dir), log)
        #     torch.save(adversarial_dataset, adv_dataset_dir)
        #     print_log("Save adversarial data successfully!", log)
        adversarial_dataset = attack_train_data(t, log, net, raw_train_data, args.batch_size, num_iter=100)

        adversarial_dataset.inputs = adversarial_dataset.inputs.cpu()
        adversarial_dataset.labels = adversarial_dataset.labels.cpu()
        adv_loader, test_loader = load_dataset(t, adversarial_dataset, raw_test_data, num_classes,
                                               args.dataset, args.batch_size, 2, args.labels_per_class, log)

        validate(t, adv_loader, net, "Adversarial attack", log)

        # Train!
        start_time = time.time()
        epoch_time = AverageMeter()
        train_loss = []
        train_acc = []
        test_loss = []
        test_acc = []

        iter_dir = exp_dir + "/iter" + str(t)
        if os.path.exists(iter_dir):
            clear_directory(iter_dir)
        else:
            os.makedirs(iter_dir)

        # for epoch in range(args.start_epoch, args.epochs):
        for epoch in range(0, 10):
            current_learning_rate = adjust_learning_rate(optimizer, epoch, args.gammas, args.schedule)

            need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs - epoch))
            need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(t, need_hour, need_mins, need_secs)

            print_log(
                ('\n[iter {}] ==>>{:s} [Epoch={:03d}/{:03d}] '
                 '{:s} [learning_rate={:6.4f}]').format(t, time_string(), epoch, args.epochs,
                                                        need_time, current_learning_rate)
                + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False),
                                                                   100 - recorder.max_accuracy(False)), log)

            # train for one epoch
            tr_acc, tr_acc5, tr_los = train(t, adv_loader, net, optimizer, epoch, args, log)

            # evaluate on validation set
            val_acc, val_los = validate(t, test_loader, net, "Test", log)

            train_loss.append(tr_los)
            train_acc.append(tr_acc)
            test_loss.append(val_los)
            test_acc.append(val_acc)

            recorder.update(epoch, tr_los, tr_acc, val_los, val_acc)

            is_best = False
            if val_acc > best_acc:
                is_best = True
                best_acc = val_acc

            if is_best:
                save_checkpoint(net.state_dict(), iter_dir, "model_best.pth.tar")

            # measure elapsed time
            epoch_time.update(time.time() - start_time)
            start_time = time.time()
            result_png_path = os.path.join(iter_dir, 'results.png')
            recorder.plot_curve(result_png_path)

            train_log = OrderedDict()
            train_log['train_loss'] = train_loss
            train_log['train_acc'] = train_acc
            train_log['test_loss'] = test_loss
            train_log['test_acc'] = test_acc

            pickle.dump(train_log, open(os.path.join(iter_dir, 'pickle_log.pkl'), 'wb'))
            plotting(iter_dir)

    print_log("\nfinish", log)
    log.close()


if __name__ == '__main__':
    main()