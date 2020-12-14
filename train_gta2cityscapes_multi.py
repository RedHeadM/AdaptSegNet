import argparseimport torch
import torch.nn as nn
import datetime
from torch.utils import data, model_zoo
import numpy as np
import pickle
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import sys
import os
import os.path as osp
import matplotlib.pyplot as plt
import random

from model.deeplab_multi import DeeplabMulti
from model.discriminator import FCDiscriminator
from utils.loss import CrossEntropy2d
from dataset.gta5_dataset import GTA5DataSet
from dataset.cityscapes_dataset import cityscapesDataSet
from dataset.mulitview import MulitviewSegLoader
from metrics import eval_metrics, AverageMeter,colorize_mask
from torch.utils import tensorboard
from torchvision import transforms
from tqdm import tqdm
from torchvision.utils import make_grid
IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

MODEL = 'DeepLab'
BATCH_SIZE = 2
ITER_SIZE = 1
NUM_WORKERS = 4
DATA_DIRECTORY = '/media/markus/DATA/mutil_tex_3p/texture_multibot_push_left10050_bot/videos/val'
IGNORE_LABEL = 255
INPUT_SIZE = '300,300'
DATA_DIRECTORY_TARGET = '/media/markus/DATA/mutil_tex_3p/texture_multibot_push_left10050_bot/videos/val'
INPUT_SIZE_TARGET = '300,300'
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 12
NUM_STEPS = 250000
NUM_STEPS_STOP = 150000  # early stopping
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 5000
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005

LEARNING_RATE_D = 1e-4
LAMBDA_SEG = 0.1
LAMBDA_ADV_TARGET1 = 0.0002
LAMBDA_ADV_TARGET2 = 0.001
GAN = 'Vanilla'

TARGET = 'cityscapes'
SET = 'train'


def data_loader_cycle(iterable):
    '''
    Using itertools.cycle has an important drawback, in that it does not shuffle the data after each iteration:
        WARNING  itertools.cycle  does not shuffle the data after each iteratio
        usage         data_iter = iter(cycle(self.data_loader))
        '''
    while True:
        for x in iterable:
            yield x

class ValHelper():
    def __init__(self,gpu,val_loader,model,loss,num_classes=NUM_CLASSES,writer=None):

        self.val_loader =val_loader
        self.gpu=gpu
        self.num_classes=num_classes
        self._reset_metrics()
        self.wrt_step =0
        self.model=model
        self.loss=loss
        self.writer=writer
        # TRANSORMS FOR VISUALIZATION
        self.restore_transform = transforms.Compose([
            # local_transforms.DeNormalize(self.train_loader.MEAN, self.train_loader.STD),
            transforms.ToPILImage()])

        self.viz_transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor()])

    def _reset_metrics(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.total_loss = AverageMeter()
        self.total_inter, self.total_union = 0, 0
        self.total_correct, self.total_label = 0, 0

    def _update_seg_metrics(self, correct, labeled, inter, union):
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union


    def _get_seg_metrics(self, no_bg = False):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        ret =  {
            "Pixel_Accuracy": np.round(pixAcc, 3),
            "Mean_IoU": np.round(mIoU, 3),
            "Class_IoU": dict(zip(range(self.num_classes), np.round(IoU, 3)))
        }

        return ret

    def valid_epoch(self, epoch):
        print('\n###### EVALUATION ######')

        self.wrt_mode = 'val'

        self._reset_metrics()
        tbar = tqdm(self.val_loader, ncols=130)
        with torch.no_grad():
            val_visual = []
            for batch_idx, (data, target,*_) in enumerate(tbar):
                data, target = data.to(self.gpu), target.to(self.gpu)
                # LOSS
                output = self.model(data.cuda(self.gpu))
                loss = self.loss(output, target)
                if isinstance(self.loss, torch.nn.DataParallel):
                    loss = loss.mean()
                self.total_loss.update(loss.item())

                seg_metrics = eval_metrics(output, target, self.num_classes)
                # rm backgroud form loss with lable zero
                self._update_seg_metrics(*seg_metrics)

                # LIST OF IMAGE TO VIZ (15 images)
                if len(val_visual) < 15:
                    target_np = target.data.cpu().numpy()
                    output_np = output.data.max(1)[1].cpu().numpy()
                    val_visual.append([data[0].data.cpu(), target_np[0], output_np[0]])
                else:
                    break# debug
                # PRINT INFO
                pixAcc, mIoU, _ = self._get_seg_metrics().values()
                tbar.set_description('EVAL ({}) | Loss: {:.3f}, PixelAcc: {:.2f}, Mean IoU: {:.2f} |'.format( epoch,
                                                self.total_loss.average,
                                                pixAcc, mIoU))

            # WRTING & VISUALIZING THE MASKS
            val_img = []
            palette = self.val_loader.dataset.palette
            for d, t, o in val_visual:
                d = self.restore_transform(d)
                t, o = colorize_mask(t, palette), colorize_mask(o, palette)
                d, t, o = d.convert('RGB'), t.convert('RGB'), o.convert('RGB')
                [d, t, o] = [self.viz_transform(x) for x in [d, t, o]]
                val_img.extend([d, t, o])
            val_img = torch.stack(val_img, 0)
            val_img = make_grid(val_img.cpu(), nrow=3, padding=5)
            self.writer.add_image(f'{self.wrt_mode}/inputs_targets_predictions', val_img, self.wrt_step)

            # METRICS TO TENSORBOARD
            self.wrt_step = (epoch) * len(self.val_loader)
            self.writer.add_scalar(f'{self.wrt_mode}/loss', self.total_loss.average, self.wrt_step)
            seg_metrics = self._get_seg_metrics(no_bg=True)
            for k, v in list(seg_metrics.items())[:-1]:
                if not isinstance(v,dict ):
                    self.writer.add_scalar(f'{self.wrt_mode}/{k}', v, self.wrt_step)

            log = {
                'val_loss': self.total_loss.average,
                **seg_metrics
            }

        return log


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab")
    parser.add_argument("--target", type=str, default=TARGET,
                        help="available options : cityscapes")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--val-steps", type=int, default=5000,
                        help="val iter")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--lambda-seg", type=float, default=LAMBDA_SEG,
                        help="lambda_seg.")
    parser.add_argument("--lambda-adv-target1", type=float, default=LAMBDA_ADV_TARGET1,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--lambda-adv-target2", type=float, default=LAMBDA_ADV_TARGET2,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose adaptation set.")
    parser.add_argument("--gan", type=str, default=GAN,
                        help="choose the GAN objective.")
    parser.add_argument("-n","--name", type=str, default="name",
                        help="name ")
    return parser.parse_args()


args = get_arguments()



class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=255, reduction='mean'):
        super(CrossEntropyLoss2d, self).__init__()
        self.CE =  nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, output, target):
        loss = self.CE(output, target)
        return loss


def loss_calc(pred, label, gpu,criterion):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).cuda(gpu)

    return criterion(pred, label)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def main():
    """Create the model and start the training."""

    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)

    w, h = map(int, args.input_size_target.split(','))
    input_size_target = (w, h)

    cudnn.enabled = True
    gpu = args.gpu

    # Create network
    if args.model == 'DeepLab':
        model = DeeplabMulti(num_classes=args.num_classes)
        if args.restore_from[:4] == 'http' :
            saved_state_dict = model_zoo.load_url(args.restore_from)
        else:
            saved_state_dict = torch.load(args.restore_from)

        new_params = model.state_dict().copy()

        for i in saved_state_dict:
            # Scale.layer5.conv2d_list.3.weight
            i_parts = i.split('.')
            # print i_parts
            if  args.num_classes != 19 and i_parts[1] != 'layer5':
                new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
        # if args.num_classes !=19:
                # print i_parts
        model.load_state_dict(new_params)
    start_time = datetime.datetime.now().strftime('%m-%d_%H-%M')
    writer_dir = os.path.join("./logs/", args.name, start_time)
    writer = tensorboard.SummaryWriter(writer_dir)

    model.train()
    model.cuda(args.gpu)

    cudnn.benchmark = True

    # init D
    model_D1 = FCDiscriminator(num_classes=args.num_classes)
    model_D2 = FCDiscriminator(num_classes=args.num_classes)

    model_D1.train()
    model_D1.cuda(args.gpu)

    model_D2.train()
    model_D2.cuda(args.gpu)

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    trainloader = data.DataLoader(
        MulitviewSegLoader(num_classes=args.num_classes,
                                    root=args.data_dir,
                                    number_views=2,
                                    view_idx=1,
            # max_iters=args.num_steps * args.iter_size * args.batch_size,
                    # crop_size=input_size,
                    # scale=args.random_scale, mirror=args.random_mirror,
                    img_mean=IMG_MEAN),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    trainloader_iter = iter(data_loader_cycle(trainloader))

    targetloader = data.DataLoader(MulitviewSegLoader(
                                    root=args.data_dir_target,
                                    num_classes=args.num_classes,
                                    number_views=1,
                                    view_idx=0,
                                     # max_iters=args.num_steps * args.iter_size * args.batch_size,
                                     # crop_size=input_size_target,
                                     # scale=False,
                                     # mirror=args.random_mirror,
                                     img_mean=IMG_MEAN,
                                     # set=args.set
                                     ),
                                   batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                   pin_memory=True)


    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear')
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear')
    def mdl_val_func(x):
        return interp_target(model(x)[1])

    targetloader_iter = iter(data_loader_cycle(targetloader))
    val_loader = data.DataLoader(MulitviewSegLoader(
                                    root=args.data_dir_target,
                                    number_views=1,
                                    view_idx=0,
                                    num_classes=args.num_classes,
                                     # max_iters=args.num_steps * args.iter_size * args.batch_size,
                                     # crop_size=input_size_target,
                                     # scale=False,
                                     # mirror=args.random_mirror,
                                     img_mean=IMG_MEAN,
                                     # set=args.set
                                     ),
                                   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                   pin_memory=True)


    criterion = CrossEntropyLoss2d().cuda(args.gpu)
    valhelper= ValHelper(gpu=args.gpu,model=mdl_val_func,val_loader=val_loader,loss=criterion,writer=writer)
    # implement model.optim_parameters(args) to handle different models' lr setting

    optimizer = optim.SGD(model.optim_parameters(args),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    optimizer_D1 = optim.Adam(model_D1.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D1.zero_grad()

    optimizer_D2 = optim.Adam(model_D2.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D2.zero_grad()

    if args.gan == 'Vanilla':
        bce_loss = torch.nn.BCEWithLogitsLoss()
    elif args.gan == 'LS':
        bce_loss = torch.nn.MSELoss()


    # labels for adversarial training
    source_label = 0
    target_label = 1

    for i_iter in range(args.num_steps):

        loss_seg_value1 = 0
        loss_adv_target_value1 = 0
        loss_D_value1 = 0

        loss_seg_value2 = 0
        loss_adv_target_value2 = 0
        loss_D_value2 = 0

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)

        optimizer_D1.zero_grad()
        optimizer_D2.zero_grad()
        adjust_learning_rate_D(optimizer_D1, i_iter)
        adjust_learning_rate_D(optimizer_D2, i_iter)


        for sub_i in range(args.iter_size):

            # train G

            # don't accumulate grads in D
            for param in model_D1.parameters():
                param.requires_grad = False

            for param in model_D2.parameters():
                param.requires_grad = False

            # train with source

            batch = next(trainloader_iter)
            images, labels, *_ = batch
            images = Variable(images).cuda(args.gpu)

            pred1, pred2 = model(images)
            pred1 = interp(pred1)
            pred2 = interp(pred2)

            loss_seg1 = loss_calc(pred1, labels, args.gpu,criterion)
            loss_seg2 = loss_calc(pred2, labels, args.gpu,criterion)
            loss = loss_seg2 + args.lambda_seg * loss_seg1

            # proper normalization
            loss = loss / args.iter_size
            loss.backward()
            loss_seg_value1 += loss_seg1.data.cpu().numpy() / args.iter_size
            loss_seg_value2 += loss_seg2.data.cpu().numpy() / args.iter_size

            # train with target

            batch = next(targetloader_iter)
            images,* _  = batch
            images = Variable(images).cuda(args.gpu)

            pred_target1, pred_target2 = model(images)
            pred_target1 = interp_target(pred_target1)
            pred_target2 = interp_target(pred_target2)

            D_out1 = model_D1(F.softmax(pred_target1))
            D_out2 = model_D2(F.softmax(pred_target2))

            loss_adv_target1 = bce_loss(D_out1,
                                       Variable(torch.FloatTensor(D_out1.data.size()).fill_(source_label)).cuda(
                                           args.gpu))

            loss_adv_target2 = bce_loss(D_out2,
                                        Variable(torch.FloatTensor(D_out2.data.size()).fill_(source_label)).cuda(
                                            args.gpu))

            loss = args.lambda_adv_target1 * loss_adv_target1 + args.lambda_adv_target2 * loss_adv_target2
            loss = loss / args.iter_size
            loss.backward()
            loss_adv_target_value1 += loss_adv_target1.data.cpu().numpy() / args.iter_size
            loss_adv_target_value2 += loss_adv_target2.data.cpu().numpy() / args.iter_size

            # train D

            # bring back requires_grad
            for param in model_D1.parameters():
                param.requires_grad = True

            for param in model_D2.parameters():
                param.requires_grad = True

            # train with source
            pred1 = pred1.detach()
            pred2 = pred2.detach()

            D_out1 = model_D1(F.softmax(pred1))
            D_out2 = model_D2(F.softmax(pred2))

            loss_D1 = bce_loss(D_out1,
                              Variable(torch.FloatTensor(D_out1.data.size()).fill_(source_label)).cuda(args.gpu))

            loss_D2 = bce_loss(D_out2,
                               Variable(torch.FloatTensor(D_out2.data.size()).fill_(source_label)).cuda(args.gpu))

            loss_D1 = loss_D1 / args.iter_size / 2
            loss_D2 = loss_D2 / args.iter_size / 2

            loss_D1.backward()
            loss_D2.backward()

            loss_D_value1 += loss_D1.data.cpu().numpy()
            loss_D_value2 += loss_D2.data.cpu().numpy()

            # train with target
            pred_target1 = pred_target1.detach()
            pred_target2 = pred_target2.detach()

            D_out1 = model_D1(F.softmax(pred_target1))
            D_out2 = model_D2(F.softmax(pred_target2))

            loss_D1 = bce_loss(D_out1,
                              Variable(torch.FloatTensor(D_out1.data.size()).fill_(target_label)).cuda(args.gpu))

            loss_D2 = bce_loss(D_out2,
                               Variable(torch.FloatTensor(D_out2.data.size()).fill_(target_label)).cuda(args.gpu))

            loss_D1 = loss_D1 / args.iter_size / 2
            loss_D2 = loss_D2 / args.iter_size / 2

            loss_D1.backward()
            loss_D2.backward()

            loss_D_value1 += loss_D1.data.cpu().numpy()
            loss_D_value2 += loss_D2.data.cpu().numpy()

        optimizer.step()
        optimizer_D1.step()
        optimizer_D2.step()
        # if i_iter%args.val_steps==0 and i_iter:
        model.eval()
        log=valhelper.valid_epoch(i_iter)
        print('log: {}'.format(log))
        model.train()

        if i_iter%10==0 and i_iter:
            print('exp = {}'.format(args.snapshot_dir))
            print(
            'iter = {0:8d}/{1:8d}, loss_seg1 = {2:.3f} loss_seg2 = {3:.3f} loss_adv1 = {4:.3f}, loss_adv2 = {5:.3f} loss_D1 = {6:.3f} loss_D2 = {7:.3f}'.format(
                i_iter, args.num_steps, loss_seg_value1, loss_seg_value2, loss_adv_target_value1, loss_adv_target_value2, loss_D_value1, loss_D_value2))

        if i_iter >= args.num_steps_stop - 1:
            print ('save model ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps_stop) + '.pth'))
            torch.save(model_D1.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps_stop) + '_D1.pth'))
            torch.save(model_D2.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps_stop) + '_D2.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print ('taking snapshot ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '.pth'))
            torch.save(model_D1.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '_D1.pth'))
            torch.save(model_D2.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '_D2.pth'))


if __name__ == '__main__':
    main()
