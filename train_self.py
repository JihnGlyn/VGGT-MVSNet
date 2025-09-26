import torch.nn.parallel
import argparse
import datetime
import gc
import json
import os
import sys
import time

import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# from tensorboardX import SummaryWriter
from datasets import find_dataset_def
from model.loss import *
from model.net import *
from utils import *

cudnn.benchmark = True

num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1


def load_config_from_json(myparser, json_file_path='./config/default.json'):
    args = myparser.parse_args()
    try:
        with open(json_file_path, 'r') as f:
            config = json.load(f)
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
            else:
                print(f"Warning: Key '{key}' in JSON file is not a valid argument.")
    except FileNotFoundError:
        print(f"Error: JSON file '{json_file_path}' not found.")
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON file '{json_file_path}'.")

    return args


def train(model, model_loss, optimizer, TrainImgLoader, TestImgLoader, start_epoch, args):
    milestones = [len(TrainImgLoader) * int(epoch_idx) for epoch_idx in args.lrepochs.split(':')[0].split(',')]
    lr_gamma = 1 / float(args.lrepochs.split(':')[1])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=lr_gamma, last_epoch=start_epoch - 1)

    for epoch_idx in range(start_epoch, args.epochs):
        print("Epoch {}:".format(epoch_idx + 1))
        global_step = len(TrainImgLoader) * epoch_idx

        for batch_idx, sample in enumerate(TrainImgLoader):
            start_time = time.time()
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs, image_outputs = train_sample(model, model_loss, optimizer, sample, True, args)
            lr_scheduler.step()
            if do_summary:
                save_scalars(logger, 'train', scalar_outputs, global_step)
                save_images(logger, 'train', image_outputs, global_step)
                print(
                    "Epoch:{}/{}, Iter:{}/{}, lr:{:.6f}, train loss:{:.3f}, recon loss:{:.3f}, ssim loss:{:.3f}, smooth loss:{:.3f}, time:{:.3f}".format(
                        epoch_idx, args.epochs, batch_idx, len(TrainImgLoader),
                        optimizer.param_groups[0]["lr"], loss,
                        scalar_outputs['recon_loss'],
                        scalar_outputs['ssim_loss'],
                        scalar_outputs['smooth_loss'],
                        time.time() - start_time))
            del scalar_outputs, image_outputs

        for batch_idx, sample in enumerate(TestImgLoader):
            start_time = time.time()
            global_step = len(TestImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs, image_outputs = train_sample(model, model_loss, optimizer, sample, False, args)
            if do_summary:
                save_scalars(logger, 'test', scalar_outputs, global_step)
                save_images(logger, 'test', image_outputs, global_step)
                print(
                    "Epoch:{}/{}, Iter:{}/{}, lr:{:.6f}, test loss:{:.3f}, recon loss:{:.3f}, ssim loss:{:.3f}, smooth loss:{:.3f}, time:{:.3f}".format(
                        epoch_idx, args.epochs, batch_idx, len(TestImgLoader),
                        optimizer.param_groups[0]["lr"], loss,
                        scalar_outputs['recon_loss'],
                        scalar_outputs['ssim_loss'],
                        scalar_outputs['smooth_loss'],
                        time.time() - start_time))
            del scalar_outputs, image_outputs

        # checkpoint
        if (epoch_idx + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch_idx,
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict()},
                "{}/model_{:0>6}.ckpt".format(args.logdir, epoch_idx))
        gc.collect()


def train_sample(model, model_loss, optimizer, sample, is_training, args):
    if is_training:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()

    sample_cuda = tocuda(sample)

    outputs = model(model=vggt_model,
                    imgs=sample_cuda["imgs"],
                    num_depths=args.num_depths,
                    depth_interal_ratio=args.depth_interal_ratio,
                    iteration=args.iteration,
                    )
    depth_est = outputs["depth"]

    loss, recon_loss, ssim_loss, smooth_loss = model_loss(depth_est, sample_cuda["imgs"],
                                                          outputs["proj"],
                                                          dlossw=[float(e) for e in args.dlossw.split(",") if e])

    if np.isnan(loss.item()):
        raise NanError

    loss.backward()
    optimizer.step()

    scalar_outputs = {"loss": loss,
                      "recon_loss": recon_loss,
                      "ssim_loss": ssim_loss,
                      "smooth_loss": smooth_loss,
                      }

    image_outputs = {"depth_1": depth_est[0],
                     "depth_2": depth_est[1],
                     "depth_3": depth_est[2],
                     "ref_img": sample["imgs"][:, 0],
                     }

    return tensor2float(scalar_outputs["loss"]), tensor2float(scalar_outputs), tensor2numpy(image_outputs)


if __name__ == '__main__':
    # args = myparser.parse_args()
    parser = argparse.ArgumentParser(description='A PyTorch Implementation of VGGT4MVS')
    parser.add_argument('--mode', default='train', help='train or test', choices=['train'])
    parser.add_argument('--device', default='cuda', help='select model')
    parser.add_argument('--dataset', default='dtu', help='select dataset')
    parser.add_argument('--trainpath', help='train datapath')
    parser.add_argument('--testpath', help='test datapath')
    parser.add_argument('--trainlist', help='train list')
    parser.add_argument('--testlist', help='test list')
    parser.add_argument('--epochs', type=int, default=16, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lrepochs', type=str, default="4,6,8:2",
                        help='epoch ids to downscale lr and the downscale rate')
    parser.add_argument('--wd', type=float, default=0.0001, help='weight decay')
    parser.add_argument('--nviews', type=int, default=5, help='total number of views')
    parser.add_argument('--num_depths', type=int, default=8, help='total number of depths')
    parser.add_argument('--depth_interal_ratio', type=float, default=0.25, help='search range')
    parser.add_argument('--batch_size', type=int, default=1, help='train batch size')
    parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
    parser.add_argument('--logdir', default='./checkpoints', help='the directory to save checkpoints/logs')
    parser.add_argument('--resume', action='store_true', help='continue to train the model')
    parser.add_argument('--summary_freq', type=int, default=10, help='print and summary frequency')
    parser.add_argument('--save_freq', type=int, default=1, help='save checkpoint frequency')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
    parser.add_argument('--pin_m', action='store_true', help='data loader pin memory')
    parser.add_argument('--dlossw', type=str, default="0.5,1.0,2.0", help='depth loss weight for different stage')
    # json_file = './config/default.json'
    # args = load_config_from_json(parser, json_file)
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.resume:
        assert args.loadckpt is None
    if args.testpath is None:
        args.testpath = args.trainpath

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # create logger for mode "train" and "testall"
    if not os.path.isdir(args.logdir):
        os.makedirs(args.logdir)
    current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    print("current time", current_time_str)
    print("creating new summary file")
    logger = SummaryWriter(args.logdir)
    print("argv:", sys.argv[1:])
    print_args(args)

    vggt_model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    vggt_model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    vggt_model.eval()
    vggt_model = vggt_model.to(device)
    print(f"VGGT Model loaded")
    model = VGGT4MVS()
    model.to(device)

    model_loss = unsup_loss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999),
                           weight_decay=args.wd)

    # load parameters
    start_epoch = 0
    if args.resume:
        saved_models = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
        saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        # use the latest checkpoint file
        loadckpt = os.path.join(args.logdir, saved_models[-1])
        print("resuming", loadckpt)
        state_dict = torch.load(loadckpt, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        start_epoch = state_dict['epoch'] + 1
    elif args.loadckpt:
        # load checkpoint file specified by args.loadckpt
        print("loading model {}".format(args.loadckpt))
        state_dict = torch.load(args.loadckpt, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict['model'])

    print("start at epoch {}".format(start_epoch))
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    if torch.cuda.is_available():
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    # dataset, dataloader
    MVSDataset = find_dataset_def(args.dataset)
    train_dataset = MVSDataset(args.trainpath, args.trainlist, "train", args.nviews, robust_train=True)
    test_dataset = MVSDataset(args.testpath, args.testlist, "test", args.nviews, robust_train=False)

    TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=8, drop_last=True,
                                pin_memory=args.pin_m)
    TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=4, drop_last=False,
                               pin_memory=args.pin_m)

    train(model, model_loss, optimizer, TrainImgLoader, TestImgLoader, start_epoch, args)

