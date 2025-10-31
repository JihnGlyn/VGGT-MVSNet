import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import time
from datasets import find_dataset_def
from model.net import *
from utils import *
import sys
from datasets.data_io import read_pfm, save_pfm
import cv2
from plyfile import PlyData, PlyElement
from PIL import Image

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Predict depth, filter, and fuse')
parser.add_argument('--model', default='VGGT-MVSNet', help='select model')
parser.add_argument('--dataset', default='dtu_eval', help='select dataset')
parser.add_argument('--testpath', default='/home/hnu/lxx/dataset/dtu_testing/dtu', help='testing data path')
parser.add_argument('--testlist', default='./lists/dtu/test1.txt', help='testing scan list')
parser.add_argument('--split', default='intermediate', help='select data')
parser.add_argument('--batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--n_views', type=int, default=5, help='num of view')
parser.add_argument('--img_wh', nargs='+', type=int, default=[1600, 1152],
                    help='height and width of the image')
parser.add_argument('--loadckpt', default='./checkpoints/model_000015.ckpt', help='load a specific checkpoint')
parser.add_argument('--outdir', default='./outputs_vggt', help='output dir')
parser.add_argument('--display', action='store_true', help='display depth images and masks')

parser.add_argument('--iteration', type=int, default=2, help='num of iteration')
parser.add_argument('--num_depths', type=int, default=[48, 16, 8], help='num of depth')
parser.add_argument('--depth_interal_ratio', type=float, default=[0.025, 0.025, 0.0125], help='search range')

parser.add_argument('--geo_pixel_thres', type=float, default=1,
                    help='pixel threshold for geometric consistency filtering')
parser.add_argument('--geo_depth_thres', type=float, default=0.01,
                    help='depth threshold for geometric consistency filtering')
parser.add_argument('--photo_thres', type=float, default=10.0, help='threshold for photometric consistency filtering')

# parse arguments and check
args = parser.parse_args()
print("argv:", sys.argv[1:])
print_args(args)

if args.dataset == "dtu_eval":
    img_wh = (1600, 1152)
elif args.dataset == "tanks":
    img_wh = (1920, 1024)
elif args.dataset == "eth3d":
    img_wh = (1920, 1280)
else:
    img_wh = (args.img_wh[0], args.img_wh[1])  # custom dataset


# read intrinsics and extrinsics
def read_camera_parameters(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))

    return intrinsics, extrinsics


# read an image
def read_img(filename, img_wh):
    img = Image.open(filename)
    # scale 0~255 to 0~1
    np_img = np.array(img, dtype=np.float32) / 255.
    original_h, original_w, _ = np_img.shape
    np_img = cv2.resize(np_img, img_wh, interpolation=cv2.INTER_LINEAR)
    return np_img, original_h, original_w


# save a binary mask
def save_mask(filename, mask):
    assert mask.dtype == np.bool_
    mask = mask.astype(np.uint8) * 255
    Image.fromarray(mask).save(filename)


def save_depth_img(filename, depth):
    # assert mask.dtype == np.bool
    depth = depth.astype(np.float32) * 255
    Image.fromarray(depth).save(filename)


def read_pair_file(filename):
    data = []
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        # 49 viewpoints
        for view_idx in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            if len(src_views) != 0:
                data.append((ref_view, src_views))
    return data


def save_camera_matrices(extrinsics, intrinsic, cam_filename):
    """

    :param extrinsics: [N, 4, 4]
    :param intrinsic: [N, 3, 3]
    :param cam_filename:
    """
    N = extrinsics.shape[0]
    if not os.path.exists(cam_filename):
        os.makedirs(cam_filename)
    # cams / {: 0 > 8}_cam.txt
    for i in range(N):
        filename = os.path.join(cam_filename, "{:0>8}.txt".format(i))
        with open(filename, 'w') as f:
            f.write('extrinsic\n')
            for row in extrinsics[i]:
                row_str = ' '.join(map(str, row))
                f.write(row_str + '\n')
            f.write('\n')
            f.write('intrinsic\n')
            for row in intrinsic[i]:
                row_str = ' '.join(map(str, row))
                f.write(row_str + '\n')

    print(f"Camera parameters have saved in folder: {cam_filename}")


# run MVS model to save depth maps
def save_depth():
    # dataset, dataloader
    MVSDataset = find_dataset_def(args.dataset)
    if args.dataset == "dtu_eval":
        test_dataset = MVSDataset(args.testpath, args.testlist, args.n_views, img_wh)
    elif args.dataset == "tanks":
        test_dataset = MVSDataset(args.testpath, args.n_views, img_wh, args.split)
    elif args.dataset == "eth3d":
        test_dataset = MVSDataset(args.testpath, args.split, args.n_views, img_wh)
    else:
        test_dataset = MVSDataset(args.testpath, args.n_views, img_wh)
    TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=4, drop_last=False)

    # model
    model = VGGT4MVS()
    model = nn.DataParallel(model)
    model.cuda()

    # load checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt, map_location='cuda')["model"]
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    vggt_model = VGGT()
    LOCAL_MODEL = True
    if LOCAL_MODEL:
        vggt_model_path = "./vggtckpt/model.pt"
        vggt_model.load_state_dict(torch.load(vggt_model_path, map_location='cuda'))
    else:
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        vggt_model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    vggt_model.eval()
    vggt_model = vggt_model.cuda()
    print(f"VGGT Model loaded")

    with torch.no_grad():
        for batch_idx, sample in enumerate(TestImgLoader):
            start_time = time.time()
            sample_cuda = tocuda(sample)
            # TODO:
            vggt_outputs = run_VGGT(vggt_model, sample_cuda["imgs"]["level_2"])
            outputs = model(vggt_outputs=vggt_outputs,
                            imgs=sample_cuda["imgs"],
                            num_depths=args.num_depths,
                            depth_interal_ratio=args.depth_interal_ratio,
                            iteration=args.iteration,
                            pair=sample_cuda["pair_view"],
                            )

            depth_list = tensor2numpy(outputs["depths"])
            conf_list = tensor2numpy(outputs["vggt_confs"])

            intrinsics = tensor2numpy(outputs["intrinsic"])
            intrinsics = np.squeeze(intrinsics, 0)
            extrinsics = tensor2numpy(outputs["extrinsic"])
            extrinsics = np.squeeze(extrinsics, 0)
            del sample_cuda
            print('Iter {}/{}, time = {:.3f}'.format(batch_idx, len(TestImgLoader), time.time() - start_time))
            filenames = sample["filename"]
            scannames = sample["scan"]
            # save depth maps and confidence maps
            for filename, scanname in zip(filenames, scannames):
                cam_filename = os.path.join(args.outdir, scanname, 'cams')
                save_camera_matrices(extrinsics, intrinsics, cam_filename)
                idx = 0
                for depth_est, confidence in zip(depth_list, conf_list):
                    depth_filename = os.path.join(args.outdir, filename.format('depth_est', idx, '.pfm'))
                    confidence_filename = os.path.join(args.outdir, filename.format('confidence', idx, '.pfm'))
                    os.makedirs(depth_filename.rsplit('/', 1)[0], exist_ok=True)
                    os.makedirs(confidence_filename.rsplit('/', 1)[0], exist_ok=True)
                    # save depth maps
                    depth_est = np.squeeze(depth_est, 0)
                    save_pfm(depth_filename, depth_est)
                    # save confidence maps
                    confidence = np.squeeze(confidence, 0)
                    save_pfm(confidence_filename, confidence)
                    idx += 1


# project the reference point cloud into the source view, then project back
def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    ## step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    # reference 3D space
    xyz_ref = np.matmul(np.linalg.inv(intrinsics_ref),
                        np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1]))
    # source 3D space
    xyz_src = np.matmul(np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)),
                        np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]
    # source view x, y
    K_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)
    # mask = sampled_depth_src > 0

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = np.matmul(np.linalg.inv(intrinsics_src),
                        np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]))
    # reference 3D space
    xyz_reprojected = np.matmul(np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)),
                                np.vstack((xyz_src, np.ones_like(x_ref))))[:3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:2] / (K_xyz_reprojected[2:3] + 1e-6)
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src


def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src,
                                geo_pixel_thres, geo_depth_thres):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(depth_ref,
                                                                                                 intrinsics_ref,
                                                                                                 extrinsics_ref,
                                                                                                 depth_src,
                                                                                                 intrinsics_src,
                                                                                                 extrinsics_src)

    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref

    mask = np.logical_and(dist < geo_pixel_thres, relative_depth_diff < geo_depth_thres)
    depth_reprojected[~mask] = 0

    return mask, depth_reprojected, x2d_src, y2d_src


def filter_depth(scan_folder, out_folder, scan, plyfilename, geo_pixel_thres, geo_depth_thres, photo_thres, img_wh,
                 geo_mask_thres=3):
    # the pair file
    pair_file = os.path.join(scan_folder, "pair.txt")
    # for the final point cloud
    vertexs = []
    vertex_colors = []

    pair_data = read_pair_file(pair_file)
    nviews = len(pair_data)

    # for each reference view and the corresponding source views
    for ref_view, src_views in pair_data:
        # load the camera parameters
        ref_intrinsics, ref_extrinsics = read_camera_parameters(
            os.path.join(out_folder, 'cams/{:0>8}.txt'.format(ref_view)))
        ref_img, original_h, original_w = read_img(os.path.join(scan_folder, 'images/{:0>8}.jpg'.format(ref_view)),
                                                   img_wh)
        ref_depth_est = read_pfm(os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(ref_view)))[0]
        confidence = read_pfm(os.path.join(out_folder, 'confidence/{:0>8}.pfm'.format(ref_view)))[0]
        new_h, new_w = ref_depth_est.shape
        ref_img = cv2.resize(ref_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        photo_mask = confidence > photo_thres

        all_srcview_depth_ests = []
        # compute the geometric mask
        geo_mask_sum = 0

        for src_view in src_views:
            # camera parameters of the source view
            src_intrinsics, src_extrinsics = read_camera_parameters(
                os.path.join(out_folder, 'cams/{:0>8}.txt'.format(src_view)))
            _, original_h, original_w = read_img(os.path.join(scan_folder, 'images/{:0>8}.jpg'.format(src_view)),
                                                 img_wh)

            # the estimated depth of the source view
            src_depth_est = read_pfm(os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(src_view)))[0]

            geo_mask, depth_reprojected, _, _ = check_geometric_consistency(ref_depth_est, ref_intrinsics,
                                                                            ref_extrinsics,
                                                                            src_depth_est,
                                                                            src_intrinsics, src_extrinsics,
                                                                            geo_pixel_thres, geo_depth_thres)
            geo_mask_sum += geo_mask.astype(np.int32)
            all_srcview_depth_ests.append(depth_reprojected)

        depth_est_averaged = (sum(all_srcview_depth_ests) + ref_depth_est) / (geo_mask_sum + 1)
        geo_mask = geo_mask_sum >= geo_mask_thres
        final_mask = np.logical_and(photo_mask, geo_mask)
        # final_mask = np.logical_and(photo_mask, 1)

        os.makedirs(os.path.join(out_folder, "mask"), exist_ok=True)
        save_mask(os.path.join(out_folder, "mask/{:0>8}_photo.png".format(ref_view)), photo_mask)
        save_mask(os.path.join(out_folder, "mask/{:0>8}_geo.png".format(ref_view)), geo_mask)
        save_mask(os.path.join(out_folder, "mask/{:0>8}_final.png".format(ref_view)), final_mask)

        print("processing {}, ref-view{:0>2}, geo_mask:{:3f} photo_mask:{:3f} final_mask: {:3f}".format(scan_folder,
                                                                                                        ref_view,
                                                                                                        geo_mask.mean(),
                                                                                                        photo_mask.mean(),
                                                                                                        final_mask.mean()))

        if args.display:
            cv2.imshow('ref_img', ref_img[:, :, ::-1])
            cv2.imshow('ref_depth', ref_depth_est / np.max(ref_depth_est))
            cv2.imshow('ref_depth * photo_mask', ref_depth_est * photo_mask.astype(np.float32) / np.max(ref_depth_est))
            cv2.imshow('ref_depth * geo_mask', ref_depth_est * geo_mask.astype(np.float32) / np.max(ref_depth_est))
            cv2.imshow('ref_depth * mask', ref_depth_est * final_mask.astype(np.float32) / np.max(ref_depth_est))
            cv2.waitKey(0)

        height, width = depth_est_averaged.shape[:2]
        x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))

        valid_points = final_mask
        # print("valid_points", valid_points.mean())
        x, y, depth = x[valid_points], y[valid_points], depth_est_averaged[valid_points]

        color = ref_img[valid_points]
        xyz_ref = np.matmul(np.linalg.inv(ref_intrinsics),
                            np.vstack((x, y, np.ones_like(x))) * depth)
        xyz_world = np.matmul(np.linalg.inv(ref_extrinsics),
                              np.vstack((xyz_ref, np.ones_like(x))))[:3]
        vertexs.append(xyz_world.transpose((1, 0)))
        vertex_colors.append((color * 255).astype(np.uint8))

    vertexs = np.concatenate(vertexs, axis=0)
    vertex_colors = np.concatenate(vertex_colors, axis=0)
    vertexs = np.array([tuple(v) for v in vertexs], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    vertex_colors = np.array([tuple(v) for v in vertex_colors], dtype=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    vertex_all = np.empty(len(vertexs), vertexs.dtype.descr + vertex_colors.dtype.descr)
    for prop in vertexs.dtype.names:
        vertex_all[prop] = vertexs[prop]
    for prop in vertex_colors.dtype.names:
        vertex_all[prop] = vertex_colors[prop]

    el = PlyElement.describe(vertex_all, 'vertex')
    PlyData([el]).write(plyfilename)
    print("saving the final model to", plyfilename)


if __name__ == '__main__':
    save_depth()
    if args.dataset == "dtu_eval":
        with open(args.testlist) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

        for scan in scans:
            scan_id = int(scan[4:])
            scan_folder = os.path.join(args.testpath, scan)
            out_folder = os.path.join(args.outdir, scan)
            filter_depth(scan_folder, out_folder, scan,
                         os.path.join(args.outdir, 'vggtmvs{:0>3}_l3.ply'.format(scan_id)),
                         args.geo_pixel_thres, args.geo_depth_thres, args.photo_thres, img_wh, 4)
    elif args.dataset == "tanks":
        # intermediate dataset
        if args.split == "intermediate":
            scans = ['Family', 'Francis', 'Horse', 'Lighthouse',
                     'M60', 'Panther', 'Playground', 'Train']
            geo_mask_thres = {'Family': 5,
                              'Francis': 6,
                              'Horse': 5,
                              'Lighthouse': 6,
                              'M60': 5,
                              'Panther': 5,
                              'Playground': 5,
                              'Train': 5}

            for scan in scans:
                scan_folder = os.path.join(args.testpath, args.split, scan)
                out_folder = os.path.join(args.outdir, scan)

                filter_depth(scan_folder, out_folder, os.path.join(args.outdir, scan + '.ply'),
                             args.geo_pixel_thres, args.geo_depth_thres, args.photo_thres, img_wh, geo_mask_thres[scan])

        # advanced dataset
        elif args.split == "advanced":
            scans = ['Auditorium', 'Ballroom', 'Courtroom',
                     'Museum', 'Palace', 'Temple']
            geo_mask_thres = {'Auditorium': 3,
                              'Ballroom': 4,
                              'Courtroom': 4,
                              'Museum': 4,
                              'Palace': 5,
                              'Temple': 4}

            for scan in scans:
                scan_folder = os.path.join(args.testpath, args.split, scan)
                out_folder = os.path.join(args.outdir, scan)
                filter_depth(scan_folder, out_folder, os.path.join(args.outdir, scan + '.ply'),
                             args.geo_pixel_thres, args.geo_depth_thres, args.photo_thres, img_wh, geo_mask_thres[scan])

    elif args.dataset == "eth3d":
        if args.split == "test":
            scans = ['botanical_garden', 'boulders', 'bridge', 'door',
                     'exhibition_hall', 'lecture_room', 'living_room', 'lounge',
                     'observatory', 'old_computer', 'statue', 'terrace_2']

            geo_mask_thres = {'botanical_garden': 1,  # 30 images, outdoor
                              'boulders': 1,  # 26 images, outdoor
                              'bridge': 2,  # 110 images, outdoor
                              'door': 2,  # 6 images, indoor
                              'exhibition_hall': 2,  # 68 images, indoor
                              'lecture_room': 2,  # 23 images, indoor
                              'living_room': 2,  # 65 images, indoor
                              'lounge': 1,  # 10 images, indoor
                              'observatory': 2,  # 27 images, outdoor
                              'old_computer': 2,  # 54 images, indoor
                              'statue': 2,  # 10 images, indoor
                              'terrace_2': 2  # 13 images, outdoor
                              }
            for scan in scans:
                start_time = time.time()
                scan_folder = os.path.join(args.testpath, scan)
                out_folder = os.path.join(args.outdir, scan)
                filter_depth(scan_folder, out_folder, os.path.join(args.outdir, scan + '.ply'),
                             args.geo_pixel_thres, args.geo_depth_thres, args.photo_thres, img_wh, geo_mask_thres[scan])
                print('scan: ' + scan + ' time = {:3f}'.format(time.time() - start_time))

        elif args.split == "train":
            scans = ['courtyard', 'delivery_area', 'electro', 'facade',
                     'kicker', 'meadow', 'office', 'pipes', 'playground',
                     'relief', 'relief_2', 'terrace', 'terrains']

            geo_mask_thres = {'courtyard': 1,  # 38 images, outdoor
                              'delivery_area': 2,  # 44 images, indoor
                              'electro': 1,  # 45 images, outdoor
                              'facade': 2,  # 76 images, outdoor
                              'kicker': 1,  # 31 images, indoor
                              'meadow': 1,  # 15 images, outdoor
                              'office': 1,  # 26 images, indoor
                              'pipes': 1,  # 14 images, indoor
                              'playground': 1,  # 38 images, outdoor
                              'relief': 1,  # 31 images, indoor
                              'relief_2': 1,  # 31 images, indoor
                              'terrace': 1,  # 23 images, outdoor
                              'terrains': 2  # 42 images, indoor
                              }

            for scan in scans:
                start_time = time.time()
                scan_folder = os.path.join(args.testpath, scan)
                out_folder = os.path.join(args.outdir, scan)
                filter_depth(scan_folder, out_folder, os.path.join(args.outdir, scan + '.ply'),
                             args.geo_pixel_thres, args.geo_depth_thres, args.photo_thres, img_wh, geo_mask_thres[scan])
                print('scan: ' + scan + ' time = {:3f}'.format(time.time() - start_time))
    else:
        filter_depth(args.testpath, args.outdir, os.path.join(args.outdir, 'custom.ply'),
                     args.geo_pixel_thres, args.geo_depth_thres, args.photo_thres, img_wh, geo_mask_thres=3)
