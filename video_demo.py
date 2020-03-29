import os
import time
import argparse
import numpy as np
import pprint
import torch
import cv2

import torchvision.transforms as transforms
import torchvision.datasets as dset
# from scipy.misc import imread
import PIL.Image as Image
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
from utils import AverageMeter


def parse_args():
    parser = argparse.ArgumentParser(
        'Running faster-rcnn detection on video dataset')

    parser.add_argument(
        '--dataset',
        default='pascal_voc',
        type=str,
        help='pascal_voc or coco')
    parser.add_argument(
        '--class_txt',
        default='',
        type=str,
        help='a txt file listing class names')
    parser.add_argument('--cfg_file', default='cfgs/vgg16.yml', type=str)
    parser.add_argument('--net', default='res101', type=str)
    parser.add_argument(
        '--load_dir',
        help='directory to load models',
        type=str)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression', action='store_true')
    parser.add_argument('--parallel_type',
                        help='which part of model to parallel, 0: all, 1: model before roi pool',
                        type=int, default=0)
    parser.add_argument(
        '--checksession',
        help='checksession to load model',
        default=1,
        type=int)
    parser.add_argument(
        '--checkepoch',
        help='checkepoch to load network',
        default=10021,
        type=int)
    parser.add_argument('--bs', help='batch size', default=1, type=int)
    parser.add_argument(
        '--trained_model',
        help='path of trained model to use',
        type=str)
    parser.add_argument('--video_path', type=str)
    parser.add_argument('--outdir', type=str)

    args = parser.parse_args()

    return args


def _get_image_blob(im):
    """Converts an image into a network input.
    Arguments:
        im (ndarray): a color image in BGR order
    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(
            im_orig,
            None,
            None,
            fx=im_scale,
            fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)


def read_class_from_txt(txt_file):
    classes = ['__background__']
    with open(txt_file) as infile:
        lines = infile.readlines()
    for line in lines:
        line = line.strip()
        classes.append(line)

    return classes


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    cfg.USE_GPU_NMS = args.cuda

    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    """
    # ============== pascal labels ======================
    pascal_classes = np.asarray(['__background__', 'aeroplane', 'bicycle', 'bird', 'boat',
                                 'bottle', 'bus', 'car', 'cat', 'chair',
                                 'cow', 'diningtable', 'dog', 'horse',
                                 'motorbike', 'person', 'pottedplant',
                                 'sheep', 'sofa', 'train', 'tvmonitor'])

    # ============== COCO labels ========================
    coco_classes = np.asarray(['person', 'bicycle', 'car', 'motorcycle', 'airplane'])
    """
    if args.dataset == 'pascal_voc':
        classes = np.asarray(['__background__', 'aeroplane', 'bicycle', 'bird', 'boat',
                              'bottle', 'bus', 'car', 'cat', 'chair',
                              'cow', 'diningtable', 'dog', 'horse',
                              'motorbike', 'person', 'pottedplant',
                              'sheep', 'sofa', 'train', 'tvmonitor'])
        assert len(classes) == 21, 'Fatal Error, class number not correct!'
        person_id = 15
    else:
        classes = np.asarray(read_class_from_txt(args.class_txt))
        assert len(classes) == 81, 'Fatal Error, class number not correct!'
        person_id = 1

    # initialize network
    if args.net == 'vgg16':
        fasterRCNN = vgg16(
            classes,
            pretrained=False,
            class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fasterRCNN = resnet(classes, 101, pretrained=False,
                            class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        fasterRCNN = resnet(classes, 50, pretrained=False,
                            class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        fasterRCNN = resnet(classes, 152, pretrained=False,
                            class_agnostic=args.class_agnostic)

    fasterRCNN.create_architecture()
    print('load checkpoint {}'.format(args.trained_model))
    if args.cuda > 0:
        checkpoint = torch.load(args.trained_model)
    else:
        checkpoint = torch.load(
            args.trained_model, map_location=(
                lambda storage, loc: storage))
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']

    print('load model successfully!')

    # initialize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    if args.cuda > 0:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    if args.cuda > 0:
        cfg.CUDA = True
        fasterRCNN.cuda()

    fasterRCNN.eval()

    start = time.time()
    mas_per_image = 100
    thresh = 0.05

    '''video writer'''
    cap = cv2.VideoCapture(args.video_path)
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    output_vids = os.path.join(args.outdir, 'detected.avi')
    writer = cv2.VideoWriter(output_vids, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 25,
                             (int(frame_width), int(frame_height)))

    '''meter'''
    detect_meter = AverageMeter()
    nms_meter = AverageMeter()

    print('Start detecting...')
    num_frame = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret is False:
            break
        if frame is None:
            continue
        num_frame += 1
        im_in = np.array(frame)
        if len(im_in.shape) == 2:
            im_in = im_in[:, :, np.newaxis]
            im_in = np.concatenate((im_in, im_in, im_in), axis=2)
        # rgb -> bgr
        # im = im_in[:, :, ::-1]
        im = im_in

        blobs, im_scales = _get_image_blob(im)
        assert len(im_scales) == 1, "Only single-image batch implementated"
        im_blob = blobs
        im_info_np = np.array(
            [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

        im_data_pt = torch.from_numpy(im_blob)
        im_data_pt = im_data_pt.permute(0, 3, 1, 2)  # (NCHW)
        im_info_pt = torch.from_numpy(im_info_np)

        with torch.no_grad():
            im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
            im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
            gt_boxes.resize_(1, 1, 5).zero_()
            num_boxes.resize_(1).zero_()

        det_tic = time.time()

        rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

        scores = cls_prob.detach()
        boxes = rois.data[:, :, 1:5]

        if cfg.TEST.BBOX_REG:
            # Apply bbox regression delta
            box_deltas = bbox_pred.detach()
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdv
                if args.class_agnostic:
                    if args.cuda > 0:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                            + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                            + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    if args.cuda > 0:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                            + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    else:
                        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                            + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                    box_deltas = box_deltas.view(1, -1, 4 * len(classes))
                pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
            else:
                # Simply repeat the boxes, once for each class
                pred_boxes = np.tile(boxes, (1, scores.shape[1]))

            pred_boxes /= im_scales[0]

            scores = scores.squeeze()
            pred_boxes = pred_boxes.squeeze()
            detect_meter.update(time.time() - det_tic)

            im2show = np.copy(im)
            misc_tic = time.time()
            # for j in range(0, len(classes)):
            #     inds = torch.nonzero(scores[:, j] > thresh).view(-1)
            #     # if there is a det
            #     if inds.numel() > 0:
            #         cls_scores = scores[:, j][inds]
            #         _, order = torch.sort(cls_scores, 0, True)
            #         if args.class_agnostic:
            #             cls_boxes = pred_boxes[inds, :]
            #         else:
            #             cls_boxes = pred_boxes[inds][:, j * 4: (j + 1) * 4]

            #         cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
            #         cls_dets = cls_dets[order]
            #         keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
            #         cls_dets = cls_dets[keep.view(-1).long()]
            #         im2show = vis_detections(im2show, pascal_classes[j], cls_dets.cpu().numpy(), 0.5)

            inds = torch.nonzero(scores[:, person_id] > thresh).view(-1)
            # if there's a det
            if inds.numel() > 0:
                cls_scores = scores[:, person_id][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if args.class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4: (j + 1) * 4]
                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_dets[order, :], cls_scores[order], cfg.TEST.NMS)
                cls_dets = cls_dets[keep.view(-1).long()]
                im2show = vis_detections(
                    im2show, classes[person_id], cls_dets.cpu().numpy(), 0.5)

            nms_meter.update(time.time() - misc_tic)

            print('Frame {0}: {1} persons detected, detect time {det_meter.val:.3f}({det_meter.avg:.3f})\t'
                  'nms time {nms_meter.val:.3f}({nms_meter.avg:.3f})'.format(
                      num_frame, len(cls_dets), det_meter=detect_meter, nms_meter=nms_meter
                  ))

            # print('writing frame_{}'.format(num_frame))
            writer.write(im2show)

    print('Detection complete!')
    cap.release()
    writer.release()
