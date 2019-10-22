import argparse

import torch
from torch.nn import functional as F
import numpy as np
import torch.backends.cudnn as cudnn
import os
import os.path as osp
import glob
import cv2
from PIL import Image
import pickle
from networks.deeplabv3 import ResNetDeepLabv3
from networks.nets import AnchorDiffNet, ConcatNet, InterFrameNet, IntraFrameNet

import timeit
from metrics.iou import get_iou

from sklearn.metrics import precision_recall_curve


start = timeit.default_timer()

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
BACKBONE = 'ResNet101'
BN_TYPE = 'sync'
DATA_DIRECTORY = './data/DAVIS/'
EMBEDDING_SIZE = 128
THRESHOLD = 0.5  # the threshold over raw scores (not the output of a sigmoid function)
PYRAMID_POOLING = 'deeplabv3'
VISUALIZE = False  # False True
MS_MIRROR = False  # False True
INSTANCE_PRUNING = False  # False True
PARENT_DIR_WEIGHTS = './snapshots/'
SAVE_MASK = True  # False True
SAVE_MASK_DIR = './pred_masks/'
EVAL_SAL = False  # False True
SAVE_HEATMAP_DIR = './pred_heatmaps/'
# MODEL = 'base'
# MODEL = 'concat'
# MODEL = 'intra'
# MODEL = 'inter'
MODEL = 'ad'


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def bool2str(v):
    if v:
        return 'True'
    else:
        return 'False'


def get_arguments():
    parser = argparse.ArgumentParser(description="Anchor Diffusion VOS Test")
    parser.add_argument("--backbone", type=str, default=BACKBONE,
                        help="Feature encoder.")
    parser.add_argument("--bn-type", type=str, default=BN_TYPE,
                        help="BatchNorm MODE, [old/sync].")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the data directory.")
    parser.add_argument("--embedding-size", type=int, default=EMBEDDING_SIZE,
                        help="Number of dimensions along the channel axis of pixel embeddings.")
    parser.add_argument("--eval-sal", type=str2bool, default=bool2str(EVAL_SAL),
                        help="Whether to report MAE and F-score.")
    parser.add_argument("--inst-prune", type=str2bool, default=bool2str(INSTANCE_PRUNING),
                        help="Whether to post-process the results with instance pruning")
    parser.add_argument("--ms-mirror", type=str2bool, default=bool2str(MS_MIRROR),
                        help="Whether to mirror and re-scale the input image.")
    parser.add_argument("--model", type=str, default=MODEL, help="Overall models.")
    parser.add_argument("--pyramid-pooling", type=str, default=PYRAMID_POOLING,
                        help="Pyramid pooling methods.")
    parser.add_argument("--parent-dir-weights", type=str, default=PARENT_DIR_WEIGHTS,
                        help="Parent directory of pre-trained weights")
    parser.add_argument("--save-heatmap-dir", type=str, default=SAVE_HEATMAP_DIR,
                        help="Path to save the outputs of sigmoid for MAE and F-score evaluation.")
    parser.add_argument("--save-mask", type=str2bool, default=bool2str(SAVE_MASK),
                        help="Whether to save the predicted masks.")
    parser.add_argument("--save-mask-dir", type=str, default=SAVE_MASK_DIR,
                        help="Path to save the predicted masks.")
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help="Threshold on the raw scores (the logits before sigmoid normalization).")
    parser.add_argument("--visualize", type=str2bool, default=bool2str(VISUALIZE),
                        help="Whether to visualize the predicted masks during inference.")
    parser.add_argument("--video", type=str, default='',
                        help="If non-empty, then run inference on the specified video.")
    return parser.parse_args()


args = get_arguments()


def main():
    cudnn.enabled = True

    if args.model == 'base':
        model = ResNetDeepLabv3(backbone=args.backbone)
    elif args.model == 'intra':
        model = IntraFrameNet(backbone=args.backbone, pyramid_pooling=args.pyramid_pooling,
                              embedding=args.embedding_size, batch_mode='sync')
    elif args.model == 'inter':
        model = InterFrameNet(backbone=args.backbone, pyramid_pooling=args.pyramid_pooling,
                              embedding=args.embedding_size, batch_mode='sync')
    elif args.model == 'concat':
        model = ConcatNet(backbone=args.backbone, pyramid_pooling=args.pyramid_pooling,
                          embedding=args.embedding_size, batch_mode='sync')
    elif args.model == 'ad':
        model = AnchorDiffNet(backbone=args.backbone, pyramid_pooling=args.pyramid_pooling,
                              embedding=args.embedding_size, batch_mode='sync')
    
    model.load_state_dict(torch.load(osp.join(args.parent_dir_weights, args.model+'.pth')))
    model.eval()
    model.float()
    model.cuda()
    
    with torch.no_grad():
        video_mean_iou_list = []
        model.eval()
        videos = [i_id.strip() for i_id in open(osp.join(args.data_dir, 'ImageSets', '2016', 'val.txt'))]
        if args.video and args.video in videos:
            videos = [args.video]

        for vid, video in enumerate(videos, start=1):
            curr_video_iou_list = []
            img_files = sorted(glob.glob(osp.join(args.data_dir, 'JPEGImages', '480p', video, '*.jpg')))
            ann_files = sorted(glob.glob(osp.join(args.data_dir, 'Annotations', '480p', video, '*.png')))

            if args.ms_mirror:
                resize_shape = [(857*0.75, 481*0.75), (857, 481), (857*1.5, 481*1.5)]
                resize_shape = [(int((s[0]-1)//8*8+1), int((s[1]-1)//8*8+1)) for s in resize_shape]
                mirror = True
            else:
                resize_shape = [(857, 481)]
                mirror = False

            reference_img = []
            for s in resize_shape:
                reference_img.append((np.asarray(cv2.resize(cv2.imread(img_files[0], cv2.IMREAD_COLOR), s),
                    np.float32) - IMG_MEAN).transpose((2, 0, 1)))
            if mirror:
                for r in range(len(reference_img)):
                    reference_img.append(reference_img[r][:, :, ::-1].copy())
            reference_img = [torch.from_numpy(np.expand_dims(r, axis=0)).cuda() for r in reference_img]
            reference_mask = np.array(Image.open(ann_files[0])) > 0
            reference_mask = torch.from_numpy(np.expand_dims(np.expand_dims(reference_mask.astype(np.float32),
                                                                            axis=0), axis=0)).cuda()
            H, W = reference_mask.size(2), reference_mask.size(3)

            if args.visualize:
                colors = np.random.randint(128, 255, size=(1, 3), dtype="uint8")
                colors = np.vstack([[0, 0, 0], colors]).astype("uint8")

            last_mask_num = 0
            last_mask = None
            last_mask_final = None
            kernel1 = np.ones((15, 15), np.uint8)
            kernel2 = np.ones((101, 101), np.uint8)
            kernel3 = np.ones((31, 31), np.uint8)
            predictions_all = []
            gt_all = []

            for f, (img_file, ann_file) in enumerate(zip(img_files, ann_files)):
                current_img = []
                for s in resize_shape:
                    current_img.append((np.asarray(cv2.resize(
                        cv2.imread(img_file, cv2.IMREAD_COLOR), s),
                        np.float32) - IMG_MEAN).transpose((2, 0, 1)))

                if mirror:
                    for c in range(len(current_img)):
                        current_img.append(current_img[c][:, :, ::-1].copy())

                current_img = [torch.from_numpy(np.expand_dims(c, axis=0)).cuda() for c in current_img]

                current_mask = np.array(Image.open(ann_file)) > 0
                current_mask = torch.from_numpy(np.expand_dims(np.expand_dims(current_mask.astype(np.float32), axis=0), axis=0)).cuda()

                if args.model in ['base']:
                    predictions = [model(cur) for ref, cur in zip(reference_img, current_img)]
                    predictions = [F.interpolate(input=p[0], size=(H, W), mode='bilinear', align_corners=True) for p in predictions]
                elif args.model in ['intra']:
                    predictions = [model(cur) for ref, cur in zip(reference_img, current_img)]
                    predictions = [F.interpolate(input=p, size=(H, W), mode='bilinear', align_corners=True) for p in predictions]
                elif args.model in ['inter', 'concat', 'ad']:
                    predictions = [model(ref, cur) for ref, cur in zip(reference_img, current_img)]
                    predictions = [F.interpolate(input=p, size=(H, W), mode='bilinear', align_corners=True) for p in predictions]

                if mirror:
                    for r in range(len(predictions)//2, len(predictions)):
                        predictions[r] = torch.flip(predictions[r], [3])
                predictions = torch.mean(torch.stack(predictions, dim=0), 0)

                predictions_all.append(predictions.sigmoid().data.cpu().numpy()[0, 0].copy())
                gt_all.append(current_mask.data.cpu().numpy()[0, 0].astype(np.uint8).copy())

                if args.inst_prune:
                    result_dir = os.path.join('inst_prune', video)
                    if os.path.exists(os.path.join(result_dir, img_file.split('/')[-1].split('.')[0] + '.png')):
                        detection_mask = np.array(
                            Image.open(os.path.join(result_dir, img_file.split('/')[-1].split('.')[0] + '.png'))) > 0
                        detection_mask = torch.from_numpy(
                            np.expand_dims(np.expand_dims(detection_mask.astype(np.float32), axis=0), axis=0)).cuda()
                        predictions = predictions * detection_mask

                    process_now = (predictions > args.threshold).data.cpu().numpy().astype(np.uint8)[0, 0]
                    if 100000 > process_now.sum() > 40000:
                        last_mask_numpy = (predictions > args.threshold).data.cpu().numpy().astype(np.uint8)[0, 0]
                        last_mask_numpy = cv2.morphologyEx(last_mask_numpy, cv2.MORPH_OPEN, kernel1)
                        dilation = cv2.dilate(last_mask_numpy, kernel3, iterations=1)
                        contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                        cnt_area = [cv2.contourArea(cnt) for cnt in contours]
                        if len(contours) > 1:
                            contour = contours[np.argmax(cnt_area)]
                            polygon = contour.reshape(-1, 2)
                            x, y, w, h = cv2.boundingRect(polygon)
                            x0, y0 = x, y
                            x1 = x + w
                            y1 = y + h
                            mask_rect = torch.from_numpy(np.zeros_like(dilation).astype(np.float32)).cuda()
                            mask_rect[y0:y1, x0:x1] = 1
                            mask_rect = mask_rect.unsqueeze(0).unsqueeze(0)
                            if np.max(cnt_area) > 30000:
                                if last_mask_final is None or get_iou(last_mask_final, mask_rect, thresh=args.threshold) > 0.3:
                                    predictions = predictions * mask_rect
                    last_mask_final = predictions.clone()

                if 100000 > last_mask_num > 5000:
                    last_mask_numpy = (last_mask > args.threshold).data.cpu().numpy().astype(np.uint8)[0, 0]
                    last_mask_numpy = cv2.morphologyEx(last_mask_numpy, cv2.MORPH_OPEN, kernel1)
                    dilation = cv2.dilate(last_mask_numpy, kernel2, iterations=1)
                    dilation = torch.from_numpy(dilation.astype(np.float32)).cuda()

                    last_mask = predictions.clone()
                    last_mask_num = (predictions > args.threshold).sum()

                    predictions = predictions*dilation
                else:
                    last_mask = predictions.clone()
                    last_mask_num = (predictions > args.threshold).sum()

                iou_temp = get_iou(predictions, current_mask, thresh=args.threshold)
                if 0 < f < (len(ann_files)-1):
                    curr_video_iou_list.append(iou_temp)

                if args.visualize:
                    mask = colors[predictions.squeeze() > args.threshold]
                    output = ((0.4 * cv2.imread(img_file)) + (0.6 * mask)).astype("uint8")
                    cv2.putText(output, "%.3f" % (iou_temp.item()),
                                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                    cv2.imshow(video, output)
                    cv2.waitKey(1)

                    suffix = args.ms_mirror*'ms_mirror'+(not args.ms_mirror)*'single'+args.inst_prune*'_prune'
                    visual_path = osp.join('visualization', args.model + '_' + suffix, img_file.split('/')[-2])
                    if not osp.exists(visual_path):
                        os.makedirs(visual_path)
                    cv2.imwrite(osp.join(visual_path, ann_file.split('/')[-1]), output)

                if args.save_mask:
                    suffix = args.ms_mirror*'ms_mirror'+(not args.ms_mirror)*'single'+args.inst_prune*'_prune'
                    if not osp.exists(osp.join(args.save_mask_dir, args.model, suffix, video)):
                        os.makedirs(osp.join(args.save_mask_dir, args.model, suffix, video))
                    cv2.imwrite(osp.join(args.save_mask_dir, args.model, suffix, video, ann_file.split('/')[-1]),
                                (predictions.squeeze() > args.threshold).cpu().numpy())

            cv2.destroyAllWindows()
            video_mean_iou_list.append(sum(curr_video_iou_list)/len(curr_video_iou_list))
            print('{} {} {}'.format(vid, video, video_mean_iou_list[-1]))

            if args.eval_sal:
                if not osp.exists(args.save_heatmap_dir):
                    os.makedirs(args.save_heatmap_dir)
                with open(args.save_heatmap_dir + video + '.pkl', 'wb') as f:
                    pickle.dump({'pred': np.array(predictions_all), 'gt': np.array(gt_all)}, f, pickle.HIGHEST_PROTOCOL)

        mean_iou = sum(video_mean_iou_list)/len(video_mean_iou_list)
        print('mean_iou {}'.format(mean_iou))
    end = timeit.default_timer()
    print(end-start, 'seconds')
    # ==========================
    if args.eval_sal:
        pkl_files = glob.glob(args.save_heatmap_dir + '*.pkl')
        heatmap_gt = []
        heatmap_pred = []
        for i, pkl_file in enumerate(pkl_files):
            with open(pkl_file, 'rb') as f:
                info = pickle.load(f)
                heatmap_gt.append(np.array(info['gt'][1:-1]).flatten())
                heatmap_pred.append(np.array(info['pred'][1:-1]).flatten())
        heatmap_gt = np.hstack(heatmap_gt).flatten()
        heatmap_pred = np.hstack(heatmap_pred).flatten()
        precision, recall, _ = precision_recall_curve(heatmap_gt, heatmap_pred)
        Fmax = 2 * (precision * recall) / (precision + recall)
        print('MAE', np.mean(abs(heatmap_pred - heatmap_gt)))
        print('F_max', Fmax.max())

        n_sample = len(precision)//1000
        import scipy.io
        scipy.io.savemat('davis.mat', {'recall': recall[0::n_sample], 'precision': precision[0::n_sample]})


if __name__ == '__main__':
    main()
