import os
import glob
import cv2
import numpy as np
import pickle
import shutil


def _IoU(rect1, rect2):
    def inter(rect1, rect2):
        x1 = max(rect1[0], rect2[0])
        y1 = max(rect1[1], rect2[1])
        x2 = min(rect1[2], rect2[2])
        y2 = min(rect1[3], rect2[3])
        return max(x2 - x1 + 1, 0) * max(y2 - y1 + 1, 0) * 1.

    def area(rect):
        x1, y1, x2, y2 = rect
        return (x2 - x1 + 1) * (y2 - y1 + 1)

    ii = inter(rect1, rect2)
    iou = ii / (area(rect1) + area(rect2) - ii)
    return iou


def vis_mask(img, mask, col, alpha=0.4, show_border=True, border_thick=2):
    """Visualizes a single binary mask."""

    img = img.astype(np.float32)
    idx = np.nonzero(mask)

    img[idx[0], idx[1], :] *= 1.0 - alpha
    img[idx[0], idx[1], :] += alpha * col
    _WHITE = (255, 255, 255)
    if show_border:
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contours, -1, _WHITE, border_thick, cv2.LINE_AA)

    return img.astype(np.uint8)


def colormap(rgb=False):
    color_list = np.array(
        [
            0.000, 0.447, 0.741,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
            0.635, 0.078, 0.184,
            0.300, 0.300, 0.300,
            0.600, 0.600, 0.600,
            1.000, 0.000, 0.000,
            1.000, 0.500, 0.000,
            0.749, 0.749, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 1.000,
            0.667, 0.000, 1.000,
            0.333, 0.333, 0.000,
            0.333, 0.667, 0.000,
            0.333, 1.000, 0.000,
            0.667, 0.333, 0.000,
            0.667, 0.667, 0.000,
            0.667, 1.000, 0.000,
            1.000, 0.333, 0.000,
            1.000, 0.667, 0.000,
            1.000, 1.000, 0.000,
            0.000, 0.333, 0.500,
            0.000, 0.667, 0.500,
            0.000, 1.000, 0.500,
            0.333, 0.000, 0.500,
            0.333, 0.333, 0.500,
            0.333, 0.667, 0.500,
            0.333, 1.000, 0.500,
            0.667, 0.000, 0.500,
            0.667, 0.333, 0.500,
            0.667, 0.667, 0.500,
            0.667, 1.000, 0.500,
            1.000, 0.000, 0.500,
            1.000, 0.333, 0.500,
            1.000, 0.667, 0.500,
            1.000, 1.000, 0.500,
            0.000, 0.333, 1.000,
            0.000, 0.667, 1.000,
            0.000, 1.000, 1.000,
            0.333, 0.000, 1.000,
            0.333, 0.333, 1.000,
            0.333, 0.667, 1.000,
            0.333, 1.000, 1.000,
            0.667, 0.000, 1.000,
            0.667, 0.333, 1.000,
            0.667, 0.667, 1.000,
            0.667, 1.000, 1.000,
            1.000, 0.000, 1.000,
            1.000, 0.333, 1.000,
            1.000, 0.667, 1.000,
            0.167, 0.000, 0.000,
            0.333, 0.000, 0.000,
            0.500, 0.000, 0.000,
            0.667, 0.000, 0.000,
            0.833, 0.000, 0.000,
            1.000, 0.000, 0.000,
            0.000, 0.167, 0.000,
            0.000, 0.333, 0.000,
            0.000, 0.500, 0.000,
            0.000, 0.667, 0.000,
            0.000, 0.833, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 0.167,
            0.000, 0.000, 0.333,
            0.000, 0.000, 0.500,
            0.000, 0.000, 0.667,
            0.000, 0.000, 0.833,
            0.000, 0.000, 1.000,
            0.000, 0.000, 0.000,
            0.143, 0.143, 0.143,
            0.286, 0.286, 0.286,
            0.429, 0.429, 0.429,
            0.571, 0.571, 0.571,
            0.714, 0.714, 0.714,
            0.857, 0.857, 0.857,
            1.000, 1.000, 1.000
        ]
    ).astype(np.float32)
    color_list = color_list.reshape((-1, 3)) * 255
    if not rgb:
        color_list = color_list[:, ::-1]
    return color_list


videos = [i_id.strip() for i_id in open(os.path.join('./data/DAVIS/', 'ImageSets', '2016', 'val.txt'))]
train_videos = [i_id.strip() for i_id in open(os.path.join('./data/DAVIS/', 'ImageSets', '2016', 'train.txt'))]
frame_count = []
for video in train_videos:
    img_files = sorted(
        glob.glob(os.path.join('./data/DAVIS/', 'JPEGImages', '480p', video, '*.jpg')))
    frame_count.append(len(img_files))
mean_frame_count = np.mean(frame_count)

out_dir = './inst_prune'
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)

for vid, video in enumerate(videos):
    def load_obj(name):
        with open('detection/' + name + '.pkl', 'rb') as f:
            return pickle.load(f)
    if not os.path.exists('detection/' + video + '.pkl'):
        print('no detection on:', video)
        continue
    detect_res = load_obj(video)
    frame_len = len(detect_res)
    bboxes_all = []
    for frame_info in detect_res:
        for instance in detect_res[frame_info]:
            bboxes_all.append(instance['bbox'])

    mean_bboxes = len(bboxes_all)/frame_len
    first_remove = -1
    if mean_bboxes > 3:
        color_list = colormap(rgb=True)
        size_bboxes = [(bbox[3] - bbox[1]) * (bbox[2] - bbox[0]) for bbox in bboxes_all]
        size_bboxes = sorted(size_bboxes)
        size_bboxes_target = size_bboxes[-frame_len]

        img_files = sorted(
            glob.glob(os.path.join('./data/DAVIS/', 'JPEGImages', '480p', video, '*.jpg')))

        for f, img_file in enumerate(img_files):
            im = cv2.imread(img_file, cv2.IMREAD_COLOR)
            frame_bboxes = []
            frame_masks = []
            for id, instance in enumerate(detect_res[f]):
                frame_bboxes.append(instance['bbox'])
                frame_masks.append(instance['mask'])
                bbox = instance['bbox']
                mask = instance['mask']
                cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)
                im = vis_mask(im, mask, color_list[id % len(color_list), :3], alpha=0.4)

            size_bboxes = [(bbox[3]-bbox[1])*(bbox[2]-bbox[0])for bbox in frame_bboxes]
            score_bboxes = [bbox[4] for bbox in frame_bboxes]
            size_bboxes_ind = np.argsort(size_bboxes)
            size_bboxes = sorted(size_bboxes)

            target_box = frame_bboxes[size_bboxes_ind[-1]]

            im_mask = np.ones((im.shape[0], im.shape[1]))

            for bbox, mask in zip(frame_bboxes, frame_masks):
                static_object_count = 0
                if (bbox[3]-bbox[1])*(bbox[2]-bbox[0]) > 47000 or (bbox[3]-bbox[1])*(bbox[2]-bbox[0]) == size_bboxes[-1]:
                    continue
                for i in range(len(bboxes_all)):
                    if _IoU(bbox[:4], bboxes_all[i][:4]) > 0.6:
                        static_object_count += 1

                if static_object_count > 0.4 * mean_frame_count:
                    im_mask = im_mask * (1 - mask)
                    cv2.putText(im, 'static', (bbox[0], bbox[1]), 2, 2, (0, 255, 0))

            if len(size_bboxes) > 1:
                if size_bboxes[-1] > 10000 and size_bboxes[-1] > 2*size_bboxes[-2] and \
                        size_bboxes[-1] > size_bboxes_target and \
                        (target_box[-1] == 0 or target_box[-1] == 2) and \
                        (target_box[2]-target_box[0])/(target_box[3]-target_box[1]) < 3:
                    suppress_small = True
                    if first_remove == -1:
                        first_remove = f
                        if first_remove > 20:
                            break
                else:
                    suppress_small = False

            if suppress_small:
                for bbox, mask in zip(frame_bboxes, frame_masks):
                    cx = (bbox[3]+bbox[1])/2
                    cy = (bbox[2]+bbox[0])/2
                    cx0 = (target_box[3] + target_box[1]) / 2
                    cy0 = (target_box[2] + target_box[0]) / 2
                    d_dist = abs(cx-cx0)+ abs(cy-cy0)
                    if ((bbox[3]-bbox[1])*(bbox[2]-bbox[0]) < size_bboxes[-1]//3 or
                        ((cy < 300 or cy > 600) and d_dist > 200)) and \
                            _IoU(target_box[:4], bbox[:4]) <= 0.1 and \
                            bbox[-1] == frame_bboxes[size_bboxes_ind[-1]][-1]:
                        # print(cx)

                        # cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 5)
                        # im_mask[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = 0
                        im_mask = im_mask*(1 - mask)

            result_dir = os.path.join(out_dir, video)
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)

            cv2.imwrite(os.path.join(result_dir, img_file.split('/')[-1].split('.')[0] + '.png'), im_mask*255)
            im_mask = (im_mask*255).astype(np.uint8)
            im = cv2.vconcat((cv2.cvtColor(im_mask.copy(), cv2.COLOR_GRAY2BGR), im))
            im = cv2.resize(im, dsize=None, fx=0.5, fy=0.5)
            # cv2.imshow('mask', im_mask*255)
            # cv2.imshow(video, im)
            # cv2.waitKey(1)
        cv2.destroyAllWindows()
