import os
import re
import argparse
from pycocotools.coco import COCO
import os.path as osp
import tensorflow as tf
from dataset.dataloader import preprocess
from dataset.coco import cn as cfg
import numpy as np
from validate import get_preds
import cv2
import glob


KP_PAIRS = [[5, 6], [6, 12], [12, 11], [11, 5],
            [5, 7], [7, 9], [11, 13], [13, 15],
            [6, 8], [8, 10], [12, 14], [14, 16]]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', default='evopose2d_M_f32.yaml')
    parser.add_argument('-p', '--coco-path', required=True,
                        help='Path to folder containing COCO images and annotation directories.')
    parser.add_argument('-i', '--img-id', type=int, default=785)
    parser.add_argument('--alpha', type=float, default=0.8)
    args = parser.parse_args()

    # load the config .yaml file
    cfg.merge_from_file('configs/' + args.cfg)

    # load the trained model
    model = tf.keras.models.load_model('models/{}.h5'.format(args.cfg.split('.yaml')[0]), compile=False)
    cfg.DATASET.OUTPUT_SHAPE = model.output_shape[1:]

    # load the dataset annotations
    coco = COCO(osp.join(args.coco_path, 'annotations', 'person_keypoints_val2017.json'))
    img_data = coco.loadImgs([args.img_id])[0]

    annotation = coco.loadAnns(coco.getAnnIds([args.img_id]))[0]
    bbox = annotation['bbox']
    kp = np.array(annotation['keypoints']).reshape(-1, 3)  # not used

    # cut video to images
    if not os.listdir('frames'):
        vidcap = cv2.VideoCapture('videos/vid1.mp4')
        success, image = vidcap.read()
        count = 0
        while success:
            cv2.imwrite("frames/frame%d.jpg" % count, image)  # save frame as JPEG file
            success, image = vidcap.read()
            print('Read a new frame: ', success, count)
            count += 1


    # get all images
    files = os.listdir('frames')
    files.sort()

    if not os.listdir('done_frames'):
        for file in files:
            img_bytes = open(osp.join('frames', file), 'rb').read()
            img = tf.image.decode_jpeg(img_bytes, channels=3)

            # preprocess
            _, norm_img, _, M, _ = preprocess(0, img, bbox, kp, 0., cfg.DATASET, split='val', predict_kp=True)
            M = np.expand_dims(np.array(M), axis=0)

            # generate heatmap predictions
            hms = model.predict(tf.expand_dims(norm_img, 0))

            # get keypoint predictions from heatmaps
            preds = get_preds(hms, M, cfg.DATASET.INPUT_SHAPE, cfg.DATASET.OUTPUT_SHAPE)[0]

            # plot results
            img = img.numpy()[:, :, [2, 1, 0]]
            overlay = img.copy()

            for i, (x, y, v) in enumerate(preds):
                overlay = cv2.circle(overlay, (int(np.round(x)), int(np.round(y))), 3, [255, 255, 255], 4)

            for p in KP_PAIRS:
                overlay = cv2.line(overlay,
                                   tuple(np.int32(np.round(preds[p[0], :2]))),
                                   tuple(np.int32(np.round(preds[p[1], :2]))), [255, 255, 255], 4)

            img = cv2.addWeighted(overlay, args.alpha, img, 1 - args.alpha, 0)
            print('file', file, 'completed')
            cv2.imwrite(osp.join('done_frames', file), img)


    # putting images back to video
    img_array = []
    files = glob.glob('C:/Users/user/Desktop/pythonProjects/EvoPose2D/evopose2d/done_frames/*.jpg')
    files.sort(key=lambda f: int(re.sub('\D', '', f)))
    for filename in files:
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter('vid1.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()