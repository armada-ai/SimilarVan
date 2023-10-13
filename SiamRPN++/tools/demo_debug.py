from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse

import cv2
import torch
import numpy as np
from glob import glob

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from pysot.utils.bbox import get_axis_aligned_bbox, cxy_wh_2_rect

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', type=str, default="../experiments/siamrpn_r50_l234_dwxcorr/config.yaml",
                    help='config file')
parser.add_argument('--snapshot', type=str, default="experiments/siamrpn_r50_l234_dwxcorr/model.pth", help='model name')
parser.add_argument('--video_name', default='../20230710170713.mp4', type=str,
                    help='videos or image files')


# args = parser.parse_args()


def get_frames1(video_path):
    reader = cv2.VideoCapture(video_path)
    count = 0
    while True:
        ret, frame = reader.read()
        if not ret:
            break
        count += 1
        # if count <= 140:
        #     continue
        yield frame


def main(config, snapshot, video_url, init_rect):
    init_rect = np.array(init_rect)
    if len(init_rect) == 8:
        init_rect = get_axis_aligned_bbox(init_rect)
        init_rect = cxy_wh_2_rect(init_rect[: 2], init_rect[2:])
    # load config
    cfg.merge_from_file(config)
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder()

    # load model
    if snapshot is not None:
        params = torch.load(snapshot, map_location=lambda storage, loc: storage.cpu())
        if "state_dict" in params.keys():
            params = params["state_dict"]
        model.load_state_dict(params)
        # model.load_state_dict(torch.load(snapshot,
        #                                  map_location=lambda storage, loc: storage.cpu()))
        model.eval().to(device)

    # build tracker
    tracker = build_tracker(model)

    first_frame = True
    if video_url:
        video_name = video_url.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
    idx = 0
    for frame in get_frames1(video_url):
        idx += 1
        if first_frame:
            # try:
            #     init_rect = cv2.selectROI(video_name, frame, False, False)
            # except:
            #     exit()
            print("init_rect: ", init_rect)
            # init_rect = []
            tracker.init(frame, init_rect)
            first_frame = False
        else:
            outputs = tracker.track(frame)
            if 'polygon' in outputs:
                polygon = np.array(outputs['polygon']).astype(np.int32)
                cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                              True, (0, 255, 0), 3)
                mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                mask = mask.astype(np.uint8)
                mask = np.stack([mask, mask * 255, mask]).transpose(1, 2, 0)
                frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
            else:
                bbox = list(map(int, outputs['bbox']))
                print("idx: {};bbox: {}".format(idx, bbox))
                cv2.rectangle(frame, (bbox[0], bbox[1]),
                              (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                              (0, 255, 0), 3)
            cv2.putText(frame, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow(video_name, frame)
            cv2.waitKey(10)


if __name__ == '__main__':
    # SiamRPN++
    config = "../experiments/siamrpn_r50_l234_dwxcorr/config.yaml"
    snapshot = "../experiments/siamrpn_r50_l234_dwxcorr/model.pth"

    video_url = "../test_vids/20230710170713.mp4"
    init_rect = [0, 306, 240, 174]

    main(config, snapshot, video_url, init_rect)
