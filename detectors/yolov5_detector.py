import numpy as np
import torch
import cv2
import os
import shutil

from glob import glob
from tqdm import tqdm
from PIL import Image
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_coords


class Yolov5Detector(object):
    def __init__(self, ckpt_path, device="cuda"):
        model = torch.jit.load(ckpt_path).to(device)
        model.eval()
        self.model = model
        self.device = device
        
    def detect(self, image_path, conf_thresh=0.25, size=640, verbose=False):
        assert isinstance(image_path, str)
        img = cv2.imread(image_path)
        im = letterbox(img, [size, size], stride=32, auto=False)[0]
        # Convert
        im = im[:, :, ::-1].transpose((2, 0, 1))  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to('cuda')
        im = im.float()  # uint8 to fp16/32
        im /= 255
        im = im.unsqueeze(0)

        # mdoel infer
        pred = self.model(im)[0]  # 1 25200 7
        pred = non_max_suppression(pred, conf_thresh, 0.45, None, False, max_det=1000)
        # Process predictions
        dets = pred[0]
        if len(dets):
            # Rescale boxes from img_size to img size
            dets[:, :4] = scale_coords(im.shape[2:], dets[:, :4], img.shape).round()
        new_det = []
        for det in dets:
            if int(det[-1]) in [0, 1, 2, 3, 5, 7]:
                new_det.append(det)
        if verbose:
            print("yolov5_dets: ", new_det)
        if len(new_det) != 0:
            new_det = torch.stack(new_det)
            return new_det
        else:
            return []

        
if __name__ == "__main__":
    ckpt_path = "/home/ubuntu/codes/SimilarVan/ckpts/coco_yolov5x.torchscript"
    detector = Yolov5Detector(ckpt_path)
    img_path = "/home/ubuntu/codes/SimilarVan/data/Girl/0005.jpg"
    detector.detect(img_path, verbose=True)