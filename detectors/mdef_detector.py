import numpy as np
import torch
import cv2
import os
import shutil

from glob import glob
from tqdm import tqdm
from PIL import Image

from models.model import Model


class MdefDetector(object):
    def __init__(self, model_name="mdef_detr_minus_language", ckpt_path="ckpts/MDef_DETR_minus_language_r101_epoch10.pth"):
        self.model = Model(model_name, ckpt_path).get_model()
        
    def detect(self, image_path, conf_thresh=0.25, caption="all objects", multi_crop=False, verbose=False):
        assert isinstance(image_path, str)
        # Note: Caption is only rquired for MViTs
        if multi_crop:
            dets = self.model.infer_image_multi_crop(image_path, caption=caption)
        else:
            dets = self.model.infer_image(image_path, caption=caption)

        bboxes = np.array(dets[0], dtype=np.int32)
        confs = np.array(dets[1], dtype=np.float32)
        idx = np.where(confs >= conf_thresh)
        new_dets = np.zeros((len(idx[0]), 6), dtype=np.float32)
        new_dets[:, :4] = bboxes[idx]
        new_dets[:, -2] = confs[idx]
        new_dets[:, -1] = -1
        h, w, _ = cv2.imread(image_path).shape
        new_dets[np.where(new_dets[:, 0] < 0), 0] = 0
        new_dets[np.where(new_dets[:, 1] < 0), 1] = 0
        new_dets[np.where(new_dets[:, 2] > w), 2] = w
        new_dets[np.where(new_dets[:, 3] > h), 3] = h
        if verbose:
            print("mdef_dets: ", new_dets)
        return new_dets

        
if __name__ == "__main__":
    model_name="mdef_detr_minus_language"
    ckpt_path="ckpts/MDef_DETR_minus_language_r101_epoch10.pth"
    detector = MdefDetector(model_name, ckpt_path)
    img_path = "/home/ubuntu/codes/SimilarVan/data/Girl/0005.jpg"
    detector.detect(img_path, verbose=True)