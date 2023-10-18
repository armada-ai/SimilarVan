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

import sys
sys.path.append("./inference")
from models.model import Model


def load_coco_model(ckpt_path, device="cuda"):
    model = torch.jit.load(ckpt_path).to(device)
    model.eval()
    return model


def yolov5_detect(model, img, size=640):
    if isinstance(img, str):
        img = cv2.imread(img)
    im = letterbox(img, [size, size], stride=32, auto=False)[0]
    # Convert
    im = im[:, :, ::-1].transpose((2, 0, 1))  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)
    im = torch.from_numpy(im).to('cuda')
    im = im.float()  # uint8 to fp16/32
    im /= 255
    im = im.unsqueeze(0)

    # mdoel infer
    pred = model(im)[0]  # 1 25200 7
    pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)
    # Process predictions
    dets = pred[0]
    if len(dets):
        # Rescale boxes from img_size to img size
        dets[:, :4] = scale_coords(im.shape[2:], dets[:, :4], img.shape).round()
    new_det = []
    for det in dets:
        if int(det[-1]) in [0, 1, 2, 3, 5, 7]:
            new_det.append(det)
    if len(new_det) != 0:
        new_det = torch.stack(new_det)
        return new_det.detach().cpu().numpy()
    else:
        return []
    
    
def load_od_model(model_name="mdef_detr_minus_language", ckpt_path="ckpts/MDef_DETR_minus_language_r101_epoch10.pth"):
    model = Model(model_name, ckpt_path).get_model()
    return model


def od_detect(model, image_path, caption="all objects", multi_crop=False, thresh=0.15):
    # Note: Caption is only rquired for MViTs
    if multi_crop:
        dets = model.infer_image_multi_crop(image_path, caption=caption)
    else:
        dets = model.infer_image(image_path, caption=caption)
    
    bboxes = np.array(dets[0], dtype=np.int32)
    confs = np.array(dets[1], dtype=np.float32)
    idx = np.where(confs >= thresh)
    new_dets = np.zeros((len(idx[0]), 6), dtype=np.float32)
    new_dets[:, :4] = bboxes[idx]
    new_dets[:, -2] = confs[idx]
    new_dets[:, -1] = -1
    h, w, _ = cv2.imread(image_path).shape
    new_dets[np.where(new_dets[:, 0] < 0), 0] = 0
    new_dets[np.where(new_dets[:, 1] < 0), 1] = 0
    new_dets[np.where(new_dets[:, 2] > w), 2] = w
    new_dets[np.where(new_dets[:, 3] > h), 3] = h
    # import pdb
    # pdb.set_trace()
    return new_dets


def crop_img(img, dets):
    crops = []
    h, w, _ = img.shape
    for i, det in enumerate(dets):
        x1, y1, x2, y2, conf, cls_id = det
        # if cls_id > 0 and int(cls_id) not in [1, 2, 3, 5, 7]:
        #     continue
        x1, y1, x2, y2 = max(int(x1), 0), max(int(y1), 0), min(int(x2), w), min(int(y2), h)
        img_crop = img[y1: y2, x1: x2]
        crops.append(img_crop)
    return crops



def gen_crop_imgs(img_root, model, save_root, det_func=yolov5_detect):
    img_paths = glob("%s/*" % img_root)
    print("len(img_paths): ", len(img_paths))
    for img_path in img_paths:
        img = cv2.imread(img_path)
        dets = det_func(model, img, size=640)
        crops = crop_img(img, dets)
        for idx, crop in enumerate(crops):
            cv2.imwrite("%s/%s_%04d.jpg" % (save_root, os.path.basename(img_path), idx), crop)
            
                


image_path = "data/0001.jpg"

device = "cuda"
coco_ckpt_path = "ckpts/coco_yolov5x.torchscript"
coco_model = load_coco_model(coco_ckpt_path, device)
coco_dets = yolov5_detect(coco_model, image_path, size=640)
print("coco_dets: ", coco_dets)

img_root = "data/AmazonVan"
amazon_model = load_coco_model("ckpts/amazon_van_detect.torchscript", device)
save_root = "data/AmazonVanCrop"
os.makedirs(save_root, exist_ok=True)
gen_crop_imgs(img_root, amazon_model, save_root)


model_name="mdef_detr_minus_language"
od_ckpt_path="ckpts/MDef_DETR_minus_language_r101_epoch10.pth"
od_model = load_od_model(model_name, od_ckpt_path)
od_dets = od_detect(od_model, image_path, caption="all objects", multi_crop=False, thresh=0.15)
print("od_dets: ", od_dets)