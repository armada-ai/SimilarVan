import os
import cv2
import argparse
import numpy as np
import sys
sys.path.insert(0, "./")
from tqdm import tqdm
from models.model import Model


def plot_img(image_path, output_path, new_dets):
    result_img = cv2.imread(image_path)
    for bbox in new_dets:
        cv2.rectangle(result_img, (int(bbox[0]), int(bbox[1])), 
                      (int(bbox[2]), int(bbox[3])), (0, 0, 255), 1, 1)
        cv2.putText(result_img, str(round(bbox[-2], 2)), 
                    (int(bbox[0]), int(bbox[1] - 5)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 1)
    print("save_path: ", "%s/%s" % (output_path, os.path.basename(image_path)))
    cv2.imwrite("%s/%s" % (output_path, os.path.basename(image_path)), result_img)
    
    
def run_inference(model, image_path, caption="all objects", multi_crop=False):
    image_path = cv2.imread(image_path)
    # Note: Caption is only rquired for MViTs
    if multi_crop:
        dets = model.infer_image_multi_crop(image_path, caption=caption)
    else:
        dets = model.infer_image(image_path, caption=caption)
    
    thresh = 0.25
    thresh = 0.15
    bboxes = np.array(dets[0], dtype=np.int32)
    confs = np.array(dets[1], dtype=np.float32)
    idx = np.where(confs >= thresh)
    new_dets = np.zeros((len(idx[0]), 6), dtype=np.float32)
    new_dets[:, :4] = bboxes[idx]
    new_dets[:, -2] = confs[idx]
    new_dets[:, -1] = -1
    return new_dets
        
        
def main():
    model_name = "mdef_detr_minus_language"
    image_path = "test_imgs/leverkusen_000008_000019_leftImg8bit.png"
    image_path = "test_imgs/amazon_van_01_00390.jpg"
    checkpoints_path = "ckpts/MDef_DETR_minus_language_r101_epoch10.pth"
    text_query = "all objects"
    multi_crop = False
    model = Model(model_name, checkpoints_path).get_model()
    output_dir = "results/%s" % (model_name)
    os.makedirs(output_dir, exist_ok=True)
    new_dets = run_inference(model, image_path, caption=text_query, multi_crop=multi_crop)
    plot_img(image_path, output_dir, new_dets)
    


if __name__ == "__main__":
    main()
