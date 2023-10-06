import numpy as np
import torch
import torchvision
import cv2
from PIL import Image

Image.MAX_IMAGE_PIXELS = 1000000000000000
from glob import glob as g
import os
import warnings
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_coords

warnings.filterwarnings(action='ignore')



def load_model(ckpt_path, device="cuda"):
    model = torch.jit.load(ckpt_path).to(device)
    model.eval()
    return model


def yolov5_detect(model, img, size=640):
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
    det = pred[0]
    if len(det):
        # Rescale boxes from img_size to img size
        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], img.shape).round()
    return det.detach().cpu().numpy()



def crop_img(img, dets):
    crops = []
    for i, det in enumerate(dets):
        x1, y1, x2, y2, conf, cls_id = det  # 两个点
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        img_crop = img[y1: y2, x1: x2]
        crops.append(img_crop)
    return crops


def main():
    device = "cuda"
    amazon_ckpt_path = "ckpts/amazon_van_detect.torchscript"
    amazon_model = load_model(amazon_ckpt_path, device)
    coco_ckpt_path = "ckpts/coco_yolov5m.torchscript"
    coco_model = load_model(coco_ckpt_path, device)
    
    img_path = "/home/ubuntu/codes/Aerial-I8N/data/Delivery_Van/images/train_val/20230710170713.MP4_000011.jpg"
    img = cv2.imread(img_path)
    
    amazon_model_dets = yolov5_detect(amazon_model, img, size=640)
    
    """
    category: cls_id(start from 0)
    bicycle: 1
    car: 2
    motorbike: 3
    bus: 5
    truck: 7
    """
    coco_model_dets = yolov5_detect(coco_model, img, size=640)
    
    print("coco_model: ", coco_model_dets)
    print("amazon_model: ", amazon_model_dets)
    

    
def embedding_extract():
    from transformers import ViTImageProcessor, ViTModel
    from PIL import Image
    import requests

    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)

    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
    inputs = processor(images=image, return_tensors="pt")

    outputs = model(**inputs)
    print("last_hidden_state: ", outputs["last_hidden_state"].shape)
    print("pooler_output: ", outputs["pooler_output"].shape)
    
    
    


if __name__ == "__main__":
    main()
    embedding_extract()