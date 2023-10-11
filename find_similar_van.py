import numpy as np
import torch
import cv2
import os

from glob import glob
from PIL import Image
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_coords
from transformers import ViTImageProcessor, ViTModel

Image.MAX_IMAGE_PIXELS = 1000000000000000


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
        x1, y1, x2, y2, conf, cls_id = det
        # if cls_id > 0 and int(cls_id) not in [1, 2, 3, 5, 7]:
        #     continue
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        img_crop = img[y1: y2, x1: x2]
        crops.append(img_crop)
    return crops


def embedding_extract(imgs):
    """
    :param: imgs: list, each is a numpy img with shape, h * w * 3
    
    returns:
        shape: len(imgs) * 768
    """
    if len(imgs) == 0:
        return torch.zeros(0)
    
    inputs = vit_processor(images=imgs, return_tensors="pt")

    outputs = vit_model(**inputs)
    return outputs["pooler_output"]


def IoU(box1, box2) -> float:
    """
    IOU, Intersection over Union

    :param box1: coordinates [x1, y1, x2, y2]
    :param box2: coordinates [x1, y1, x2, y2]
    :return: float, iou
    """
    width = max(min(box1[2], box2[2]) - max(box1[0], box2[0]), 0)
    height = max(min(box1[3], box2[3]) - max(box1[1], box2[1]), 0)
    s_inter = width * height
    s_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    s_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    s_union = s_box1 + s_box2 - s_inter
    return s_inter / s_union


def remove_similar_dets(amazon_dets, coco_dets):
    """
    if det in coco_dets is similar(IoU >= similar_thresh) to one of amazon_dets, then remove the det in coco_dets
    """
    similar_thresh = 0.85
    new_coco_dets = []
    for amazon_det in amazon_dets:
        for coco_det in coco_dets:
            iou = IoU(amazon_det[: 4], coco_det[: 4])
            if iou < similar_thresh:
                new_coco_dets.append(coco_det)
    new_coco_dets = np.array(new_coco_dets)
    return new_coco_dets


def det_post_process(dets, img, save_path):
    """
    after detection: 1. crop by dets; 2. extract embedding
    :param dets: det results
    :param img: original image
    :param save_path: embedding save path
    """
    crop_imgs = crop_img(img, dets)
    embeddings = embedding_extract(crop_imgs)
    
    if save_path and len(save_path) > 0:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(embeddings, save_path)
    return embeddings


def get_diff_car_embeddings(img_root, save_root):
    coco_model_embeddings = []
    amazon_model_embeddings = []
    
    img_paths = glob("%s/*" % img_root)
    print("%s images in %s" % (len(img_paths), img_root))
    for img_path in img_paths:
        img = cv2.imread(img_path)

        """
        category: cls_id(start from 0)
        bicycle: 1
        car: 2
        motorbike: 3
        bus: 5
        truck: 7
        """
        coco_model_dets = yolov5_detect(coco_model, img, size=640)
        amazon_model_dets = yolov5_detect(amazon_model, img, size=640)

        coco_model_dets = remove_similar_dets(amazon_model_dets, coco_model_dets)

        # print("amazon_model: ", amazon_model_dets)
        # print("coco_model: ", coco_model_dets)

        amazon_save_path = "%s/amazon/%s.emb" % (save_root, os.path.basename(img_path))
        coco_save_path = "%s/coco/%s.emb" % (save_root, os.path.basename(img_path))

        amazon_embedding = det_post_process(amazon_model_dets, img, amazon_save_path)
        coco_embedding = det_post_process(coco_model_dets, img, coco_save_path)

        coco_model_embeddings.append(coco_embedding)
        amazon_model_embeddings.append(amazon_embedding)
    # shape: 768
    coco_model_embeddings_ = torch.cat(coco_model_embeddings).mean(dim=0)
    amazon_model_embeddings_ = torch.cat(amazon_model_embeddings).mean(dim=0)
    embeddings = {
        "other": coco_model_embeddings_,
        "amazon": amazon_model_embeddings_
    }
    torch.save(embeddings, "%s/dataset_embeddings.emb" % save_root)
    return embeddings


def infer(infer_img_path, embeddings):
    """
    :param: infer_img_path: the image's absolute path
    :param: embeddings: dict, {"other": other cars' embedding, "amazon": amazon car's embedding}
    :returns
        predicts category
    """
    CLASSES = sorted(list(embeddings.keys()))
    
    img = cv2.imread(infer_img_path)
    dets = yolov5_detect(coco_model, img, size=640)
    car_embeddings = det_post_process(dets, img, "")
    if len(car_embeddings) > 0:
        res = []
        for idx, car_embedding in enumerate(car_embeddings):
            min_dist = 1e9
            for cls in embeddings.keys():
                cls_embedding = embeddings[cls]
                dist = torch.abs(car_embedding - cls_embedding).mean()
                if dist <= min_dist:
                    min_dist = dist
                    dets[idx][-1] = CLASSES.index(cls)
        return dets, CLASSES
    else:
        return None, CLASSES
    

def plot_img(img_path, dets, CLASSES, save_path):
    img = cv2.imread(img_path)
    colors = [
        (0, 0, 255),
        (0, 255, 0)
    ]
    if dets is not None:
        for det in dets:
            x1, y1, x2, y2, conf, cls_id = det
            x1, y1, x2, y2, cls_id = int(x1), int(y1), int(x2), int(y2), int(cls_id)
            cv2.rectangle(img, (x1, y1), (x2, y2), colors[cls_id], 2, 2)
            cv2.putText(img, CLASSES[cls_id], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[cls_id], 2)
            print("car coord: %s; cls: %s" % (det, CLASSES[int(det[-1])]))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)
        
        
def main(dataset_root, embeddings_save_root, infer_img_path):
    dataset_embedding_path = "%s/dataset_embeddings.emb" % embeddings_save_root
    if os.path.exists(dataset_embedding_path):
        print("load existing dataset embeddings: ", dataset_embedding_path)
        embeddings = torch.load(dataset_embedding_path)
    else:
        print("calculating embeddings, please waiting......")
        embeddings = get_diff_car_embeddings(dataset_root, embeddings_save_root)
    
    dets, CLASSES = infer(infer_img_path, embeddings)
    
    img_save_path = "results/v0/%s" % os.path.basename(infer_img_path)
    plot_img(infer_img_path, dets, CLASSES, img_save_path)
        

if __name__ == "__main__":
    """
    pipeline:
    1. cache dataset embeddings
        a. traverse all images in the dataset, for each image, do:
        b. detect amazon van by amazon_van model
        c. detect cars by coco model
        d. if the cars are detected by both amazon_van model and coco model, remove the cars in coco model's detections
        e. crop the cars of amazon_van model and coco model respectively
        f. extract the embeddings of amazon_van model and coco model respectively
        g. add all the embeddings of amazon_van/coco model to get avg_embedding of amazon_van/coco model
        h. save all the embeddings
    2. inference
        a. read an image
        b. detect cars by coco model
        c. crop detected cars and extract these cars' embedding respectively
        d. traverse these detected cars' embedding, for each image's embedding, do:
        e. find the closest embedding's catrgoy
        f. output & save result
    """
    
    device = "cuda"
    amazon_ckpt_path = "ckpts/amazon_van_detect.torchscript"
    amazon_model = load_model(amazon_ckpt_path, device)
    coco_ckpt_path = "ckpts/coco_yolov5x.torchscript"
    coco_model = load_model(coco_ckpt_path, device)
    vit_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
    
    
    dataset_root = "/home/ubuntu/codes/SimilarVan/data/Delivery_Van/train"
    embeddings_save_root = "embeddings/v0"
    # infer_img_path = "/home/ubuntu/codes/SimilarVan/data/Delivery_Van/train/20230710170713.MP4_000011.jpg"
    # infer_img_path = "/home/ubuntu/codes/SimilarVan/data/Delivery_Van/train/20230710170432.MP4_000139.jpg"
    # infer_img_path = "/home/ubuntu/codes/SimilarVan/data/Delivery_Van/val/20230710165403.MP4_000092.jpg"
    infer_img_path = "/home/ubuntu/codes/SimilarVan/data/Delivery_Van/val/20230710170432.MP4_000099.jpg"
    infer_img_path = "/home/ubuntu/codes/SimilarVan/data/Delivery_Van/val/20230710170713.MP4_000027.jpg"
    infer_img_path = "/home/ubuntu/codes/SimilarVan/data/Delivery_Van/val/20230710170713.MP4_000095.jpg"
    infer_img_path = "/home/ubuntu/codes/SimilarVan/data/Delivery_Van/val/20230710164737.MP4_000134.jpg"
    
    main(dataset_root, embeddings_save_root, infer_img_path)
    
