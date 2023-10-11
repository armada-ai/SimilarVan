import numpy as np
import torch
import cv2
import os

from glob import glob
from tqdm import tqdm
from PIL import Image
from sklearn.cluster import  KMeans
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
    dets = pred[0]
    if len(dets):
        # Rescale boxes from img_size to img size
        dets[:, :4] = scale_coords(im.shape[2:], dets[:, :4], img.shape).round()
    new_det = []
    for det in dets:
        if int(det[-1]) in [1, 2, 3, 5, 7]:
            new_det.append(det)
    if len(new_det) != 0:
        new_det = torch.stack(new_det)
        return new_det.detach().cpu().numpy()
    else:
        return []



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


def save_embeddings(embeddings, save_path):
    """
    :param save_path: embedding save path
    """
    if save_path and len(save_path) > 0:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(embeddings, save_path)


def get_img_embeddings(img, save_path):
    """
    category: cls_id(start from 0)
    bicycle: 1
    car: 2
    motorbike: 3
    bus: 5
    truck: 7
    """
    dets = yolov5_detect(coco_model, img, size=640)
    # print("coco_model: ", dets)
    if len(dets) == 0:
        return None
    crop_imgs = crop_img(img, dets)
    img_embeddings = embedding_extract(crop_imgs)
    
    save_embeddings(img_embeddings, save_path)
    return img_embeddings


def get_video_embeddings(vid_path, save_root, interval=10):
    reader = cv2.VideoCapture(vid_path)
        
    vid_name = os.path.basename(vid_path).split(".")[0]
    count = 0
    
    vid_embeddings = []
    while True:
        ret, frame = reader.read()
        if not ret:
            break
        if count % interval == 0:
            save_path = "%s/%s/%06d_img_embedding.emb" % (save_root, vid_name, count)
            img_embedding = get_img_embeddings(frame, save_path)
            if img_embedding is not None:
                vid_embeddings.append(img_embedding)
        count += 1
    
    vid_embeddings_ = torch.cat(vid_embeddings)
    print("vid_embeddings_.shape: ", vid_embeddings_.shape)
    save_embeddings(vid_embeddings_, "%s/%s/vid_embedding.emb" % (save_root, vid_name))
    return vid_embeddings_
            
            
def cache_dataset_embeddings(vid_root, save_root, interval=10):
    vid_paths = glob("%s/*.MP4" % (dataset_root))
    
    dataset_embeddings = []
    
    for vid_path in tqdm(vid_paths):
        # shape: n, 768
        vid_embedding = get_video_embeddings(vid_path, save_root, interval)
        if len(vid_embedding) != 0:
            dataset_embeddings.append(vid_embedding)
        
    # shape: n, 768
    dataset_embeddings_ = torch.cat(dataset_embeddings)
    print("dataset_embeddings_.shape: ", dataset_embeddings_.shape)
    
    torch.save(dataset_embeddings_, "%s/dataset_embeddings.emb" % save_root)
    return dataset_embeddings_


def infer(infer_img_path, embeddings, num_cls=5):
    """
    :param: infer_img_path: the image's absolute path
    :param: embeddings: dict, {"other": other cars' embedding, "amazon": amazon car's embedding}
    :returns
        predicts category
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    
    CLASSES = list(range(num_cls))
    if len(embeddings) < num_cls:
        print("database only have %s samples, less than pre-difined number of cls: %s" % (len(embeddings), num_cls))
        return None, CLASSES
    
    # define kmeans model
    model = KMeans(n_clusters=num_cls, n_init=10)
    # kmeans model cluster
    model.fit(embeddings)
    
    img = cv2.imread(infer_img_path)
    dets = yolov5_detect(coco_model, img, size=640)

    crop_imgs = crop_img(img, dets)
    car_embeddings = embedding_extract(crop_imgs)

    if len(car_embeddings) > 0:
        car_embeddings = car_embeddings.detach().cpu().numpy()
        # kmeans model predict
        pred_cls = model.predict(car_embeddings)
        dets[:, -1] = pred_cls
        return dets, CLASSES
    else:
        return None, CLASSES
    

def plot_img(img_path, dets, CLASSES, save_path):
    img = cv2.imread(img_path)
    colors = [
        (0, 0, 255),
        (0, 255, 0),
        (255, 0, 0),
        (0, 255, 255),
        (255, 255, 0),
        (127, 127, 127),
        (255, 255, 255),
        (0, 0, 0),
    ]
    if dets is not None:
        for det in dets:
            x1, y1, x2, y2, conf, cls_id = det
            x1, y1, x2, y2, cls_id = int(x1), int(y1), int(x2), int(y2), int(cls_id)
            cv2.rectangle(img, (x1, y1), (x2, y2), colors[cls_id], 2, 2)
            cv2.putText(img, str(CLASSES[cls_id]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[cls_id], 2)
            print("car coord: %s; cls: %s" % (det, CLASSES[int(det[-1])]))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)

    
def main(dataset_root, embeddings_save_root, infer_img_path, interval=10, num_cls=5):
    dataset_embedding_path = "%s/dataset_embeddings.emb" % embeddings_save_root
    
    if os.path.exists(dataset_embedding_path):
        print("load existing dataset embeddings: ", dataset_embedding_path)
        embeddings = torch.load(dataset_embedding_path)
    else:
        print("calculating embeddings, please waiting......")
        embeddings = cache_dataset_embeddings(dataset_root, embeddings_save_root, interval)
    
    dets, CLASSES = infer(infer_img_path, embeddings, num_cls)
    
    img_save_path = "results/v1/%s" % os.path.basename(infer_img_path)
    plot_img(infer_img_path, dets, CLASSES, img_save_path)


if __name__ == "__main__":
    """
    pipeline:
    1. cache dataset embeddings
        a. traverse all videos in the dataset, for each video, read it to frame, for each frame, do:
        b. detect cars by coco model
        c. crop the cars
        d. extract the embeddings of these cars
        e. save all the embeddings
        f. use k-means to obtain k different avg_embeddings
    2. inference
        a. read an image
        b. detect cars by coco model
        c. crop detected cars and extract these cars' embedding respectively
        d. traverse these detected cars' embedding, do:
        e. use k-means model to predict each detected cars' category
        f. output & save result
    """
    device = "cuda"
    coco_ckpt_path = "ckpts/coco_yolov5x.torchscript"
    coco_model = load_model(coco_ckpt_path, device)
    vit_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
    
    
    dataset_root = "/home/ubuntu/codes/SimilarVan/Database"
    embeddings_save_root = "embeddings/v1"
    infer_img_path = "/home/ubuntu/codes/SimilarVan/data/Delivery_Van/train/20230710170713.MP4_000011.jpg"
    # infer_img_path = "/home/ubuntu/codes/SimilarVan/Delivery_Van/train/20230710170432.MP4_000139.jpg"
    # infer_img_path = "/home/ubuntu/codes/SimilarVan/Delivery_Van/val/20230710165403.MP4_000092.jpg"
    # infer_img_path = "/home/ubuntu/codes/SimilarVan/Delivery_Van/val/20230710170432.MP4_000099.jpg"
    # infer_img_path = "/home/ubuntu/codes/SimilarVan/Delivery_Van/val/20230710164737.MP4_000134.jpg"
    # infer_img_path = "/home/ubuntu/codes/SimilarVan/Delivery_Van/val/20230710170713.MP4_000027.jpg"
    # infer_img_path = "/home/ubuntu/codes/SimilarVan/Delivery_Van/val/20230710170713.MP4_000095.jpg"
    
    interval=10
    num_cls=5
    main(dataset_root, embeddings_save_root, infer_img_path, interval, num_cls)
    
