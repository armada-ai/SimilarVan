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


def cache_dataset_embeddings(img_root, save_root):
    img_paths = glob("%s/*" % (img_root))
    
    img_embeddings = []
    
    for img_path in tqdm(img_paths):
        img = cv2.imread(img_path)
        save_path = "%s/%s.emb" % (save_root, os.path.basename(img_path))
        # shape: n, 768
        img_embedding = get_img_embeddings(img, save_path)
        if img_embedding is not None:
            img_embeddings.append(img_embedding)
        
    # shape: n, 768
    img_embeddings_ = torch.cat(img_embeddings).mean(dim=0)
    print("img_embeddings_.shape: ", img_embeddings_.shape)
    
    torch.save(img_embeddings_, "%s/dataset_embeddings.emb" % save_root)
    return img_embeddings_


def infer(vid_path, embeddings, interval=1):
    """
    :param: infer_img_path: the image's absolute path
    :param: embeddings: dict, {"other": other cars' embedding, "amazon": amazon car's embedding}
    :returns
        predicts category
    """
    save_root = "results/v3"
    reader = cv2.VideoCapture(vid_path)
     
    width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(round(reader.get(cv2.CAP_PROP_FPS) / interval))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    os.makedirs(os.path.dirname(save_root), exist_ok=True)
    save_vid_name = "%s/%s.mp4" % (save_root, os.path.basename(vid_path).split(".")[0])
    writer = cv2.VideoWriter(save_vid_name, fourcc, fps, (width, height),True)
        
    vid_name = os.path.basename(vid_path)#.split(".")[0]
    count = 0
    thresh = 0.7
    while True:
        ret, frame = reader.read()
        if not ret:
            break
        if count % interval == 0:
            dets = yolov5_detect(coco_model, frame, size=640)
            # print("coco_model: ", dets)
            if len(dets) == 0:
                continue
            crop_imgs = crop_img(frame, dets)
            img_embeddings = embedding_extract(crop_imgs)

            if img_embeddings is not None:
                embeddings__ = embeddings.unsqueeze(0)
                embeddings_ = embeddings__ / embeddings__.norm(dim=-1, keepdim=True)
                img_embeddings_ = img_embeddings / img_embeddings.norm(dim=-1, keepdim=True)
                similarity = (embeddings_ * img_embeddings_).sum(dim=-1)
                # cosine similarity
                max_idx = torch.argmax(similarity)
                max_similarity = similarity[max_idx]
                if max_similarity >= thresh:
                    det = dets[max_idx]
                    det[-1] = 0
                    save_path = "%s/%s/%06d.jpg" % (save_root, vid_name, count)
                    result_frame = plot_img(frame, [det], ["0"], save_path)
                    writer.write(result_frame)
        count += 1
    reader.release()
    writer.release()
    

def plot_img(img, dets, CLASSES, save_path):
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
    return img

    
def main(dataset_root, embeddings_save_root, infer_vid_path, interval=10):
    dataset_embedding_path = "%s/dataset_embeddings.emb" % embeddings_save_root
    
    if os.path.exists(dataset_embedding_path):
        print("load existing dataset embeddings: ", dataset_embedding_path)
        embeddings = torch.load(dataset_embedding_path)
    else:
        print("calculating embeddings, please waiting......")
        embeddings = cache_dataset_embeddings(dataset_root, embeddings_save_root)
    
    infer(infer_vid_path, embeddings, interval)
    

if __name__ == "__main__":
    """
    pipeline:
    1. cache dataset embeddings
        a. traverse all images in the dataset, for each image, do:
        b. detect cars by coco model
        c. crop the cars
        d. extract the embeddings of these cars
        e. add all the embeddings to get avg_embedding
        f. save all the embeddings
    2. inference
        a. read an video
        b. detect cars by coco model
        c. crop detected cars and extract these cars' embedding
        d. get the most similar det, and the  cosine similarity of the det should > thresh
        e. output & save result
    """
    device = "cuda"
    coco_ckpt_path = "ckpts/coco_yolov5x.torchscript"
    coco_model = load_model(coco_ckpt_path, device)
    vit_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
    
    
    dataset_root = "/home/ubuntu/codes/SimilarVan/data/AmazonVan"
    embeddings_save_root = "embeddings/v3"
    infer_vid_path = "/home/ubuntu/codes/SimilarVan/data/videos_db/20230710170432.MP4"
    
    interval=30
    main(dataset_root, embeddings_save_root, infer_vid_path, interval)
    
