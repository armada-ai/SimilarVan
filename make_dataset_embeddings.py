import torch
import cv2
import os
import json
import shutil

from glob import glob
from tqdm import tqdm

from detectors import choose_det_model
from encoders import choose_encoding_model


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



def save_crop_imgs(det_model, img_root, save_root, **kwargs):
    img_paths = glob("%s/*" % img_root)
    os.makedirs(save_root, exist_ok=True)
    for img_path in tqdm(img_paths):
        if os.path.isdir(img_path):
            continue
        dets = det_model.detect(img_path, **kwargs)
        
        img = cv2.imread(img_path)
        crops = crop_img(img, dets)
        for idx, crop in enumerate(crops):
            cv2.imwrite("%s/%s_%04d.jpg" % (save_root, os.path.basename(img_path).split(".")[0], idx), crop)

            
def save_embeddings(embeddings, save_path):
    """
    :param save_path: embedding save path
    """
    if save_path and len(save_path) > 0:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(embeddings, save_path)            


def cache_dataset_embeddings(encoder, img_root, save_root, mode="mean"):
    """
    :param encoder: image encoder
    :param img_root: 
    :param save_root:
    :param mode: the way to summarize the embeddings. 
                "mean": get the average embeddings; 
                "sum": get the sum embeddings;
                "seperate: get the seperate embeddings;
    """
    if os.path.exists(save_root):
        print("%s exists, delete!!!" % save_root)
        shutil.rmtree(save_root)
    img_paths = glob("%s/*" % (img_root))
    
    img_embeddings = []
    
    for img_path in tqdm(img_paths):
        if os.path.isdir(img_path):
            continue
        img = cv2.imread(img_path)
        img_embedding = encoder.embedding_extract([img])
        img_embeddings.append(img_embedding)
        
        save_path = "%s/%s.emb" % (save_root, os.path.basename(img_path))
        save_embeddings(img_embeddings, save_path)
        
    if mode.lower() == "mean":
        img_embeddings_ = torch.cat(img_embeddings).mean(dim=0)
    elif mode.lower() == "sum":
        img_embeddings_ = torch.cat(img_embeddings).sum(dim=0)
    elif mode.lower() == "seperate":
        img_embeddings_ = torch.cat(img_embeddings)
    else:
        modes = ["mean", "sum", "seperate"]
        raise Error("only support %s mode" % modes)
    print("img_embeddings_.shape: ", img_embeddings_.shape)
    
    torch.save(img_embeddings_, "%s/dataset_embeddings.emb" % save_root)
    return img_embeddings_
  

def test_rightness():
    with open("configs.json") as f:
            configs = json.load(f)
    print(json.dumps(configs, indent=4))
    
    # step 1: crop images in the img_root
    det_args = configs["det_args"]
    for det_model_name in det_args.keys():
        det_model = choose_det_model(**det_args[det_model_name]["init_params"])
        img_root = "data/AmazonVan"
        crop_save_root = "%s/crop" % img_root

        # save_crop_imgs(det_model, img_root, crop_save_root, **det_args[det_model_name]["det_params"])

        """
        before starting step2, You need to clean the cropped images to obtain a reliable dataset
        """

        # step 3: extract embeddins
        encoding_args = configs["encoding_args"]
        for encoding_model_name in encoding_args.keys():
            encoding_model = choose_encoding_model(**encoding_args[encoding_model_name]["init_params"])
    
            for mode in ["mean", "seperate"]:
                embedding_save_root = "embeddings/%s_%s_%s_%s" % (det_model_name, encoding_model_name, os.path.basename(img_root), mode)
                print(embedding_save_root)
                cleaned_img_root = img_root
                cache_dataset_embeddings(encoding_model, cleaned_img_root, embedding_save_root, mode)  
    
        
def main():
    """
    whole pipeline as follows:
    1. using detection model to crop images in the image root
    2. clean up the cropped images
    3. extract cropped images' embeddings
    """
    with open("configs.json") as f:
            configs = json.load(f)
    print(json.dumps(configs, indent=4))
    
    # step 1: crop images in the img_root
    det_args = configs["det_args"]
    det_model_name = "yolov5_coco"
    det_model = choose_det_model(**det_args[det_model_name]["init_params"])
    img_root = "data/Delivery_Van_"
    crop_save_root = "%s/crop" % img_root
    
    save_crop_imgs(det_model, img_root, crop_save_root, **det_args[det_model_name]["det_params"])

    """
    before starting step2, You need to clean the cropped images to obtain a reliable dataset
    """
    
    # step 3: extract embeddins
    encoding_args = configs["encoding_args"]
    encoding_model_name = "cvnet"
    encoding_model = choose_encoding_model(**encoding_args[encoding_model_name]["init_params"])
    
    mode = "seperate"
    embedding_save_root = "embeddings/%s_%s_%s_%s" % (det_model_name, encoding_model_name, os.path.basename(img_root), mode)
    print(embedding_save_root)
    cleaned_img_root = "%s/crop" % img_root
    cache_dataset_embeddings(encoding_model, cleaned_img_root, embedding_save_root, mode)    

    
    
if __name__ == "__main__":
    main()
    # test_rightness()