import cv2
import os
import json
import torch

from detectors import choose_det_model
from encoders import choose_encoding_model
from encoders.cvnet_encoding import CVNetEncoding


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
    print("save_path: ", save_path)
    cv2.imwrite(save_path, img)
    return img


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


def infer(vid_path, det_model, encoding_model, embeddings, interval=1, similarity_thresh=0.75, save_root="results", **kwargs):
    """
    :param: infer_img_path: the image's absolute path
    :param: embeddings: 
    :returns
        predicts category
    """
    os.makedirs(save_root, exist_ok=True)
    
    reader = cv2.VideoCapture(vid_path)
     
    width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = max(int(round(reader.get(cv2.CAP_PROP_FPS) / interval)), 1)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    save_vid_name = "%s/%s.mp4" % (save_root, os.path.basename(vid_path).split(".")[0])
    print("save_vid_name: ", save_vid_name, os.path.dirname(save_vid_name))
    os.makedirs(os.path.dirname(save_vid_name), exist_ok=True)
    writer = cv2.VideoWriter(save_vid_name, fourcc, fps, (width, height),True)
        
    vid_name = os.path.basename(vid_path)#.split(".")[0]
    count = 0
    image_path = "tmp.png"
    while True:
        ret, frame = reader.read()
        if not ret:
            break
        if count % interval == 0:
            cv2.imwrite(image_path, frame)
            
            dets = det_model.detect(image_path, **kwargs)
            if len(dets) == 0:
                print("frame %s no dets, continue" % count)
                count += 1
                continue
            crop_imgs = crop_img(frame, dets)
            img_embeddings = encoding_model.embedding_extract(crop_imgs)
            
            # import pdb
            # pdb.set_trace()
            if len(embeddings.shape) == 1:
                    embeddings = embeddings.unsqueeze(0)
            
            if isinstance(encoding_model, CVNetEncoding):
                similarity = torch.matmul(img_embeddings, embeddings.T).max(dim=1)[0]
            else:
                if len(embeddings.shape) == 2 and embeddings.shape[0] != 1:
                    embeddings = embeddings.mean(dim=0)
                embeddings_ = embeddings / embeddings.norm(dim=-1, keepdim=True)
                img_embeddings_ = img_embeddings / img_embeddings.norm(dim=-1, keepdim=True)
                similarity = (embeddings_ * img_embeddings_).sum(dim=-1)
            # cosine similarity
            max_idx = torch.argmax(similarity)
            max_similarity = similarity[max_idx]
            print("\nframe %s max_similarity: %s\n" % (count, max_similarity.data))
            if max_similarity >= similarity_thresh:
                det = dets[max_idx]
                det[-1] = 0
                save_path = "%s/%s_%s/%06d.jpg" % (save_root, vid_name, similarity_thresh, count)
                result_frame = plot_img(frame, [det], ["0"], save_path)
                writer.write(result_frame)
                
        count += 1
    if os.path.exists(image_path):
        os.remove(image_path)
    reader.release()
    writer.release()
    print("saved video path: %s" % save_vid_name)
    

def test_rightness():
    with open("configs.json") as f:
        configs = json.load(f)
    print(json.dumps(configs, indent=4))
    
    det_args = configs["det_args"]
    det_model_name = "yolov5_coco"
    for det_model_name in det_args.keys():
        # if det_model_name != "yolov5_coco":
        #     continue
        det_model = choose_det_model(**det_args[det_model_name]["init_params"])

        encoding_args = configs["encoding_args"]
        for encoding_model_name in encoding_args.keys():
            # if encoding_model_name != "vit":
            #     continue
            encoding_model = choose_encoding_model(**encoding_args[encoding_model_name]["init_params"])

            dataset_name = "AmazonVan"
            infer_vid_path = "/home/ubuntu/codes/SimilarVan/data/videos_db/20230710164737.MP4"
        
            mode = "mean" # only support "mean", "seperate"
            for mode in ["mean", "seperate"]:
                # if mode != "sum":
                #     continue
                embedding_path = "embeddings/%s_%s_%s_%s/dataset_embeddings.emb" % (det_model_name, encoding_model_name, dataset_name, mode)
                if os.path.exists(embedding_path):
                    embeddings = torch.load(embedding_path)
                else:
                    print("not found: %s, please make dataset embeddings first. \nHint: see make_dataset_embeddings.py for more detail" % embedding_path)
                    continue
                interval=300
                save_root = "results/%s_%s_%s_%s" % (det_model_name, encoding_model_name, dataset_name, mode)
                print("save_root: ", save_root)
                infer(infer_vid_path, det_model, encoding_model, embeddings, interval=interval, similarity_thresh=0.75, save_root=save_root, **det_args[det_model_name]["det_params"])
    
    
def main():
    with open("configs.json") as f:
        configs = json.load(f)
    print(json.dumps(configs, indent=4))
    det_args = configs["det_args"]
    det_model_name = "yolov5_coco"
    det_model = choose_det_model(**det_args[det_model_name]["init_params"])
    
    encoding_args = configs["encoding_args"]
    encoding_model_name = "vit"
    # encoding_model_name = "cvnet"
    encoding_model = choose_encoding_model(**encoding_args[encoding_model_name]["init_params"])
    
    dataset_name = "AmazonVan"
    infer_vid_path = "/home/ubuntu/codes/SimilarVan/data/videos_db/20230710164737.MP4"
    
    mode = "mean" # only support "mean", "seperate"
    embedding_path = "embeddings/%s_%s_%s_%s/dataset_embeddings.emb" % (det_model_name, encoding_model_name, dataset_name, mode)
    if os.path.exists(embedding_path):
        embeddings = torch.load(embedding_path)
    else:
        print("not found: %s, please make dataset embeddings first. \nHint: see make_dataset_embeddings.py for more detail" % embedding_path)
    interval=30
    infer(infer_vid_path, det_model, encoding_model, embeddings, interval=interval, similarity_thresh=0.65, save_root="results", **det_args[det_model_name]["det_params"])
    
    

    
if __name__ == "__main__":
    """
    pipeline:
        a. read an video
        b. detect objects by detection model
        c. crop detected objects
        d. extract these objects' embedding by encoding model
        e. compute the similarity of these objects and the dataset 
        f. get the most similar object in the dataset, and the similarity score of the object should > similarity_thresh
        g. output & save result
    """
    # main()
    test_rightness()
    
    
    
