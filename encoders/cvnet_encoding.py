import numpy as np
import torch
import cv2
import os
import shutil
import torch.nn.functional as F

from glob import glob
from tqdm import tqdm
from PIL import Image

# import sys
# sys.path.append(os.getcwd())
# print(os.path.abspath(__file__))

from core.dataset import DataSet
from models.CVNet_Rerank_model import CVNet_Rerank



class CVNetEncoding(object):
    def __init__(self, ckpt_path="ckpts/CVNet_R50.pth", depth=50, reduction_dim=2048):
        # Build the model
        print("=> creating CVNet_Rerank model")
        model = CVNet_Rerank(depth, reduction_dim)
        # print("model: ", model)
        cur_device = torch.cuda.current_device()
        model = model.cuda(device=cur_device)
        # Load checkpoint
        self.load_checkpoint(ckpt_path, model)
        self.model = model

    def load_checkpoint(self, checkpoint_file, model):
        """Loads the checkpoint from the given file."""
        err_str = "Checkpoint '{}' not found"
        assert os.path.exists(checkpoint_file), err_str.format(checkpoint_file)
        # Load the checkpoint on CPU to avoid GPU mem spike
        checkpoint = torch.load(checkpoint_file, map_location="cpu")
        try:
            state_dict = checkpoint["model_state"]
        except KeyError:
            state_dict = checkpoint

        model_dict = model.state_dict()
        state_dict = {k : v for k, v in state_dict.items()}
        weight_dict = {k : v for k, v in state_dict.items() if k in model_dict and model_dict[k].size() == v.size()}

        if len(weight_dict) == len(state_dict):
            print('All parameters are loaded')
        else:
            raise AssertionError("The model is not fully loaded.")

        model_dict.update(weight_dict)
        model.load_state_dict(model_dict)

        return checkpoint    
    
    def embedding_extract(self, imgs):
        """
        :param: imgs: list, each is a numpy img with shape, h * w * 3

        returns:
            shape: len(imgs) * 768
        """
        data_dir = "./tmp"
        os.makedirs(data_dir, exist_ok=True)
        for idx, img in enumerate(imgs):
            cv2.imwrite("%s/%s.jpg" % (data_dir, idx), img)
        scale_list = [0.7071, 1.0, 1.4142]

        with torch.no_grad():
            dataset = DataSet(data_dir, scale_list)
            # Create a loader
            test_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                sampler=None,
                num_workers=4,
                pin_memory=False,
                drop_last=False,
            )
            
            img_feats = [[] for i in range(len(scale_list))] 

            # import pdb
            # pdb.set_trace()
            for im_list in tqdm(test_loader):
                for idx in range(len(im_list)):
                    im_list[idx] = im_list[idx].cuda()
                    desc = self.model.extract_global_descriptor(im_list[idx])
                    if len(desc.shape) == 1:
                        desc.unsqueeze_(0)
                    desc = F.normalize(desc, p=2, dim=1)
                    img_feats[idx].append(desc.detach().cpu())

            for idx in range(len(img_feats)):
                # print("len(img_feats): ", idx, len(img_feats[idx]))
                img_feats[idx] = torch.cat(img_feats[idx], dim=0)
                if len(img_feats[idx].shape) == 1:
                    img_feats[idx].unsqueeze_(0)
            img_feats_agg = F.normalize(torch.mean(torch.cat([img_feat.unsqueeze(0) for img_feat in img_feats], dim=0), dim=0), p=2, dim=1)
            # img_feats_agg = img_feats_agg.cpu().numpy()
        shutil.rmtree(data_dir)

        return img_feats_agg

        
if __name__ == "__main__":
    ckpt_path="ckpts/CVNet_R50.pth"
    depth=50
    reduction_dim=2048
    model = CVNetEncoding(ckpt_path, depth, reduction_dim)
    imgs = [np.random.rand(224, 128, 3), np.random.rand(128, 234, 3), np.random.rand(111, 222, 3)]
    model.embedding_extract(imgs)
