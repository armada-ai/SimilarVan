from model import TrajectoryModel
import importlib
import pickle
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import time

from mpl_toolkits import mplot3d


def load_trajectories(scenario_list, obs_len):
        
    ped, neis = [], []

    n_neighbors = []
    
    obs_len = 8
    dist_threshold = 5
    translation = True
    rotation = True
    smooth = True
    window_size = 3
    
    for item in scenario_list:  
        ped_obs_traj, ped_pred_traj, neis_traj = item[0], item[1], item[2] # [T 2] [N T 2] N is not a fixed number
        ped_traj = np.concatenate((ped_obs_traj[:, :2], ped_pred_traj), axis=0)
        neis_traj = neis_traj[:, :, :2].transpose(1, 0, 2)
        neis_traj = np.concatenate((np.expand_dims(ped_traj, axis=0), neis_traj), axis=0)
        distance = np.linalg.norm(np.expand_dims(ped_traj, axis=0) - neis_traj, axis=-1)
        distance = distance[:, :obs_len]
        distance = np.mean(distance, axis=-1) # mean distance
        # distance = distance[:, -1] # final distance
        neis_traj = neis_traj[distance < dist_threshold]

        n_neighbors.append(neis_traj.shape[0])
        if translation:
            origin = ped_traj[obs_len-1:obs_len] # [1, 2]
            ped_traj = ped_traj - origin
            if neis_traj.shape[0] != 0:
                neis_traj = neis_traj - np.expand_dims(origin, axis=0) 
            
        if rotation:
            ref_point = ped_traj[0]
            angle = np.arctan2(ref_point[1], ref_point[0])
            rot_mat = np.array([[np.cos(angle), -np.sin(angle)],
                                          [np.sin(angle), np.cos(angle)]])
            ped_traj = np.matmul(ped_traj, rot_mat)
            if neis_traj.shape[0] != 0:
                rot_mat = np.expand_dims(rot_mat, axis=0)
                rot_mat = np.repeat(rot_mat, neis_traj.shape[0], axis=0)
                neis_traj = np.matmul(neis_traj, rot_mat)

        if smooth:
            pred_traj = ped_traj[obs_len:]
            x_len = pred_traj.shape[0]
            x_list = []
            keep_num = int(np.floor(window_size / 2))
            for i in range(window_size):
                x_list.append(pred_traj[i:x_len-window_size+1+i])
            x = sum(x_list) / window_size
            x = np.concatenate((pred_traj[:keep_num], x, pred_traj[-keep_num:]), axis=0)
            ped_traj = np.concatenate((ped_traj[:obs_len], x), axis=0)

        ped.append(ped_traj)
        neis.append(neis_traj)
            
    max_neighbors = max(n_neighbors)
    neis_pad = []
    neis_mask = []
    for neighbor, n in zip(neis, n_neighbors):
        neis_pad.append(
            np.pad(neighbor, ((0, max_neighbors-n), (0, 0),  (0, 0)), 
            "constant")
        )
        mask = np.zeros((max_neighbors, max_neighbors))
        mask[:n, :n] = 1
        neis_mask.append(mask)

    ped = np.stack(ped, axis=0) 
    neis = np.stack(neis_pad, axis=0)
    neis_mask = np.stack(neis_mask, axis=0)

    ped = torch.tensor(ped, dtype=torch.float32)
    neis = torch.tensor(neis, dtype=torch.float32)
    neis_mask = torch.tensor(neis_mask, dtype=torch.int32)
    
    ped = ped.cuda()
    neis = neis.cuda()
    neis_mask = neis_mask.cuda() 

    ped_obs = ped[:, :obs_len]
    gt = ped[:, obs_len:]
    neis_obs = neis[:, :, :obs_len]
    return ped_obs, gt, neis_obs, neis_mask


def load_model(dataset_name, obs_len, pred_len):
    
    hp_config = importlib.import_module("config.%s" % dataset_name)
    model_hidden_dim = hp_config.model_hidden_dim
    
    model = TrajectoryModel(in_size=2, obs_len=obs_len, pred_len=pred_len, embed_size=model_hidden_dim, enc_num_layers=2, int_num_layers_list=[1,1], heads=4, forward_expansion=2)
    model = model.cuda()
    model.eval()
    return model
    

def load_motion_modes(dataset_name):
    with open("data/%s_motion_modes.pkl" % (dataset_name), "rb+") as f:
        motion_modes = pickle.load(f)
        motion_modes = torch.tensor(motion_modes, dtype=torch.float32).cuda()
    return motion_modes
    

def vis_predicted_trajectories(obs_traj, gt, pred_trajs, pred_probabilities):
    # obs_traj [B T_obs 2]
    # gt [B T_pred 2]
    # pred_trajs [B 20 T_pred 2]
    # pred_probabilities [B 20]
    title = "traj_%04d"
    for i in range(obs_traj.shape[0]):
        plt.clf()
        curr_obs = obs_traj[i].detach().cpu().numpy() # [T_obs 2]
        curr_gt = gt[i].detach().cpu().numpy()
        curr_preds = pred_trajs[i].detach().cpu().numpy()
        curr_pros = pred_probabilities[i].detach().cpu().numpy()
        obs_len = curr_obs.shape[0]
        
        ax = plt.axes(projection='3d')
        
        x_obs, y_obs, z_obs = list(range(0, obs_len)), curr_obs[:, 0], curr_obs[:, 1]
        ax.plot3D(x_obs, y_obs, z_obs, color="blue")
        ax.scatter3D(x_obs, y_obs, z_obs, color="blue", marker="*")
        
        # import pdb
        # pdb.set_trace()
        
        x_gt, y_gt, z_gt = list(range(obs_len, curr_gt.shape[0] + obs_len)), curr_gt[:, 0], curr_gt[:, 1]
        ax.plot3D(x_gt, y_gt, z_gt, color="green")
        ax.scatter3D(x_gt, y_gt, z_gt, color="green", marker="*")
        
        colors = ["red", "yellow", "orange"]
        markers = [".", "o", "v"]
        for j in range(curr_preds.shape[0]):
            if j >= len(colors):
                continue
            preds = curr_preds[j]
            y_pred, z_pred = preds[:, 0], preds[:, 1]
            ax.plot3D(x_gt, y_pred, z_pred, color=colors[j])
            ax.scatter3D(x_gt, y_pred, z_pred, color=colors[j], marker=markers[j], label="conf: %.2f" % curr_pros[j])
            
        ax.set_xlabel("sequence time")
        ax.set_ylabel("frame x_coord")
        ax.set_zlabel("frame y_coord")
        ax.set_title(title % i)
        plt.legend()
            
        
        plt.tight_layout()
        save_path = './fig/' + dataset_name
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(save_path + '/' + (title % i) + '.png')
        

if __name__ == "__main__":
    dataset_name = "sdd"

    obs_len = 8 # video observation length
    pred_len = 12 # video prediction length

    model = load_model(dataset_name, obs_len, pred_len)
    motion_modes = load_motion_modes(dataset_name)


    with open("data/%s_test.pkl" % (dataset_name), "rb+") as f:
        scenario_list = pickle.load(f)
    ped_obs, gt, neis_obs, mask = load_trajectories(scenario_list[: 2], obs_len)

    # n, topk, pred_len*2; n, 100
    pred_trajs, scores = model(ped_obs, neis_obs, motion_modes, mask, None, test=True)
    # n, topk
    top_k_scores = torch.topk(scores, k=20, dim=-1).values
    top_k_scores = F.softmax(top_k_scores, dim=-1)
    # n, topk, pred_len, 2 
    pred_trajs = pred_trajs.reshape(pred_trajs.shape[0], pred_trajs.shape[1], gt.shape[1], 2)
    # n, 1, pred_len, 2
    gt_ = gt.unsqueeze(1)
    # n, topk, pred_len, 2
    norm_ = torch.norm(pred_trajs - gt_, p=2, dim=-1)
    # n, topk, pred_len
    ade_ = torch.mean(norm_, dim=-1)
    # n, topk
    fde_ = norm_[:, :, -1]
    vis_predicted_trajectories(ped_obs, gt, pred_trajs.reshape(pred_trajs.shape[0], pred_trajs.shape[1], gt.shape[-2], -1), top_k_scores)