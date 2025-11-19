from __future__ import absolute_import
from __future__ import division
import torch
import math
from torch import nn
from torch.nn import functional as F
import sys

class MICA_Net(nn.Module):
    def __init__(self):
        super(MICA_Net, self).__init__()

        self.encoder1 = nn.Linear(768, 64)
        self.encoder2 = nn.Linear(600, 64)

        self.affine_a = nn.Linear(2, 8, bias=False)
        self.affine_v = nn.Linear(2, 8, bias=False)

        self.W_a = nn.Linear(64, 32, bias=False)
        self.W_v = nn.Linear(64, 32, bias=False)

        self.W_ca = nn.Linear(8, 32, bias=False)
        self.W_cv = nn.Linear(8, 32, bias=False)

        self.W_ha = nn.Linear(32, 8, bias=False)
        self.W_hv = nn.Linear(32, 8, bias=False)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.classifier = nn.Sequential(nn.Linear(1024, 128), nn.Dropout(0.6), nn.Linear(128, 32))
        # self.regressor = nn.Sequential(nn.Dropout(0.6), nn.Linear(1024, 12))
        self.alpha = nn.Parameter(torch.tensor(0.5))


    def forward(self, f1_norm, f2_norm):
        sequence_outs = []

        for i in range(f1_norm.shape[0]):
            imufts = f1_norm[i] # Corresponds to F_i
            visfts = f2_norm[i] # Corresponds to F_v
            imu_fts = self.encoder1(imufts)

            vis_fts = self.encoder2(visfts)
            
            # --- Step 1: Compute C_v and C_i  ---
            # Note: The code calculates 'imu_att' (Ci) and 'vis_att' (Cv)
            imu_vis_fts = torch.cat((imu_fts, vis_fts))
            
            # Calculation for C_i (IMU Attention)
            a_t = self.affine_a(imu_vis_fts.transpose(0, 1))
            att_imu = torch.mm(imu_fts, a_t)
            imu_att = self.tanh(torch.div(att_imu, math.sqrt(imu_vis_fts.shape[1]))) # This is C_i

            # Calculation for C_v (Visual Attention)
            v_t = self.affine_v(imu_vis_fts.transpose(0, 1))
            att_vis = torch.mm(vis_fts, v_t)
            vis_att = self.tanh(torch.div(att_vis, math.sqrt(imu_vis_fts.shape[1]))) # This is C_v


            # --- Step 2: Compute H_v and H_i ---
            # Note: H_a in code corresponds to H_i (IMU) in image
            # Note: H_v in code corresponds to H_v (Visual) in image
            
            # Corresponds to H_i = ReLU((1-alpha)(W_if * F_i + W_ic * C_i^T))
            # (The code flips alpha logic slightly or assumes alpha is for IMU here)
            H_a = self.relu(self.alpha *(self.W_ca(imu_att) +self.W_a(imu_fts))) 
            
            # Corresponds to H_v = ReLU(alpha(W_vf * F_v + W_vc * C_v^T))
            H_v = self.relu((1-self.alpha) *(self.W_cv(vis_att) + self.W_v(vis_fts)))


            # --- Step 3: Compute X_att,v and X_att,i  ---
            # Note: att_imu_features corresponds to X_att,i
            # Note: att_visual_features corresponds to X_att,v

            # Corresponds to X_att,i = (1-alpha)W_ih * H_i + F_i
            att_imu_features = self.alpha * self.W_ha(H_a).transpose(0, 1) + imu_fts
            
            # Corresponds to X_att,v = alpha * W_vh * H_v + F_v
            att_visual_features = (1-self.alpha) * self.W_hv(H_v).transpose(0, 1) + vis_fts

            imu_visualfeatures = torch.cat((att_imu_features, att_visual_features), 1)
            imu_visualfeatures = torch.flatten(imu_visualfeatures)
            outs = self.classifier(imu_visualfeatures)
            sequence_outs.append(outs)

        final_outs = torch.stack(sequence_outs)
        return final_outs
