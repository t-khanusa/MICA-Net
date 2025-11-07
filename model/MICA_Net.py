from __future__ import absolute_import
from __future__ import division
import torch
import math
from torch import nn
from torch.nn import functional as F
import sys

alpha = 0.8
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
        self.regressor = nn.Sequential(nn.Linear(1024, 128), nn.Dropout(0.6), nn.Linear(128, 32))
        # self.regressor = nn.Sequential(nn.Dropout(0.6), nn.Linear(1024, 12))


    def forward(self, f1_norm, f2_norm):
        sequence_outs = []

        for i in range(f1_norm.shape[0]):
            imufts = f1_norm[i]
            visfts = f2_norm[i]
            imu_fts = self.encoder1(imufts)

            vis_fts = self.encoder2(visfts)
            imu_vis_fts = torch.cat((imu_fts, vis_fts))
            a_t = self.affine_a(imu_vis_fts.transpose(0, 1))
            att_imu = torch.mm(imu_fts, a_t)
            imu_att = self.tanh(torch.div(att_imu, math.sqrt(imu_vis_fts.shape[1])))

            imu_vis_fts = torch.cat((imu_fts, vis_fts))
            v_t = self.affine_v(imu_vis_fts.transpose(0, 1))
            att_vis = torch.mm(vis_fts, v_t)
            vis_att = self.tanh(torch.div(att_vis, math.sqrt(imu_vis_fts.shape[1])))
            H_a = self.relu(alpha *(self.W_ca(imu_att) +self.W_a(imu_fts)))
            H_v = self.relu((1-alpha) *(self.W_cv(vis_att) + self.W_v(vis_fts)))

            att_imu_features = alpha * self.W_ha(H_a).transpose(0, 1) + imu_fts
            att_visual_features = (1-alpha) * self.W_hv(H_v).transpose(0, 1) + vis_fts

            imu_visualfeatures = torch.cat((att_imu_features, att_visual_features), 1)
            imu_visualfeatures = torch.flatten(imu_visualfeatures)
            outs = self.regressor(imu_visualfeatures)
            sequence_outs.append(outs)

        final_outs = torch.stack(sequence_outs)
        return final_outs
