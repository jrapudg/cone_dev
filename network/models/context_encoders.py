import numpy as np
import torch
import torch.nn as nn


def init(module, weight_init, bias_init, gain=1):
    '''
    This function provides weight and bias initializations for linear layers.
    '''
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class MapEncoderPts(nn.Module):
    '''
    This class operates on the road lanes provided as a tensor with shape
    (B, num_road_segs, num_pts_per_road_seg, k_attr+1)
    '''
    def __init__(self, d_k, map_attr=3, dropout=0.1):
        super(MapEncoderPts, self).__init__()
        self.dropout = dropout
        self.d_k = d_k
        #print("d_k: {}".format(d_k))
        
        self.map_attr = map_attr
        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        
        #print("Map Encoders Pts")
        self.road_pts_lin = nn.Sequential(init_(nn.Linear(map_attr, self.d_k)))
        self.road_pts_attn_layer = nn.MultiheadAttention(self.d_k, num_heads=8, dropout=self.dropout)
        self.norm1 = nn.LayerNorm(self.d_k, eps=1e-5)
        self.norm2 = nn.LayerNorm(self.d_k, eps=1e-5)
        self.map_feats = nn.Sequential(
            init_(nn.Linear(self.d_k, self.d_k)), nn.ReLU(), nn.Dropout(self.dropout),
            init_(nn.Linear(self.d_k, self.d_k)),
        )

    def get_road_pts_mask(self, roads):
        road_segment_mask = torch.sum(roads[:, :, :, -1], dim=2) == 0
        road_pts_mask = (1.0 - roads[:, :, :, -1]).type(torch.BoolTensor).to(roads.device).view(-1, roads.shape[2])
        road_pts_mask[:, 0][road_pts_mask.sum(-1) == roads.shape[2]] = False  # Ensures no NaNs due to empty rows.
        return road_segment_mask, road_pts_mask

    def forward(self, roads, agents_emb):
        '''
        :param roads: (B, S, P, k_attr+1)  where B is batch size, S is num road segments, P is
        num pts per road segment.
        :param agents_emb: (T_obs, B, d_k) where T_obs is the observation horizon. This tensor is obtained from
        AutoBot's encoder, and basically represents the observed socio-temporal context of agents.
        :return: embedded road segments with shape (S)
        '''
        B = roads.shape[0]
        S = roads.shape[1]
        P = roads.shape[2]
        road_segment_mask, road_pts_mask = self.get_road_pts_mask(roads)
        road_pts_feats = self.road_pts_lin(roads[:, :, :, :self.map_attr]).view(B*S, P, -1).permute(1, 0, 2)
        #print("Road points feature size: {}".format(road_pts_feats.size()))
        
        # Combining information from each road segment using attention with agent contextual embeddings as queries.
        agents_emb = agents_emb[-1].unsqueeze(2).repeat(1, 1, S, 1).view(-1, self.d_k).unsqueeze(0)
        #print("Agents embeddings size: {}".format(agents_emb.size()))
        road_seg_emb = self.road_pts_attn_layer(query=agents_emb, key=road_pts_feats, value=road_pts_feats,
                                                key_padding_mask=road_pts_mask)[0]
        #print("Road segments embeddings size: {}".format(road_seg_emb.size()))
        road_seg_emb = self.norm1(road_seg_emb)
        road_seg_emb2 = road_seg_emb + self.map_feats(road_seg_emb)
        road_seg_emb2 = self.norm2(road_seg_emb2)
        #print("Road segments 2 embeddings size: {}".format(road_seg_emb2.size()))
        road_seg_emb = road_seg_emb2.view(B, S, -1)
        #print("Road segments embeddings size after view: {}".format(road_seg_emb.size()))
        return road_seg_emb.permute(1, 0, 2), road_segment_mask


