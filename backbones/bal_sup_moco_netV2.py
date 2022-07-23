# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn


class BalSupMoCoNet(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(BalSupMoCoNet, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("ba_queue", torch.randn(K, dim))
        self.register_buffer("nonba_queue", torch.randn(K, dim))
        self.ba_queue = nn.functional.normalize(self.ba_queue, dim=0)
        self.nonba_queue = nn.functional.normalize(self.nonba_queue, dim=0)

        self.register_buffer("ba_queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("nonba_queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, targets):

        ba_ptr = int(self.ba_queue_ptr)
        nonba_ptr = int(self.nonba_queue_ptr)

        nonba_ids = torch.where(targets == torch.tensor([0]).cuda())[0]
        ba_ids = torch.where(targets == torch.tensor([1]).cuda())[0]

        # replace the keys at ptr (dequeue and enqueue)
        if self.K - ba_ptr >= len(ba_ids):
            self.ba_queue[ba_ptr:ba_ptr + len(ba_ids),:] = keys[ba_ids]
            ba_ptr = (ba_ptr + len(ba_ids)) % self.K  # move pointer
        else:
            bias = len(ba_ids) - (self.K - ba_ptr)
            self.ba_queue[ba_ptr:self.K,:] = keys[ba_ids[:self.K - ba_ptr]]
            self.ba_queue[:bias,:] = keys[ba_ids[self.K - ba_ptr:]]
            ba_ptr = bias

        if self.K - nonba_ptr >= len(nonba_ids):
            self.nonba_queue[nonba_ptr:nonba_ptr + len(nonba_ids),:] = keys[nonba_ids]
            nonba_ptr = (nonba_ptr + len(nonba_ids)) % self.K  # move pointer
        else:
            bias = len(nonba_ids) - (self.K - nonba_ptr)
            self.nonba_queue[nonba_ptr:self.K,:] = keys[nonba_ids[:self.K - nonba_ptr]]
            self.nonba_queue[:bias,:] = keys[nonba_ids[self.K - nonba_ptr:]]
            nonba_ptr = bias

        self.ba_queue_ptr[0] = ba_ptr
        self.nonba_queue_ptr[0] = nonba_ptr


    def forward(self, im_q, im_k, targets):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)
        
        # for k
        qk_sim_matrix = torch.div(torch.matmul(q, k.T), self.T)
        mask_qk = (targets.unsqueeze(0) == targets.unsqueeze(1)) # boolean
        diagonal_mask  = torch.scatter(
            torch.ones_like(mask_qk),
            1, # 以行为单位替换处理
            torch.arange(mask_qk.shape[0]).view(-1, 1).cuda(),
            0
        )# 相当于单位矩阵取反，即对角线为0，其余为1
        mask_qk = mask_qk * diagonal_mask # 去掉自身

        # for nonba
        q_nonba_sim_matrix = torch.div(torch.matmul(q, self.nonba_queue.clone().detach().T), self.T)
        mask_q_nonba = (targets.unsqueeze(1) == torch.zeros((1,self.K)).cuda()) # boolean
        # for ba
        q_ba_sim_matrix = torch.div(torch.matmul(q, self.ba_queue.clone().detach().T), self.T)
        mask_q_ba = (targets.unsqueeze(1) == torch.ones((1,self.K)).cuda()) # boolean

        
        # # for numerical stability
        # qk_sim_max = torch.max(qk_sim_matrix, dim=1, keepdim=True)[0]
        # qk_sim_matrix = qk_sim_matrix - qk_sim_max.detach()

        # q_nonba_sim_max = torch.max(q_nonba_sim_matrix, dim=1, keepdim=True)[0]
        # q_nonba_sim_matrix = q_nonba_sim_matrix - q_nonba_sim_max.detach()

        # q_ba_sim_max = torch.max(q_ba_sim_matrix, dim=1, keepdim=True)[0]
        # q_ba_sim_matrix = q_ba_sim_matrix - q_ba_sim_max.detach()


        sim_matrix = torch.cat([qk_sim_matrix, q_nonba_sim_matrix, q_ba_sim_matrix], dim=1)

        bank_mask = torch.cat([mask_q_nonba, mask_q_ba], dim=1) # boolean

        # without_pos_bank_mask = torch.cat([torch.ones_like(mask_qk), (~bank_mask).float()], dim=1) # float, use negative in qk
        without_pos_bank_mask = torch.cat([mask_qk, (~bank_mask).float()], dim=1) # float, do not use negative in qk

        log_prob = qk_sim_matrix - torch.log((torch.exp(sim_matrix) * without_pos_bank_mask).sum(1))

        mean_log_prob_pos = (log_prob * mask_qk.float()).sum(1) / mask_qk.sum(1)

        loss = - mean_log_prob_pos.mean()

        self._dequeue_and_enqueue(k, targets)

        return loss

