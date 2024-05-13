import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter, Sequential, Linear, BatchNorm1d

class ABMIL(nn.Module):
    """
    Attention-based MIL classification.
    Embeddings followed by linear layer.
    """
    def __init__(self, feat_dim, output_dim):
        super().__init__()

        #embeddings are 768 dim
        self.L = feat_dim 
        self.D = (feat_dim+1) // 2 # or 128
        self.K = output_dim
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1)
        )
        
    def forward(self, data):
        #data shape = N x bag_size x 768 (N is batch_size)
        A = self.attention(data) # N x bag_size x K
        A = A.permute(0,2,1) # N x K x bag_size
        #A_noSM = A.clone()
        A = F.softmax(A, dim=2)  # softmax over bag_size
        #print(A.shape, A.mean(), A.max(), A.min())
        
        M = torch.bmm(A, data)  # N x K x L
        #Y_prob = self.linear(M)
        Y = self.classifier(M)

        return Y, A

    def save_checkpoint(self, save_path, optimizer, epoch, best_train_loss, best_val_loss, is_best):
        ckpt = {}
        ckpt["state"] = self.state_dict()
        ckpt["epoch"] = epoch
        ckpt["optimizer_state"] = optimizer.state_dict()
        ckpt["best_train_loss"] = best_train_loss
        ckpt["best_val_loss"] = best_val_loss
        torch.save(ckpt, os.path.join(save_path, "pred_model.ckpt"))
        if is_best:
            torch.save(ckpt, os.path.join(save_path, "best_pred_model.ckpt"))

    def load_checkpoint(self, load_path, optimizer, best=True):
        if best == True:
            ckpt = torch.load(os.path.join(load_path, "best_pred_model.ckpt"))
        else:
            ckpt = torch.load(os.path.join(load_path, "pred_model.ckpt"))
        self.load_state_dict(ckpt["state"])
        epoch = ckpt["epoch"]
        best_train_loss = ckpt["best_train_loss"]
        best_val_loss = ckpt["best_val_loss"]
        optimizer.load_state_dict(ckpt["optimizer_state"])
        return epoch, best_train_loss, best_val_loss

class multitask_ABMIL(ABMIL):
    '''
    Multitask classification
    '''
    def __init__(self, task_dict, feat_dim, output_dim):
        super(multitask_ABMIL, self).__init__(feat_dim, output_dim)
        # dictiionary of tasks and number of classes for each task
        self.task_dict = task_dict
        self.classifier_multi = nn.ModuleList()
        n_class = 1
        for i, task in enumerate(task_dict.keys()):
            self.classifier_multi.append(nn.Sequential(nn.Linear(self.L*self.K, 1)))

    def forward(self, data): ## x: B x num_tiles x D_feat
        #data shape = N x bag_size x 768 (N is batch_size)
        A = self.attention(data) # N x bag_size x K
        A = A.permute(0,2,1) # N x K x bag_size
        #A_noSM = A.clone()
        A = F.softmax(A, dim=2)  # softmax over bag_size
        #print(A.shape, A.mean(), A.max(), A.min())
        
        M = torch.bmm(A, data)  # N x K x L
        
        outputs_multi = {}
        for classifier, key in zip(self.classifier_multi, self.task_dict.keys()):
            outputs_multi[key] = classifier(M)

        return A, outputs_multi


class ABMIL_attention_map(multitask_ABMIL):
    '''
    Used for making attention maps for multitask classification
    (No softmax on attention)
    '''
    def __init__(self, task_dict, feat_dim, output_dim):
        super(ABMIL_attention_map, self).__init__(task_dict, feat_dim, output_dim)

    def forward(self, data): ## x: B x num_tiles x D_feat
        #data shape = N x bag_size x 768 (N is batch_size)
        A = self.attention(data) # N x bag_size x K
        A = A.permute(0,2,1) # N x K x bag_size
        
        M = torch.bmm(A, data)  # N x K x L
        
        outputs_multi = {}
        for classifier, key in zip(self.classifier_multi, self.task_dict.keys()):
            outputs_multi[key] = classifier(M)

        return A, outputs_multi

