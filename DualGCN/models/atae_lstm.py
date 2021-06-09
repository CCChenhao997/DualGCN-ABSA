'''
Description: 
version: 
Author: chenhao
Date: 2021-06-09 14:17:37
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import DynamicLSTM, SqueezeEmbedding, SoftAttention


class ATAE_LSTM(nn.Module):
    ''' Attention-based LSTM with Aspect Embedding '''
    def __init__(self, embedding_matrix, opt):
        super(ATAE_LSTM, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), freeze=True)
        self.squeeze_embedding = SqueezeEmbedding()
        self.lstm = DynamicLSTM(opt.embed_dim*2, opt.hidden_dim, num_layers=1, batch_first=True)
        self.attention = SoftAttention(opt.hidden_dim, opt.embed_dim)
        self.dense = nn.Linear(opt.hidden_dim, opt.polarities_dim)
    
    def forward(self, inputs):
        text, aspect_text = inputs[0], inputs[1]
        x_len = torch.sum(text != 0, dim=-1)
        x_len_max = torch.max(x_len)
        aspect_len = torch.sum(aspect_text != 0, dim=-1).float()
        
        x = self.embed(text)
        x = self.squeeze_embedding(x, x_len)
        aspect = self.embed(aspect_text)
        aspect_pool = torch.div(torch.sum(aspect, dim=1), aspect_len.view(aspect_len.size(0), 1))
        aspect = torch.unsqueeze(aspect_pool, dim=1).expand(-1, x_len_max, -1)
        x = torch.cat((aspect, x), dim=-1)
        h, _ = self.lstm(x, x_len)
        hs = self.attention(h, aspect)
        out = self.dense(hs)
        return out, None