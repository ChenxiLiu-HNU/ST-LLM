from time import sleep
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numpy as np
import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from einops import rearrange
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

class TemporalEmbedding(nn.Module):
    def __init__(self, time, features):
        super(TemporalEmbedding, self).__init__()

        self.time = time
        # temporal embeddings
        self.time_day = nn.Parameter(torch.empty(time, features))
        nn.init.xavier_uniform_(self.time_day)

        self.time_week = nn.Parameter(torch.empty(7, features))
        nn.init.xavier_uniform_(self.time_week)

    def forward(self, x):
        day_emb = x[..., 1]  # [64, 12, 170]
        time_day = self.time_day[
            (day_emb[:, -1, :] * self.time).type(torch.LongTensor)
        ]  # [64, 170, 64]
        time_day = time_day.transpose(1, 2).unsqueeze(-1)

        week_emb = x[..., 2]  # [64, 12, 170]
        time_week = self.time_week[
            (week_emb[:, -1, :]).type(torch.LongTensor)
        ]  # [64, 170, 64]
        time_week = time_week.transpose(1, 2).unsqueeze(-1)

        # temporal embeddings
        tem_emb = time_day + time_week
        return tem_emb


class GPT4TS(nn.Module):
    
    def __init__(self, device = "cuda:0", gpt_layers = 3):
        super(GPT4TS, self).__init__()
        config = GPT2Config(
            n_embd=768,
            n_head=4,
            n_layer=6,
            n_positions=1024,
            n_inner=3072,
            resid_pdrop=0.1,
            attn_pdrop=0.1,
            embd_pdrop=0.1
        )
        self.gpt2 = GPT2Model(config)  # loads a pretrained GPT-2 base model
        self.gpt2.h = self.gpt2.h[:gpt_layers]

    def forward(self, x):
        # [64, 42, 768]
        return self.gpt2(inputs_embeds=x).last_hidden_state


class ST_LLM(nn.Module):
    def __init__(
        self,
        device,
        input_dim=3,
        channels=64,
        num_nodes=170,
        input_len=12,
        output_len=12,
        dropout=0.1,
    ):
        super().__init__()

        # attributes
        self.device = device
        self.num_nodes = num_nodes
        self.node_dim = channels
        self.input_len = input_len
        self.input_dim = input_dim
        self.output_len = output_len

        if num_nodes == 170 or num_nodes == 307 or num_nodes == 358  or num_nodes == 883:
            time = 288
        elif num_nodes == 250 or num_nodes == 266:
            time = 48
        elif num_nodes>200:
            time = 96

        gpt_channel = 768

        self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, channels))
        nn.init.xavier_uniform_(self.node_emb)
        
        self.Temb = TemporalEmbedding(time, channels)
        
        self.start_conv = nn.Conv2d(self.input_dim*self.input_len, channels, kernel_size=(1, 1))
        
        self.concat = nn.Conv2d(channels*3, gpt_channel, kernel_size=(1, 1))

        self.gpt = GPT4TS(device=self.device,gpt_layers=1)

        self.regression_layer = nn.Conv2d(
            gpt_channel, self.output_len, kernel_size=(1, 1)
        )

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def forward(self, x):
        batch_size, channels, num_nodes, len = x.shape

        tem_emb = self.Temb(x.permute(0, 3, 2, 1))

        node_emb = []
        node_emb.append(self.node_emb.unsqueeze(0).expand(
                batch_size, -1, -1).transpose(1, 2).unsqueeze(-1))

        x = x.transpose(1, 2).contiguous()
        x = x.view(
            batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        
        x = self.start_conv(x)

        h = torch.cat([x] + [tem_emb] + node_emb, dim=1)  # [64, c, n, 1]
        h = self.concat(h)

        h = h.permute(0,2,1,3).squeeze(-1)
        h = self.gpt(h)
        h = h.permute(0,2,1).unsqueeze(-1)

        # regression
        out = self.regression_layer(h) # [64, 12, 170, 1]
        return out