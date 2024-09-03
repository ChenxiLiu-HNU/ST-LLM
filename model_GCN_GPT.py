import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from gcn import GraphConvolution

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, output, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, output)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        # print(x.shape)
        x = self.gc2(x, adj)
        # print(x.shape)
        # print('a', F.log_softmax(x, dim=1).shape)
        return F.log_softmax(x, dim=1)

class GPT4TS(nn.Module):
    def __init__(self, device="cuda:0", gpt_layers=6):
        super(GPT4TS, self).__init__()
        self.gpt2 = GPT2Model.from_pretrained(
            "gpt2", output_attentions=True, output_hidden_states=True
        )  # loads a pretrained GPT-2 base model
        self.gpt2.h = self.gpt2.h[:gpt_layers]
        print("gpt2 = {}".format(self.gpt2))

        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if "ln" in name or "wpe" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, x):
        return self.gpt2(inputs_embeds=x).last_hidden_state

class ST_LLM(nn.Module):
    def __init__(
        self,
        device,
        adj,
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
        self.adj = adj
        self.adj = torch.Tensor(self.adj).to(self.device)

        if num_nodes == 170 or num_nodes == 307:
            time = 288
        elif num_nodes == 250 or num_nodes == 266:
            time = 48

        gpt_channel = 768

        self.start_conv = nn.Conv2d(
            self.input_dim * self.input_len, gpt_channel, kernel_size=(1, 1)
        )

        self.gcn = GCN(gpt_channel, self.node_dim, gpt_channel, dropout=dropout)

        self.gpt = GPT4TS(device=self.device, gpt_layers=6)

        self.regression_layer = nn.Conv2d(
            gpt_channel, self.output_len, kernel_size=(1, 1)
        )

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def forward(self, history_data):
        batch_size, _, num_nodes, _ = history_data.shape

        history_data = history_data.transpose(1, 2).contiguous()
        history_data = (
            history_data.view(batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        )

        history_data = self.start_conv(history_data)

        data_st = self.gcn(history_data, self.adj) + history_data

        data_st = data_st.permute(0, 2, 1, 3).squeeze(-1)
        data_st = self.gpt(data_st)
        data_st = data_st.permute(0, 2, 1).unsqueeze(-1)

        prediction = self.regression_layer(data_st)  # [64, 12, 170, 1]

        return prediction
