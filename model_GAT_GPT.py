import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from gat import GraphAttentionLayer

class GPT4TS(nn.Module):
    def __init__(self, device="cuda:0", gpt_layers=6):
        super(GPT4TS, self).__init__()
        self.gpt2 = GPT2Model.from_pretrained(
            "gpt2", output_attentions=True, output_hidden_states=True
        )
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
        # hidden dimension
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

        self.gpt = GPT4TS(device=self.device, gpt_layers=6)

        self.gat = GraphAttentionLayer(
            in_features=gpt_channel,
            out_features=gpt_channel,
            n_heads=3 * 8,
            is_concat=True,
            dropout=dropout,
        )

        # regression
        self.regression_layer = nn.Conv2d(
            gpt_channel, self.output_len, kernel_size=(1, 1)
        )

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def forward(self, history_data):
        batch_size, _, num_nodes, _ = history_data.shape

        # reshape
        history_data = history_data.transpose(1, 2).contiguous()
        history_data = (
            history_data.view(batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        )
        data_st = self.start_conv(history_data)

        data_st = data_st.permute(0, 2, 3, 1)
        data_st = data_st.view(batch_size, num_nodes, -1)
        
        data_st = self.gat(data_st, self.adj.unsqueeze(-1)) + data_st
        data_st = self.gpt(data_st)
        
        data_st = data_st.permute(0, 2, 1).unsqueeze(-1)

        prediction = self.regression_layer(data_st)
        return prediction
