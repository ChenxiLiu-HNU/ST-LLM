import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Model

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
        day_emb = x[..., 1]
        time_day = self.time_day[
            (day_emb[:, -1, :] * self.time).type(torch.LongTensor)
        ]
        time_day = time_day.transpose(1, 2).unsqueeze(-1)

        week_emb = x[..., 2]
        time_week = self.time_week[
            (week_emb[:, -1, :]).type(torch.LongTensor)
        ]
        time_week = time_week.transpose(1, 2).unsqueeze(-1)

        # temporal embeddings
        tem_emb = time_day + time_week
        return tem_emb

class PFA(nn.Module):
    def __init__(self, device="cuda:0", gpt_layers=6, U=1):
        super(PFA, self).__init__()
        self.gpt2 = GPT2Model.from_pretrained(
            "gpt2", output_attentions=True, output_hidden_states=True
        )
        self.gpt2.h = self.gpt2.h[:gpt_layers]
        self.U = U

        for layer_index, layer in enumerate(self.gpt2.h):
            for name, param in layer.named_parameters():
                if layer_index < gpt_layers - self.U:
                    if "ln" in name or "wpe" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                else:
                    if "mlp" in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True

    def forward(self, x):
        return self.gpt2(inputs_embeds=x).last_hidden_state

class ST_LLM(nn.Module):
    def __init__(
        self,
        input_dim=3,
        channels=64,
        num_nodes=170,
        input_len=12,
        output_len=12,
        llm_layer=6,
        U=1,
        device= "cuda:7"
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.node_dim = channels
        self.input_len = input_len
        self.input_dim = input_dim
        self.output_len = output_len
        self.llm_layer = llm_layer
        self.U = U
        self.device = device

        if num_nodes == 170 or num_nodes == 307:
            time = 288
        elif num_nodes == 250 or num_nodes == 266:
            time = 48

        gpt_channel = 256
        to_gpt_channel = 768

        self.Temb = TemporalEmbedding(time, gpt_channel)

        self.node_emb = nn.Parameter(torch.empty(self.num_nodes, gpt_channel))
        nn.init.xavier_uniform_(self.node_emb)

        self.start_conv = nn.Conv2d(
            self.input_dim * self.input_len, gpt_channel, kernel_size=(1, 1)
        )

        # embedding layer
        self.gpt = PFA(device=self.device, gpt_layers=self.llm_layer, U=self.U)

        self.feature_fusion = nn.Conv2d(
            gpt_channel * 3, to_gpt_channel, kernel_size=(1, 1)
        )

        # regression
        self.regression_layer = nn.Conv2d(
            gpt_channel * 3, self.output_len, kernel_size=(1, 1)
        )

    # return the total parameters of model
    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def forward(self, history_data):
        input_data = history_data
        batch_size, _, num_nodes, _ = input_data.shape
        history_data = history_data.permute(0, 3, 2, 1)

        tem_emb = self.Temb(history_data)
        node_emb = []
        node_emb.append(
            self.node_emb.unsqueeze(0)
            .expand(batch_size, -1, -1)
            .transpose(1, 2)
            .unsqueeze(-1)
        )

        input_data = input_data.transpose(1, 2).contiguous()
        input_data = (
            input_data.view(batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        )
        input_data = self.start_conv(input_data)

        data_st = torch.cat(
            [input_data] + [tem_emb] + node_emb, dim=1
        )
        data_st = self.feature_fusion(data_st)
        # data_st = F.leaky_relu(data_st)

        data_st = data_st.permute(0, 2, 1, 3).squeeze(-1)
        data_st = self.gpt(data_st)
        data_st = data_st.permute(0, 2, 1).unsqueeze(-1)

        prediction = self.regression_layer(data_st)

        return prediction
