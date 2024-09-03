import torch
from torch import nn

class GraphAttentionLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_heads: int,
        is_concat: bool = True,
        dropout: float = 0.1,
        leaky_relu_negative_slope: float = 0.2,
    ):
        super().__init__()

        self.is_concat = is_concat
        self.n_heads = n_heads

        if is_concat:
            assert out_features % n_heads == 0
            self.n_hidden = out_features // n_heads
        else:
            self.n_hidden = out_features

        self.proj = nn.Linear(in_features, self.n_hidden * n_heads, bias=False)
        self.proj_attn = nn.Linear(in_features, self.n_hidden * n_heads // 2, bias=False)
        self.attn = nn.Linear(self.n_hidden, 1, bias=False)
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj_mat: torch.Tensor):
        batch_size, n_nodes = h.shape[0:2]
        g = self.proj(h).view(batch_size, n_nodes, self.n_heads, self.n_hidden)
        ga = self.proj_attn(h).view(batch_size, n_nodes, self.n_heads, self.n_hidden // 2)
        g_repeat = ga.repeat(1, n_nodes, 1, 1)
        g_repeat_interleave = ga.repeat_interleave(n_nodes, dim=1)
        g_concat = torch.cat([g_repeat_interleave, g_repeat], dim=-1)
        g_concat = g_concat.view(
            batch_size, n_nodes, n_nodes, self.n_heads, self.n_hidden
        )

        e = self.activation(self.attn(g_concat))
        e = e.squeeze(-1)

        assert adj_mat.shape[0] == 1 or adj_mat.shape[0] == n_nodes
        assert adj_mat.shape[1] == 1 or adj_mat.shape[1] == n_nodes
        assert adj_mat.shape[2] == 1 or adj_mat.shape[2] == self.n_heads

        e = e.masked_fill(adj_mat < 0.5, float("-inf"))

        a = self.softmax(e)
        a = self.dropout(a)

        attn_res = torch.einsum("bijh,bjhf->bihf", a, g)

        if self.is_concat:
            return attn_res.reshape(batch_size, n_nodes, self.n_heads * self.n_hidden)
        else:
            return attn_res.mean(dim=2)