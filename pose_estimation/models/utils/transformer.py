# Author: Chenhongyi Yang


import torch.nn as nn
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_

class FFN(nn.Module):
    def __init__(self, embed_dims, feedforward_dims, num_fcs, ffn_drop):
        super().__init__()
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_dims
        self.num_fcs = num_fcs

        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(in_channels, feedforward_dims),
                    nn.GELU(),
                    nn.Dropout(ffn_drop),
                )
            )
            in_channels = feedforward_dims

        layers.append(nn.Linear(feedforward_dims, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        return out


class CustomMultiheadAttention(nn.Module):

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=False,
        with_output_proj=True,
        device=None,
        dtype=None
    ):
        super(CustomMultiheadAttention, self).__init__()
        """
        This custom MultiHeadAttention only support the basic fuction of
        torch.nn.MultiHeadAttention
        """
        assert dropout == 0.0
        assert kdim is None
        assert vdim is None
        assert add_bias_kv == False
        assert add_zero_attn == False
        assert batch_first

        self.input_dims = embed_dim
        self.embed_dims = embed_dim
        self.num_heads = num_heads
        self.with_bias = bias

        self.head_dims = embed_dim // num_heads
        self.scale = self.head_dims**-0.5

        self.q_proj = nn.Linear(self.input_dims, embed_dim, bias=bias)
        self.k_proj = nn.Linear(self.input_dims, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.input_dims, embed_dim, bias=bias)
        if with_output_proj:
            self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        else:
            self.out_proj = None

        self._init_params()

    def _init_params(self):
        xavier_uniform_(self.q_proj.weight)
        xavier_uniform_(self.k_proj.weight)
        xavier_uniform_(self.v_proj.weight)

        if self.with_bias:
            constant_(self.q_proj.bias, 0.)
            constant_(self.k_proj.bias, 0.)
            constant_(self.v_proj.bias, 0.)
            if self.out_proj is not None:
                constant_(self.out_proj.bias, 0.)

    def forward(self, q, k, v):
        B = q.shape[0]

        _q = self.q_proj(q).reshape(B, -1, self.num_heads, self.head_dims).permute(0, 2, 1, 3)
        _k = self.k_proj(k).reshape(B, -1, self.num_heads, self.head_dims).permute(0, 2, 1, 3)
        _v = self.v_proj(v).reshape(B, -1, self.num_heads, self.head_dims).permute(0, 2, 1, 3)

        attn = (_q @ _k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ _v).transpose(1, 2).reshape(B, -1, self.embed_dims)
        if self.out_proj is not None:
            x = self.out_proj(x)
        return x, attn