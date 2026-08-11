import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = context if context is not None else x
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: t.reshape(*t.shape[:-1], h, -1).permute(0, 2, 1, 3), (q, k, v))

        sim = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = mask.reshape(mask.shape[0], 1, 1, mask.shape[1])
            sim = sim.masked_fill(mask == 0, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.permute(0, 2, 1, 3).reshape(*x.shape[:-1], -1)
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, out_channels),
            nn.GELU()
        )

        self.layer1 = nn.Linear(in_channels, out_channels)
        self.layer2 = nn.Linear(out_channels, out_channels)
        self.layer_out = nn.Linear(out_channels, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        
        self.skip_connection = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_emb):
        h = self.norm1(x)
        h = self.layer1(h)
        h = F.gelu(h)

        time_emb = self.time_mlp(time_emb)
        h = h + time_emb.unsqueeze(1).repeat(1, h.shape[1], 1)
        
        h = self.norm2(h)
        h = self.layer2(h)
        h = F.gelu(h)
        h = self.dropout(h)
        h = self.layer_out(h)

        return h + self.skip_connection(x)

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, context_dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, context_dim, heads, dim_head, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dim * 4, dropout)

    def forward(self, x, context, mask=None):
        x = self.norm1(x) + self.cross_attn(x, context, mask)
        x = self.norm2(x) + self.ff(x)
        return x

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.residual = ResidualBlock(in_channels, in_channels, time_emb_dim)
        self.down = nn.Linear(in_channels, out_channels)
        self.downsample = nn.Linear(in_channels, out_channels)

    def forward(self, x, time_emb):
        residual = self.residual(x, time_emb)
        return self.down(residual), residual

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, flag=0):
        super().__init__()
        self.up = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LayerNorm(out_channels)
        )
        if flag == -1:
            self.residual = ResidualBlock(out_channels+ out_channels, out_channels, time_emb_dim)
        else:
            self.residual = ResidualBlock(out_channels+ out_channels//2, out_channels, time_emb_dim)

    def forward(self, x, skip_x, time_emb):
        x = self.up(x)
        x = torch.cat([x, skip_x], dim=-1)
        return self.residual(x, time_emb)

class TransConv(nn.Module):
    def __init__(
        self, 
        in_channels=2, 
        out_channels=2, 
        time_emb_dim=128, 
        model_channels=64,
        condition_channels=2,
        channel_mults=(1, 2, 4, 8),
        attention_levels=[1, 2],
        num_heads=8,
        dropout=0.1
    ):
        super().__init__()

        self.time_embedder = SinusoidalPositionEmbeddings(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.GELU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        self.input_proj = nn.Linear(in_channels, model_channels)

        self.condition_proj = nn.Linear(condition_channels, model_channels)

        self.down_blocks = nn.ModuleList()
        self.self_attn_down = nn.ModuleList()
        self.cross_attn_down = nn.ModuleList()
        
        channels = [model_channels]
        for i, mult in enumerate(channel_mults):
            out_channels = model_channels * mult
            self.down_blocks.append(DownBlock(channels[-1], out_channels, time_emb_dim))
            channels.append(out_channels)
            # Self-attention
            if i in attention_levels:
                self.self_attn_down.append(
                    CrossAttentionBlock(out_channels, out_channels, num_heads, out_channels // num_heads, dropout)
                )
            else:
                self.self_attn_down.append(nn.Identity())
            # Cross-attention
            if i in attention_levels:
                self.cross_attn_down.append(
                    CrossAttentionBlock(out_channels, model_channels, num_heads, out_channels // num_heads, dropout)
                )
            else:
                self.cross_attn_down.append(nn.Identity())

        self.mid_block1 = ResidualBlock(channels[-1], channels[-1], time_emb_dim)
        self.mid_self_attn  = CrossAttentionBlock(channels[-1], channels[-1], num_heads, channels[-1] // num_heads, dropout)
        self.mid_cross_attn = CrossAttentionBlock(channels[-1], model_channels, num_heads, channels[-1] // num_heads, dropout)
        self.mid_block2 = ResidualBlock(channels[-1], channels[-1], time_emb_dim)

        self.up_blocks = nn.ModuleList()
        self.self_attn_up = nn.ModuleList()
        self.cross_attn_up = nn.ModuleList()
        
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels = model_channels * mult
            if i == 0:
                self.up_blocks.append(UpBlock(channels[-1], out_channels, time_emb_dim, flag=-1))
            else:
                self.up_blocks.append(UpBlock(channels[-1], out_channels, time_emb_dim))
            channels.append(out_channels)
            # Self-attention
            if i in attention_levels:
                self.self_attn_up.append(
                    CrossAttentionBlock(out_channels, out_channels, num_heads, out_channels // num_heads, dropout)
                )
            else:
                self.self_attn_up.append(nn.Identity())
            # Cross-attention
            if i in attention_levels:
                self.cross_attn_up.append(
                    CrossAttentionBlock(out_channels, model_channels, num_heads, out_channels // num_heads, dropout)
                )
            else:
                self.cross_attn_up.append(nn.Identity())

        self.output_proj = nn.Sequential(
            nn.LayerNorm(channels[-1]),
            nn.Linear(channels[-1], 2)
        )
        
    def forward(self, x, time, condition, mask=None):
        if time.ndim != 1:
            time = time.squeeze(-1)
        time_emb = self.time_embedder(time)
        time_emb = self.time_mlp(time_emb)

        h = self.input_proj(x)

        cond = self.condition_proj(condition)

        skip_connections = []
        for i, (block, self_attn, cross_attn) in enumerate(zip(self.down_blocks, self.self_attn_down, self.cross_attn_down)):
            h, skip = block(h, time_emb)
            skip_connections.append(skip)
            if isinstance(self_attn, CrossAttentionBlock):
                h = self_attn(h, context=None, mask=mask)
            else:
                h = self_attn(h)
            if isinstance(cross_attn, CrossAttentionBlock):
                h = cross_attn(h, cond, mask)
            else:
                h = cross_attn(h)

        h = self.mid_block1(h, time_emb)
        h = self.mid_self_attn(h, context=None, mask=mask)
        h = self.mid_cross_attn(h, cond, mask)
        h = self.mid_block2(h, time_emb)

        for block, self_attn, cross_attn in zip(self.up_blocks, self.self_attn_up, self.cross_attn_up):
            skip = skip_connections.pop()
            h = block(h, skip, time_emb)
            if isinstance(self_attn, CrossAttentionBlock):
                h = self_attn(h, context=None, mask=mask)
            else:
                h = self_attn(h)
            if isinstance(cross_attn, CrossAttentionBlock):
                h = cross_attn(h, cond, mask)
            else:
                h = cross_attn(h)

        return self.output_proj(h)
