import torch
from einops import rearrange
from inspect import isfunction
import numpy as np


def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d

class LearnableSigmoid(torch.nn.Module):
    def __init__(self):
        super(LearnableSigmoid, self).__init__()
        self.weight = torch.nn.Parameter(torch.ones(1, 1, 1), requires_grad=True)

    def forward(self, input):
        return (1 + torch.exp(self.weight)) / (1 + torch.exp(self.weight - input))

# Default Attenton
class Attention(torch.nn.Module):
    def __init__(
            self, dim, num_heads,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.,
            proj_drop=0.,
            patch_group=7):
        super(Attention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.group_num = patch_group
        self.qkv_bias = qkv_bias
        self.head_dim = head_dim

        self.to_q = torch.nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = torch.nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = torch.nn.Linear(dim, dim, bias=qkv_bias)
        
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.proj = torch.nn.Linear(dim, dim)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        
    def qkv_cal(self, q, k, v, mask=None):
        # [B, P, D]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v))
        # [B, H, P, d]

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        if mask is not None:
            dots = dots + mask
        attn = dots.softmax(dim=-1)  # [B, H, P_q, P_kv]
        attn = self.attn_drop(attn)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)  # [B, H, P_q, d]
        out = rearrange(out, 'b h n d -> b n (h d)')  # [B, P_q, D]
        return out

    def forward(self, x, context=None, mask=None):
        b, n, _ = x.shape
        kv_input = default(context, x)
        q_input = x

        q = self.to_q(q_input)
        k = self.to_k(kv_input)
        v = self.to_v(kv_input)
        # [B, P, D]

        out = self.qkv_cal(q, k, v, mask)
        # [B, P, D]

        out = self.proj(out)
        out = self.proj_drop(out)
        return out

def covariance(X):
    B, H, N, D = X.shape
    mean = torch.mean(X, dim=-1).unsqueeze(-1)
    X = (X - mean).unsqueeze(-1)
    return 1 / (D - 1) * X @ X.transpose(-1, -2)

def dist_covariance(x1, x2):
    B, H, N, _, _ = x1.shape
    attn_spd = torch.cdist(x1.view(B*H,N,-1), x2.view(B*H,N,-1))
    return torch.reshape(attn_spd, (B,H,N,N))

def grassmannian_point(Om, I):
    Sm = 0.25 * (Om - Om.transpose(-1,-2))
    I = I.to(Sm.device)
    return (I + Sm) @ torch.linalg.inv(I - Sm)

def dist_grassmann(x1, x2):
    B, H, N, C, _ = x1.shape
    I = torch.tile(torch.eye(C), (B,H,N,1,1)) 
    g1 = grassmannian_point(x1, I)
    g2 = grassmannian_point(x2, I)
    g1r = g1 @ g1.transpose(-1,-2)
    g2r = g2 @ g2.transpose(-1,-2)
    attn_gm = torch.cdist(g1r.view(B*H,N,-1), g2r.view(B*H,N,-1))
    return torch.reshape(attn_gm, (B,H,N,N))

class DGMAtention(Attention):
    def __init__(
            self, *args, **kwargs):
        super(DGMAtention, self).__init__(*args, **kwargs)
        self.IFS = torch.nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1, groups=self.dim)
        self.cov_size = 12
        self.s_scale = self.cov_size * 1.
        self.conv_proj = torch.nn.Linear(self.head_dim, self.cov_size)
        self.layer_norm = torch.nn.LayerNorm([196, 196])
        self.conv_2d_attn = torch.nn.Sequential(
            torch.nn.InstanceNorm2d(2 * self.num_heads),
            torch.nn.Conv2d(in_channels=2 * self.num_heads, out_channels=self.num_heads, kernel_size=(1, 1))
        )
        self.conv_3d_attn = torch.nn.Sequential(
            torch.nn.InstanceNorm3d(2 * self.group_num * self.group_num),
            torch.nn.Conv3d(in_channels=2 * self.group_num * self.group_num, out_channels=self.group_num * self.group_num, kernel_size=(1, 1, 1))
        )

        if torch.distributed.get_rank() == 0:
            print('Attention type: DGMA')

    def InterViewSignature(self, x):
        B, H, N, C = x.shape
        hight = weight = round(np.power(N, float(1) / float(2)))
        x = rearrange(x, 'b h (h1 w1) c -> b (h c) h1 w1', h1=hight, w1=weight)

        ifs = self.IFS(x)
        ifs = rearrange(ifs, 'b (h c) h1 w1 -> b h (h1 w1) c', h=H, c=C)

        return ifs

    def forward(self, x, view_num=3, context=None, mask=None, dgma=True):
        b, n, _ = x.shape  
        kv_input = default(context, x)
        q_input = x

        q = self.to_q(q_input)
        k = self.to_k(kv_input)
        v = self.to_v(kv_input)

        # [B, P, D]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v))
        # [B, H, P, d]
        proj_q = self.conv_proj(q)
        proj_k = self.conv_proj(k)
        cov_q = covariance(proj_q)
        cov_k = covariance(proj_k)

        if dgma is True:
            b, _, _, _ = q.shape
            b_s = int(b // view_num)
            token_num = 14 // self.group_num
            attn_s = dist_covariance(cov_q, cov_k) * self.s_scale
            attn_e = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
            attn_e = self.layer_norm(attn_e)
            attn_s = self.layer_norm(attn_s)
            attn_group = self.conv_2d_attn(torch.cat((attn_e, attn_s), dim=1))
            att_diag = torch.diagonal(attn_group, 0, dim1=-2, dim2=-1)
            indices = torch.argsort(att_diag, dim=2)
            [B, H, N, D] = q.shape
            indices = indices.unsqueeze(-1).expand(B, H, N, D)
            q_s = torch.gather(q, 2, indices)
            k_s = torch.gather(k, 2, indices)
            v_s = torch.gather(v, 2, indices)
            q_cls, k_cls, v_cls = map(lambda t: rearrange(t, '(b v) h (t1 s1 t2 s2) d -> b (s1 s2) h (t1 t2 v) d',
                                                          b=b_s, v=view_num, t1=token_num, s1=self.group_num, t2=token_num,
                                                          s2=self.group_num), (q_s, k_s, v_s))

            qC_cls, kC_cls = map(lambda t: rearrange(t, '(b v) h (t1 s1 t2 s2) d1 d2 -> b (s1 s2) h (t1 t2 v) (d1 d2)',
                                                          b=b_s, v=view_num, t1=token_num, s1=self.group_num, t2=token_num,
                                                          s2=self.group_num), (cov_q, cov_k))

            dots = torch.einsum('b n h i d, b n h j d -> b n h i j', q_cls, k_cls) * self.scale
            dots_cov = torch.einsum('b n h i d, b n h j d -> b n h i j', qC_cls, kC_cls) * self.s_scale
            dots = self.conv_3d_attn(torch.cat((dots, dots_cov), dim=1))
            if mask is not None:
                dots = dots + mask

            attn = dots.softmax(dim=-1)  # [B, H, P_q, P_kv]
            attn = self.attn_drop(attn)
            out = torch.einsum('b n h i j, b n h j d -> b n h i d', attn, v_cls)  # [B, H, P_q, d]

            out = rearrange(out, 'b (s1 s2) h (t1 t2 v) d -> (b v) h (t1 s1 t2 s2) d', v=view_num, t1=token_num,
                            s1=self.group_num, t2=token_num, s2=self.group_num)
            ifs = self.InterViewSignature(v)
            out = out + ifs
        else:
            dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
            if mask is not None:
                dots = dots + mask
            attn = dots.softmax(dim=-1)  # [B, H, P_q, P_kv]
            attn = self.attn_drop(attn)
            out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)  # [B, H, P_q, d]
        out = rearrange(out, 'b h n d -> b n (h d)')  # [B, P_q, D]

        out = self.proj(out)
        out = self.proj_drop(out)
        return out

# STM block
class STMAttention(Attention):
    def forward(self, q_input, kv_input, token_score):
        # q [B, P_q, D]; kv [B, P_kv, D]; token_score [B, P_kv, 1]
        b, n, _ = q_input.shape
        
        q = self.to_q(q_input)
        k = self.to_k(kv_input)
        v = self.to_v(kv_input)
        # [B, P, D]

        # qkv_cal
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v))
        # [B, H, P, d]

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale # b,12(h),196,196*v
        token_score = token_score.squeeze(-1)[:, None, None, :]  # [B, 1, 1, P_kv] # b,1,1,196*v
        attn = (dots + token_score).softmax(dim=-1)  # [B, H, P_q, P_kv]
        attn = self.attn_drop(attn)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)  # [B, H, P_q, d]
        out = rearrange(out, 'b h n d -> b n (h d)')  # [B, P_q, D]

        out = self.proj(out)
        out = self.proj_drop(out)
        return out
