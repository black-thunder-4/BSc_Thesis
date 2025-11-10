import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CNN(nn.Module):
    def __init__(self, channels=[640, 64, 32, 8, 3], kernel_sizes=[5, 5, 5, 5]):
        super().__init__()

        layers = []
        for i in range(len(channels) - 1):
            layers.append(nn.Conv1d(
                in_channels=channels[i],
                out_channels=channels[i+1],
                kernel_size=kernel_sizes[i],
                padding="same"
            ))

        self.cnns = nn.ModuleList(layers)

    def forward(self, x, mask):
        # (B, L, C) -> (B, C, L)
        x = x.permute(0, 2, 1)

        for conv in self.cnns:
            x = conv(x)
            if conv != self.cnns[-1]:
                x = F.relu(x)

        # back to (B, L, C)
        x = x.permute(0, 2, 1)

        # apply mask: (B, L) -> (B, L, 1)
        return x * mask.unsqueeze(-1)



class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=5):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, padding="same")
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class UNet1D(nn.Module):
    def __init__(self, in_ch=640, base=128, depth=4):
        super().__init__()

        # DOWN
        self.down_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()

        ch = in_ch
        for i in range(depth):
            out = base * (2 ** i)
            self.down_blocks.append(ConvBlock(ch, out))
            self.pools.append(nn.AvgPool1d(kernel_size=2))
            ch = out

        # BOTTLENECK
        self.bottleneck = ConvBlock(ch, ch * 2)
        ch = ch * 2

        # UP
        self.up_transpose = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        for i in reversed(range(depth)):
            skip_ch = base * (2 ** i)
            self.up_transpose.append(nn.ConvTranspose1d(ch, skip_ch, kernel_size=2, stride=2))
            self.up_blocks.append(ConvBlock(skip_ch * 2, skip_ch))
            ch = skip_ch

        # OUTPUT (predict x,y,z per residue)
        self.out = nn.Conv1d(ch, 3, kernel_size=1)

    def forward(self, x, mask):
        """
        x: (B, L, C)
        mask: (B, L)
        """
        mask = mask.unsqueeze(1).float()  # convert to float here
        x = x.permute(0, 2, 1)            # (B,C,L)
    
        skips = []
    
        # DOWN
        for block, pool in zip(self.down_blocks, self.pools):
            x = block(x)
            x = x * mask                 # apply mask
            skips.append(x)
    
            x = pool(x)
            mask = F.avg_pool1d(mask, kernel_size=2)
    
        # BOTTLENECK
        x = self.bottleneck(x) * mask
    
        # UP
        for up_t, block, skip in zip(self.up_transpose, self.up_blocks, reversed(skips)):
            x = up_t(x)
    
            # match lengths if needed
            if x.shape[-1] != skip.shape[-1]:
                x = F.pad(x, (0, skip.shape[-1] - x.shape[-1]))
    
            # Upsampled mask for this resolution
            mask_up = (mask > 0).float()
            mask_up = F.interpolate(mask_up, size=skip.shape[-1], mode="nearest")
    
            x = torch.cat([x, skip], dim=1)
            x = block(x) * mask_up
            mask = mask_up
    
        x = self.out(x)                  # (B,3,L)
        return x.permute(0, 2, 1)        # (B,L,3)


def make_pair_mask(mask):
    # mask: (B, L) bool → (B, L, L) bool
    return mask.unsqueeze(1) & mask.unsqueeze(2)


class PairwiseProject(nn.Module):
    """
    Build a simple symmetric pair feature tensor P[i,j] from residue features X[i].
    """
    def __init__(self, d_x=256, d_p=128):
        super().__init__()
        self.lin = nn.Linear(d_x, d_p)

    def forward(self, X):  # X: (B, L, d_x)
        P1 = self.lin(X)                     # (B, L, d_p)
        P = P1.unsqueeze(2) + P1.unsqueeze(1)  # (B, L, L, d_p), symmetric outer-sum
        return P


class TriangleMulUpdate(nn.Module):
    """
    AlphaFold-like triangle multiplicative update (simplified).
    Update P_ij using multiplicative messages along k: sum_k f(P_ik) * g(P_kj).
    """
    def __init__(self, d_p=128, hidden=None, dropout=0.0):
        super().__init__()
        h = hidden or d_p
        self.left  = nn.Linear(d_p, h, bias=False)
        self.right = nn.Linear(d_p, h, bias=False)
        self.out   = nn.Linear(h, d_p, bias=False)
        self.norm  = nn.LayerNorm(d_p)
        self.drop  = nn.Dropout(dropout)

    def forward(self, P, pair_mask):
        # P: (B, L, L, d_p), pair_mask: (B, L, L) bool
        B, L, _, d = P.shape

        # Project to hidden
        Lh = self.left(P)    # (B, L, L, h)
        Rh = self.right(P)   # (B, L, L, h)

        # Mask along k when contracting: we’ll zero invalid k
        # Compose an (i,k) mask and (k,j) mask by broadcasting pair_mask
        m_ik = pair_mask.unsqueeze(-1).float()  # (B, L, L, 1)
        m_kj = pair_mask.transpose(1, 2).unsqueeze(-1).float()  # (B, L, L, 1)

        Lh = Lh * m_ik
        Rh = Rh * m_kj

        # Triangle mul: update_ijh = sum_k Lh_i k h * Rh_k j h
        # → einsum over k: (B,i,k,h) x (B,k,j,h) -> (B,i,j,h)
        upd = torch.einsum('bikh,bkjh->bijh', Lh, Rh)  # (B, L, L, h)

        upd = self.out(upd)                   # (B, L, L, d_p)
        upd = self.drop(upd)

        # Residual + norm (only where valid pair)
        P = P + upd * pair_mask.unsqueeze(-1).float()
        return self.norm(P)


class ResidueUpdate(nn.Module):
    """
    Aggregate pair info back to residues (global→local):
    msg_i = SUM_j phi(P_ij), masked and normalized.
    """
    def __init__(self, d_x=256, d_p=128, hidden=None, dropout=0.0):
        super().__init__()
        h = hidden or d_x
        self.pair_to_res = nn.Linear(d_p, d_x)
        self.mlp = nn.Sequential(
            nn.Linear(d_x, h),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h, d_x),
        )
        self.norm = nn.LayerNorm(d_x)

    def forward(self, X, P, mask, pair_mask):
        # X: (B, L, d_x), P: (B, L, L, d_p)
        # mask: (B, L), pair_mask: (B, L, L)
        msg = self.pair_to_res(P)                           # (B, L, L, d_x)
        msg = msg * pair_mask.unsqueeze(-1).float()         # mask invalid pairs
        denom = pair_mask.float().sum(dim=2, keepdim=True).clamp(min=1e-6)
        msg = msg.sum(dim=2) / denom                        # (B, L, d_x)

        X = X + self.mlp(msg)
        # keep padded positions stable
        X = torch.where(mask.unsqueeze(-1), X, torch.zeros_like(X))
        return self.norm(X)


class MiniFold(nn.Module):
    """
    ESM→(1280) → Linear(1280→d_x=256)
    Pairwise P (L×L×d_p) + Triangle updates
    Residue updates from pair → coordinates (L×3)
    """
    def __init__(self, d_in=1280, d_x=256, d_p=128, depth=4, dropout=0.1):
        super().__init__()
        self.config = dict(d_in=d_in, d_x=d_x, d_p=d_p, depth=depth, dropout=dropout)

        # Project ESM embeddings to working dim
        self.embed = nn.Linear(d_in, d_x)

        # Pairwise
        self.pair_proj = PairwiseProject(d_x=d_x, d_p=d_p)

        # Stacks of triangle + residue updates
        self.tri_blocks = nn.ModuleList([TriangleMulUpdate(d_p=d_p, dropout=dropout) for _ in range(depth)])
        self.res_blocks = nn.ModuleList([ResidueUpdate(d_x=d_x, d_p=d_p, dropout=dropout) for _ in range(depth)])

        # Coord head
        self.coord_head = nn.Sequential(
            nn.Linear(d_x, d_x),
            nn.ReLU(),
            nn.Linear(d_x, 3),
        )

    def forward(self, esm_embed, mask):
        """
        esm_embed: (B, L, 1280) from ESM-2
        mask:      (B, L)  bool
        returns:   coords (B, L, 3)
        """
        X = self.embed(esm_embed)                   # (B, L, d_x)

        pair_mask = make_pair_mask(mask)            # (B, L, L)
        P = self.pair_proj(X)                       # (B, L, L, d_p)

        for tri, res in zip(self.tri_blocks, self.res_blocks):
            P = tri(P, pair_mask)
            X = res(X, P, mask, pair_mask)

        C = self.coord_head(X) * mask.unsqueeze(-1)  # zero out padding
        return C