import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False, dropout=0.1):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, mask=None):
        Q_proj = self.fc_q(Q)
        K_proj, V_proj = self.fc_k(K), self.fc_v(K)
        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q_proj.split(dim_split, 2), 0)
        K_ = torch.cat(K_proj.split(dim_split, 2), 0)
        V_ = torch.cat(V_proj.split(dim_split, 2), 0)

        attn_scores = Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V)
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).repeat_interleave(self.num_heads, dim=0)
            attn_scores = attn_scores.masked_fill(~mask_expanded, -1e9)

        A = self.dropout(torch.softmax(attn_scores, 2))
        O = A.bmm(V_)
        O = torch.cat(O.split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False, dropout=0.1):
        super().__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln, dropout=dropout)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln, dropout=dropout)

    def forward(self, X, mask=None):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X, mask=mask)
        return self.mab1(X, H, mask=None)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False, dropout=0.1):
        super().__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln, dropout=dropout)

    def forward(self, X, mask=None):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X, mask=mask)

class SetTransformerEncoder(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=64, output_dim=256, num_heads=4, num_inds=32, num_seeds=1, ln=True, dropout=0.1):
        super().__init__()
        self.element_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.isab1 = ISAB(hidden_dim, hidden_dim, num_heads, num_inds, ln=ln, dropout=dropout)
        self.isab2 = ISAB(hidden_dim, hidden_dim, num_heads, num_inds, ln=ln, dropout=dropout)
        self.pma = PMA(hidden_dim, num_heads, num_seeds, ln=ln, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.element_encoder(x)
        x = self.isab1(x, mask=mask)
        x = self.isab2(x, mask=mask)
        x = self.pma(x, mask=mask)
        return self.dropout(x.flatten(1))