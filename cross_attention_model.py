import math
import torch
from torch import nn


class SymmetricCrossAttention(nn.Module):
    def __init__(self, dim1=768, dim2=600, hidden=64):
        super().__init__()
        self.enc1 = nn.Linear(dim1, hidden)
        self.enc2 = nn.Linear(dim2, hidden)
        self.scale = math.sqrt(hidden)

        self.q1 = nn.Linear(hidden, hidden)
        self.k1 = nn.Linear(hidden, hidden)
        self.v1 = nn.Linear(hidden, hidden)
        self.q2 = nn.Linear(hidden, hidden)
        self.k2 = nn.Linear(hidden, hidden)
        self.v2 = nn.Linear(hidden, hidden)

        self.regressor = nn.Sequential(
            nn.Linear(hidden * 2, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 36)
        )

    def forward(self, f1, f2):
        a = self.enc1(f1.squeeze(1)).unsqueeze(1)
        v = self.enc2(f2.squeeze(1)).unsqueeze(1)

        attn_a2v = torch.bmm(self.q1(a), self.k2(v).transpose(1,2)) / self.scale
        attn_v2a = torch.bmm(self.q2(v), self.k1(a).transpose(1,2)) / self.scale

        a2v_out = torch.bmm(torch.softmax(attn_a2v, -1), self.v2(v))
        v2a_out = torch.bmm(torch.softmax(attn_v2a, -1), self.v1(a))

        fused = torch.cat([a2v_out.squeeze(1), v2a_out.squeeze(1)], dim=-1)
        return self.regressor(fused)


class GatedMultimodalUnits(nn.Module):
    def __init__(self, dim1=768, dim2=600, hidden=64):
        super().__init__()
        self.enc1 = nn.Linear(dim1, hidden)
        self.enc2 = nn.Linear(dim2, hidden)
        self.gate = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.Sigmoid()
        )
        self.out = nn.Sequential(
            nn.Linear(hidden, 128), nn.ReLU(), nn.Linear(128, 36)
        )

    def forward(self, f1, f2):
        a = self.enc1(f1.squeeze(1))
        v = self.enc2(f2.squeeze(1))
        concat = torch.cat([a, v], dim=-1)
        g = self.gate(concat)
        fused = g * a + (1 - g) * v
        return self.out(fused)


class MHCrossAttention(nn.Module):
    def __init__(self, dim1=768, dim2=600, hidden=64, heads=4):
        super().__init__()
        self.enc1 = nn.Linear(dim1, hidden)
        self.enc2 = nn.Linear(dim2, hidden)
        self.multihead = nn.MultiheadAttention(embed_dim=hidden, num_heads=heads, batch_first=True)
        self.regressor = nn.Sequential(
            nn.Linear(hidden * 2, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 36)
        )

    def forward(self, f1, f2):
        a = self.enc1(f1)
        v = self.enc2(f2)
        a2v, _ = self.multihead(a, v, v)
        v2a, _ = self.multihead(v, a, a)
        fused = torch.cat([a2v.squeeze(1), v2a.squeeze(1)], dim=-1)
        return self.regressor(fused)


class BilinearAttentionFusion(nn.Module):
    def __init__(self, d_v: int, d_q: int, d_m: int, m: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        
        # Per-head projections for attention map
        self.W_v_att = nn.ModuleList([nn.Linear(d_v, d_m, bias=False) for _ in range(num_heads)])
        self.W_q_att = nn.ModuleList([nn.Linear(d_q, d_m, bias=False) for _ in range(num_heads)])
        
        # Per-head projections for low-rank bilinear fusion
        self.W_v_heads = nn.ModuleList([nn.Linear(d_v, m, bias=False) for _ in range(num_heads)])
        self.W_q_heads = nn.ModuleList([nn.Linear(d_q, m, bias=False) for _ in range(num_heads)])

    def forward(self, v: torch.Tensor, q: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            v: Refined inertial features (batch_size, N_v=1, d_v=768)
            q: Refined video features (batch_size, N_q=1, d_q=600)
        
        Returns:
            fused: Fused features per head concatenated (batch_size, num_heads)
            attention_maps: List of attention maps per head, each (batch_size, N_v, N_q)
        """
        batch_size = v.size(0)
        N_v = v.size(1)
        N_q = q.size(1)
        
        fused_heads = []
        attention_maps = []
        
        for h in range(self.num_heads):
            # Compute projections for attention
            proj_v = self.W_v_att[h](v)  # (batch_size, N_v, d_m)
            proj_q = self.W_q_att[h](q)  # (batch_size, N_q, d_m)
            
            # Compute scores as dot product in projected space
            scores = torch.bmm(proj_v, proj_q.permute(0, 2, 1))  # (batch_size, N_v, N_q)
            
            # Flatten and softmax over all elements for joint attention
            flat_scores = scores.view(batch_size, -1)  # (batch_size, N_v * N_q)
            A_flat = F.softmax(flat_scores, dim=-1)
            A = A_flat.view(batch_size, N_v, N_q)
            attention_maps.append(A)
            
            # Compute projections for fusion
            proj_v_fus = self.W_v_heads[h](v)  # (batch_size, N_v, m)
            proj_q_fus = self.W_q_heads[h](q)  # (batch_size, N_q, m)
            
            # Weighted sum over N_q using A
            weighted_proj_q = torch.bmm(A, proj_q_fus)  # (batch_size, N_v, m)
            
            # Compute dot products and sum over N_v and m (equivalent to sum_j (proj_v_fus_j ^T weighted_proj_q_j))
            f_h = (proj_v_fus * weighted_proj_q).sum(dim=(1, 2))  # (batch_size,)
            
            fused_heads.append(f_h.unsqueeze(1))
        
        fused = torch.cat(fused_heads, dim=1)  # (batch_size, num_heads)
        
        return fused, attention_maps

class BiLinearClassificationModel(nn.Module):
    def __init__(self, d_inertial: int = 768, d_video: int = 600, d_m: int = 128, m: int = 64, num_heads: int = 8, num_classes: int = 10):
        super().__init__()
        self.fusion = BilinearAttentionFusion(d_inertial, d_video, d_m, m, num_heads)
        self.classifier = nn.Linear(num_heads, num_classes)
    
    def forward(self, inertial: torch.Tensor, video: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        fused, att_maps = self.fusion(inertial, video)
        logits = self.classifier(fused)
        return logits, att_maps

def orthogonality_loss(att_maps: list[torch.Tensor], alpha: float = 0.1) -> torch.Tensor:
    batch_size = att_maps[0].size(0)
    num_heads = len(att_maps)
    S = att_maps[0].numel() // batch_size  # N_v * N_q
    
    # Stack and flatten attention maps
    att_stack = torch.stack(att_maps, dim=1)  # (batch_size, num_heads, N_v, N_q)
    flat_att = att_stack.view(batch_size, num_heads, S)  # (batch_size, num_heads, S)
    
    # Compute pairwise inner products
    sim = torch.bmm(flat_att, flat_att.permute(0, 2, 1))  # (batch_size, num_heads, num_heads)
    
    # Mask diagonal to exclude self-similarities
    eye = torch.eye(num_heads, device=sim.device).unsqueeze(0)  # (1, num_heads, num_heads)
    off_diag_mask = 1.0 - eye
    
    # Compute sum of squared inner products for off-diagonal elements
    perp_loss = (sim ** 2 * off_diag_mask).sum() / batch_size
    
    return alpha * perp_loss


class GatedInteractiveAttention(nn.Module):
    """
    Args:
        d_v: feature dim of v (e.g. 768)
        d_q: feature dim of q (e.g. 600)
        d_m: internal attention dimension (d in paper)
        num_heads: number of independent heads
    """
    def __init__(self, d_v: int, d_q: int, d_m: int, num_heads: int, num_classes: int):
        super().__init__()
        self.num_heads = num_heads
        self.d_m = d_m

        # Per-head projections for v->q attention: Q_v, K_q, V_q
        self.W_Q_v = nn.ModuleList([nn.Linear(d_v, d_m, bias=False) for _ in range(num_heads)])
        self.W_K_q = nn.ModuleList([nn.Linear(d_q, d_m, bias=False) for _ in range(num_heads)])
        self.W_V_q = nn.ModuleList([nn.Linear(d_q, d_m, bias=False) for _ in range(num_heads)])

        # Per-head projections for q->v attention: Q_q, K_v, V_v
        self.W_Q_q = nn.ModuleList([nn.Linear(d_q, d_m, bias=False) for _ in range(num_heads)])
        self.W_K_v = nn.ModuleList([nn.Linear(d_v, d_m, bias=False) for _ in range(num_heads)])
        self.W_V_v = nn.ModuleList([nn.Linear(d_v, d_m, bias=False) for _ in range(num_heads)])

        # Per-head gating networks and layernorms (applied to attended outputs)
        self.W_g_v = nn.ModuleList([nn.Linear(d_m, d_m, bias=True) for _ in range(num_heads)])  # for gating v (v<-q)
        self.W_g_q = nn.ModuleList([nn.Linear(d_m, d_m, bias=True) for _ in range(num_heads)])  # for gating q (q<-v)
        self.ln_v = nn.ModuleList([nn.LayerNorm(d_m) for _ in range(num_heads)])
        self.ln_q = nn.ModuleList([nn.LayerNorm(d_m) for _ in range(num_heads)])

        # Self projection (skip) to d_m space for gating residual
        self.W_self_v = nn.ModuleList([nn.Linear(d_v, d_m, bias=False) for _ in range(num_heads)])
        self.W_self_q = nn.ModuleList([nn.Linear(d_q, d_m, bias=False) for _ in range(num_heads)])
        
        self.classifier = nn.Sequential(
            nn.Linear(num_heads * d_m * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        # small epsilon for numerical stability if needed
        self.eps = 1e-8

    def forward(self, v: torch.Tensor, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list]:
        """
        Args:
            v: Refined inertial features (batch_size, N_v, d_v)
            q: Refined video features    (batch_size, N_q, d_q)

        Returns:
            H_v_gia: (batch_size, N_v, num_heads * d_m)  -- gated v representations (v <- q)
            H_q_gia: (batch_size, N_q, num_heads * d_m)  -- gated q representations (q <- v)
            attention_maps: list of dict per head with keys 'v2q' and 'q2v'
        """
        batch_size = v.size(0)
        N_v = v.size(1)
        N_q = q.size(1)

        heads_v = []
        heads_q = []
        attention_maps = []

        scale = 1.0 / math.sqrt(self.d_m)

        for h in range(self.num_heads):
            # ---------- v <- q direction ----------
            Q_v = self.W_Q_v[h](v)        # (B, N_v, d_m)
            K_q = self.W_K_q[h](q)        # (B, N_q, d_m)
            V_q = self.W_V_q[h](q)        # (B, N_q, d_m)

            # scores: (B, N_v, N_q)
            scores_v2q = torch.bmm(Q_v, K_q.transpose(1, 2)) * scale
            A_v2q = F.softmax(scores_v2q, dim=-1)  # softmax over keys (N_q)
            # attended representation (B, N_v, d_m)
            H_v2q = torch.bmm(A_v2q, V_q)

            # gating for v: G = sigmoid(W_g(LayerNorm(H_v2q)))
            H_v2q_ln = self.ln_v[h](H_v2q)
            G_v = torch.sigmoid(self.W_g_v[h](H_v2q_ln))  # (B, N_v, d_m)

            # skip / residual term mapped to d_m
            V_self = self.W_self_v[h](v)  # (B, N_v, d_m)

            # final gated v representation for this head
            H_v_gia_h = G_v * H_v2q + (1.0 - G_v) * V_self  # (B, N_v, d_m)

            # ---------- q <- v direction ----------
            Q_q = self.W_Q_q[h](q)       # (B, N_q, d_m)
            K_v = self.W_K_v[h](v)       # (B, N_v, d_m)
            V_v = self.W_V_v[h](v)       # (B, N_v, d_m)

            scores_q2v = torch.bmm(Q_q, K_v.transpose(1, 2)) * scale  # (B, N_q, N_v)
            A_q2v = F.softmax(scores_q2v, dim=-1)  # softmax over keys (N_v)
            H_q2v = torch.bmm(A_q2v, V_v)  # (B, N_q, d_m)

            H_q2v_ln = self.ln_q[h](H_q2v)
            G_q = torch.sigmoid(self.W_g_q[h](H_q2v_ln))  # (B, N_q, d_m)
            Q_self = self.W_self_q[h](q)  # (B, N_q, d_m)
            H_q_gia_h = G_q * H_q2v + (1.0 - G_q) * Q_self  # (B, N_q, d_m)

            # collect
            heads_v.append(H_v_gia_h)  # list of (B, N_v, d_m)
            heads_q.append(H_q_gia_h)  # list of (B, N_q, d_m)

            attention_maps.append({
                'v2q': A_v2q,   # (B, N_v, N_q)
                'q2v': A_q2v    # (B, N_q, N_v)
            })

        # Concatenate heads on feature dim
        H_v_gia = torch.cat(heads_v, dim=-1)  # (B, N_v, num_heads * d_m)
        H_q_gia = torch.cat(heads_q, dim=-1)  # (B, N_q, num_heads * d_m)
        
        H_v_pooled = H_v_gia.squeeze(1)  # (B, num_heads * d_m)
        H_q_pooled = H_q_gia.squeeze(1)  # (B, num_heads * d_m)
        # Concatenate
        fused = torch.cat([H_v_pooled, H_q_pooled], dim=-1)
        logits = self.classifier(fused)
        return logits, attention_maps
