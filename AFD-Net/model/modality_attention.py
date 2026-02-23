
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityAttentionLayer(nn.Module):
    def __init__(self, num_modalities=4):
        super().__init__()
        self.num_modalities = num_modalities
        self.residual_floor = 0.1

        self.global_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_modalities, num_modalities * 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_modalities * 4, num_modalities, 1),
            nn.Sigmoid()
        )
        
        self.noise_detector = NoiseDetectionBranch(num_modalities)
        
    def forward(self, x, enable_noise_suppress=True):
        weights = self.global_attention(x)
        
        if enable_noise_suppress:
            noise_scores = self.noise_detector(x)
            weights = weights * (1.0 - noise_scores * 0.0)
        
        weights = self.residual_floor + (1.0 - self.residual_floor) * weights
        weighted_x = x * weights
        return weighted_x, weights


class NoiseDetectionBranch(nn.Module):
    def __init__(self, num_modalities=4):
        super().__init__()
        self.num_modalities = num_modalities
        embed_dim = 16
        self.embed = nn.Sequential(
            nn.Conv2d(1, embed_dim, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.affine_a = nn.Parameter(torch.ones(num_modalities))
        self.affine_b = nn.Parameter(torch.zeros(num_modalities))
        self.low_var_weight = 0.0
        self.rel_eps = 1e-6
        
    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.num_modalities

        modalities = torch.split(x, 1, dim=1)
        embeds = [self.embed(m).flatten(1) for m in modalities]
        E = torch.stack(embeds, dim=1)
        E = F.normalize(E, p=2, dim=2)

        sim = torch.bmm(E, E.transpose(1, 2))
        sim_sum = sim.sum(dim=2) - 1.0
        sim_mean = sim_sum / max(1, (self.num_modalities - 1))

        sim_centered = sim_mean - sim_mean.mean(dim=1, keepdim=True)
        sim_std = sim_centered.std(dim=1, keepdim=True).clamp_min(self.rel_eps)
        sim_rel = sim_centered / sim_std

        a = self.affine_a.view(1, -1)
        b = self.affine_b.view(1, -1)
        reliability = torch.sigmoid(a * sim_rel + b)
        consistency_noise = 1.0 - reliability

        variance = torch.var(x.view(B, C, -1), dim=2)
        var_norm = variance / (variance.mean(dim=1, keepdim=True) + 1e-6)
        low_var_penalty = F.relu(1.0 - var_norm)

        noise_scores = torch.clamp(consistency_noise + self.low_var_weight * low_var_penalty, 0.0, 1.0)
        return noise_scores.view(B, C, 1, 1)


class CrossModalInteraction(nn.Module):
    def __init__(self, in_chans=4, out_chans=96, modality_names=None):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.modality_names = modality_names or ['T1', 'T2', 'T1ce', 'FLAIR']
        branch_channels = out_chans // in_chans
        self.modal_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, branch_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(branch_channels, branch_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
            )
            for _ in range(in_chans)
        ])
        
        total_channels = branch_channels * in_chans
        self.cross_modal_attention = nn.Sequential(
            nn.Conv2d(total_channels, total_channels // 2, 1, bias=False),
            nn.BatchNorm2d(total_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(total_channels // 2, in_chans, 1),
            nn.Sigmoid()
        )
        
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(total_channels, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
        self.fusion = nn.Sequential(
            nn.Conv2d(total_channels, out_chans, 1, bias=False),
            nn.BatchNorm2d(out_chans),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, return_interaction_weights=False):
        B, C, H, W = x.shape
        assert C == self.in_chans
        modalities = torch.split(x, 1, dim=1)
        modal_feats = [branch(modality) for branch, modality in zip(self.modal_branches, modalities)]
        concat_feats = torch.cat(modal_feats, dim=1)
        interaction_weights = self.cross_modal_attention(concat_feats)
        weighted_feats = [feat * interaction_weights[:, i:i+1] for i, feat in enumerate(modal_feats)]
        weighted_concat = torch.cat(weighted_feats, dim=1)
        spatial_weight = self.spatial_attention(weighted_concat)
        weighted_concat = weighted_concat * spatial_weight
        fused = self.fusion(weighted_concat)
        if return_interaction_weights:
            return fused, interaction_weights
        return fused


class EnhancedModalityFusion(nn.Module):
    def __init__(self, in_chans=4, out_chans=96):
        super().__init__()
        self.modality_attention = ModalityAttentionLayer(
            num_modalities=in_chans
        )
        self.cross_modal_interaction = CrossModalInteraction(
            in_chans=in_chans,
            out_chans=out_chans
        )
        
    def forward(self, x, return_weights=False):
        weighted_x, modal_weights = self.modality_attention(x, enable_noise_suppress=True)
        
        fused, interaction_weights = self.cross_modal_interaction(
            weighted_x, return_interaction_weights=True
        )
        
        if return_weights:
            return fused, {
                'modal_weights': modal_weights,
                'interaction_weights': interaction_weights,
            }
        return fused
