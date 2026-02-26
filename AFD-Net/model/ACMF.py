import torch
import torch.nn as nn

class ModalityAttentionLayer(nn.Module):

    def __init__(self, num_modalities=4, num_tasks=3, use_task_specific=True):

        super().__init__()
        self.num_modalities = num_modalities
        self.num_tasks = num_tasks
        self.use_task_specific = use_task_specific

        if use_task_specific:

            self.task_modal_attention = nn.ModuleList([
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(num_modalities, num_modalities * 4, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(num_modalities * 4, num_modalities, 1),
                    nn.Sigmoid()
                )
                for _ in range(num_tasks)
            ])

        self.global_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_modalities, num_modalities * 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_modalities * 4, num_modalities, 1),
            nn.Sigmoid()
        )

        self.noise_detector = NoiseDetectionBranch(num_modalities)

    def forward(self, x, task_idx=None, enable_noise_suppress=True):

        if self.use_task_specific and task_idx is not None:
            weights = self.task_modal_attention[task_idx](x)  # [B, 4, 1, 1]
        else:
            weights = self.global_attention(x)

        if enable_noise_suppress:
            noise_scores = self.noise_detector(x)  # [B, 4, 1, 1]

            weights = weights * (1.0 - noise_scores * 0.2)

        weighted_x = x * weights

        return weighted_x, weights


class NoiseDetectionBranch(nn.Module):

    def __init__(self, num_modalities=4):
        super().__init__()

        self.noise_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_modalities, num_modalities * 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_modalities * 2, num_modalities, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        B, C, H, W = x.shape
        variance = torch.var(x.view(B, C, -1), dim=2, keepdim=True).unsqueeze(-1)  # [B, 4, 1, 1]

        noise_scores = self.noise_encoder(x)  # [B, 4, 1, 1]

        variance_normalized = torch.clamp(variance / (variance.mean() + 1e-6), 0, 2)
        variance_penalty = torch.abs(variance_normalized - 1.0)  # 偏离1越大越可能是噪声

        noise_scores = (noise_scores + variance_penalty * 0.3) / 1.3

        return noise_scores


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
            nn.Conv2d(total_channels // 2, in_chans, 1),  # 输出每个模态的交互权重
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
        assert C == self.in_chans, f"Expected {self.in_chans} input channels, got {C}"

        modalities = torch.split(x, 1, dim=1)  # 4个 [B, 1, H, W]

        modal_feats = []
        for i, (branch, modality) in enumerate(zip(self.modal_branches, modalities)):
            feat = branch(modality)  # [B, branch_channels, H, W]
            modal_feats.append(feat)

        concat_feats = torch.cat(modal_feats, dim=1)  # [B, total_channels, H, W]

        interaction_weights = self.cross_modal_attention(concat_feats)  # [B, 4, H, W]

        weighted_feats = []
        for i, feat in enumerate(modal_feats):
            weight = interaction_weights[:, i:i + 1]  # [B, 1, H, W]
            weighted_feats.append(feat * weight)

        weighted_concat = torch.cat(weighted_feats, dim=1)

        spatial_weight = self.spatial_attention(weighted_concat)
        weighted_concat = weighted_concat * spatial_weight

        fused = self.fusion(weighted_concat)

        if return_interaction_weights:
            return fused, interaction_weights
        return fused


class EnhancedModalityFusion(nn.Module):

    def __init__(self, in_chans=4, out_chans=96, num_tasks=3, use_task_specific=True):
        super().__init__()

        self.modality_attention = ModalityAttentionLayer(
            num_modalities=in_chans,
            num_tasks=num_tasks,
            use_task_specific=use_task_specific
        )

        self.cross_modal_interaction = CrossModalInteraction(
            in_chans=in_chans,
            out_chans=out_chans
        )

    def forward(self, x, task_idx=None, return_weights=False):
        weighted_x, modal_weights = self.modality_attention(
            x,
            task_idx=task_idx,
            enable_noise_suppress=True
        )

        fused, interaction_weights = self.cross_modal_interaction(
            weighted_x,
            return_interaction_weights=True
        )

        if return_weights:
            weights_dict = {
                'modal_weights': modal_weights,  # [B, 4, 1, 1]
                'interaction_weights': interaction_weights  # [B, 4, H, W]
            }
            return fused, weights_dict

        return fused



