import torch.nn as nn
import torch.nn.functional as F
from .swin import SwinEncoder
from .modality_attention import EnhancedModalityFusion
from .fab_align import create_fab_align
from .DSGF import Self_Adaptive_Weighted_Fusion_Module


class SwinBackboneAdapter(nn.Module):
    def __init__(
        self,
        img_size=128,
        in_chans=4,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        drop_path_rate=0.2,
        use_checkpoint=False,
        use_modality_fusion=True,
        gag_residual_alpha=0.6,
        swfm_use_residual=True,
        swfm_residual_alpha=0.85,
        use_fpn=True,
        adaptive_window_size=True,
        use_fab_align=False,
        align_type='fab',
        fab_reduction=4,
        decoder_channels=None,
    ):
        super().__init__()
        
        self.use_modality_fusion = use_modality_fusion
        self.gag_residual_alpha = gag_residual_alpha
        self.swfm_use_residual = swfm_use_residual
        self.swfm_residual_alpha = swfm_residual_alpha
        self.use_fpn = use_fpn
        self.adaptive_window_size = adaptive_window_size
        self.use_fab_align = use_fab_align
        self.align_type = str(align_type).lower()
        self.fab_reduction = fab_reduction
        self.bottleneck_channels = 512

        if decoder_channels is None:
            decoder_channels = [384, 256, 128, 128]
        if not isinstance(decoder_channels, (list, tuple)) or len(decoder_channels) != 4:
            raise ValueError(
                f"decoder_channels必须是长度为4的list/tuple，但得到: {decoder_channels}"
            )
        self.decoder_channels = list(decoder_channels)

        if use_modality_fusion:
            self.modality_fusion = EnhancedModalityFusion(
                    in_chans=in_chans,
                    out_chans=embed_dim,
            )
            self.fusion_type = 'enhanced'
            swin_in_chans = embed_dim
        else:
            self.modality_fusion = None
            swin_in_chans = in_chans
            self.fusion_type = 'none'

        if adaptive_window_size:
            effective_window_size = 8
        else:
            effective_window_size = 4
        self.swin_encoder = SwinEncoder(
            img_size=img_size,
            patch_size=4,
            in_chans=swin_in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=effective_window_size,
            mlp_ratio=4.,
            qkv_bias=True,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=drop_path_rate,
            norm_layer=nn.LayerNorm,
            ape=False,
            patch_norm=True,
            use_checkpoint=use_checkpoint,
            high_level_idx=3,
            low_level_idx=0,
        )

        if use_fab_align:
            if self.align_type not in {'fab', 'se', 'eca'}:
                raise ValueError(f"align_type must be 'fab', 'se' or 'eca', but got: {align_type}")
            else:
                # FAB-Align：加性偏置校准
                print(f"✓ 使用 FAB-Align (reduction={fab_reduction}) 进行通道对齐")
                self.channel_align = nn.ModuleList([
                    create_fab_align(96, 128, reduction=fab_reduction),   # e1
                    create_fab_align(192, 128, reduction=fab_reduction),  # e2
                    create_fab_align(384, 256, reduction=fab_reduction),  # e3
                    create_fab_align(768, 384, reduction=fab_reduction),  # e4
                ])
        else:
            # 使用标准1x1卷积对齐
            self.channel_align = nn.ModuleList([
                nn.Conv2d(96, 128, kernel_size=1, bias=False),   # e1
                nn.Conv2d(192, 128, kernel_size=1, bias=False),  # e2
                nn.Conv2d(384, 256, kernel_size=1, bias=False),  # e3
                nn.Conv2d(768, 384, kernel_size=1, bias=False),  # e4
            ])

        if use_fpn:
            self.fpn_convs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(384, 256, kernel_size=1, bias=False),
                    nn.GroupNorm(32, 256),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(256, 128, kernel_size=1, bias=False),
                    nn.GroupNorm(16, 128),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(128, 128, kernel_size=1, bias=False),
                    nn.GroupNorm(16, 128),
                    nn.ReLU(inplace=True),
                ),
            ])
            self.fpn_refine = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                    nn.GroupNorm(32, 256),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
                    nn.GroupNorm(16, 128),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
                    nn.GroupNorm(16, 128),
                    nn.ReLU(inplace=True),
                ),
            ])
        else:
            self.fpn_convs = None
            self.fpn_refine = None

        self.bottleneck_proj = nn.Sequential(
            nn.Conv2d(384, 512, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, 512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, 512),
            nn.ReLU(inplace=True),
        )

        decoder_channels = self.decoder_channels
        
        # Dropout
        self.decoder_dropout = nn.Dropout2d(p=0.2)

        def _choose_gn_groups(num_channels: int, preferred: int) -> int:
            num_channels = int(num_channels)
            preferred = max(int(preferred), 1)
            g = min(preferred, num_channels)
            while g > 1 and (num_channels % g) != 0:
                g -= 1
            return max(g, 1)

        self.decoder_swfm = nn.ModuleList([
            Self_Adaptive_Weighted_Fusion_Module(
                in_chan=c, 
                is_first=(i==3),
                use_residual=swfm_use_residual,
                residual_alpha=swfm_residual_alpha,
            ) 
            for i, c in enumerate(decoder_channels)
        ])

        encoder_skip_channels = [384, 256, 128, 128]
        self.skip_align = nn.ModuleList([
            nn.Conv2d(c_skip, c_dec, kernel_size=1, bias=False) if c_skip != c_dec else nn.Identity()
            for c_skip, c_dec in zip(encoder_skip_channels, decoder_channels)
        ])

        prev_channels = [512] + list(decoder_channels[:-1])
        self.h_prev_align = nn.ModuleList([
            nn.Conv2d(c_prev, c_dec, kernel_size=1, bias=False) if c_prev != c_dec else nn.Identity()
            for c_prev, c_dec in zip(prev_channels, decoder_channels)
        ])

        self.h_curr_align = nn.ModuleList([
            nn.Identity() for _ in decoder_channels
        ])

        self.decoder_upsample = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(512, decoder_channels[0], kernel_size=2, stride=2),
                nn.GroupNorm(_choose_gn_groups(decoder_channels[0], 32), decoder_channels[0]),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(decoder_channels[0], decoder_channels[1], kernel_size=2, stride=2),
                nn.GroupNorm(_choose_gn_groups(decoder_channels[1], 32), decoder_channels[1]),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(decoder_channels[1], decoder_channels[2], kernel_size=2, stride=2),
                nn.GroupNorm(_choose_gn_groups(decoder_channels[2], 16), decoder_channels[2]),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(decoder_channels[2], decoder_channels[3], kernel_size=2, stride=2),
                nn.GroupNorm(_choose_gn_groups(decoder_channels[3], 16), decoder_channels[3]),
                nn.ReLU(inplace=True),
            ),
        ])


        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(decoder_channels[3], decoder_channels[3], kernel_size=2, stride=2),
            nn.GroupNorm(_choose_gn_groups(decoder_channels[3], 16), decoder_channels[3]),
            nn.ReLU(inplace=True),
        )

        self.out = nn.Conv2d(decoder_channels[3], 3, kernel_size=1)

        self.model_channels = decoder_channels[3]

        class NetCompat:
            def __init__(self, model_channels, bottleneck_channels, out_layer):
                self.model_channels = model_channels
                self.bottleneck_channels = bottleneck_channels
                self.out = out_layer
        
        self.net = NetCompat(self.model_channels, self.bottleneck_channels, self.out)
        
        self._cached_gating = None
        self._h_end = None
        self.edge_preds = []
        self._decoder_final_features = None

        self.skip_transformers = nn.ModuleList([
            nn.Identity() for _ in range(4)  # e1, e2, e3, e4
        ])

        
    def encode(self, x, task_idx=None, return_fusion_weights=False):
        B, C, H, W = x.shape
        
        fusion_weights = None

        if self.modality_fusion is not None:
            if return_fusion_weights:
                x, fusion_weights = self.modality_fusion(x, return_weights=True)
            else:
                x = self.modality_fusion(x, return_weights=False)
            # x: [B, 4, H, W] -> [B, 96, H, W]

        x_patch = self.swin_encoder.patch_embed(x)
        if self.swin_encoder.ape:
            x_patch = x_patch + self.swin_encoder.absolute_pos_embed
        x_patch = self.swin_encoder.pos_drop(x_patch)

        features = []
        down = None
        for i, layer in enumerate(self.swin_encoder.layers):
            x_patch, down = layer(x_patch if down is None else down)

            H_feat = H // (4 * (2 ** i))
            W_feat = W // (4 * (2 ** i))
            feat = x_patch.view(B, H_feat, W_feat, -1).permute(0, 3, 1, 2)
            features.append(feat)

        e1 = self.channel_align[0](features[0])
        e2 = self.channel_align[1](features[1])
        e3 = self.channel_align[2](features[2])
        e4 = self.channel_align[3](features[3])

        if self.use_fpn and self.fpn_convs is not None:
            e4_to_e3 = self.fpn_convs[0](e4)
            e4_to_e3_up = F.interpolate(e4_to_e3, size=e3.shape[-2:], mode='bilinear', align_corners=False)
            e3 = e3 + e4_to_e3_up
            e3 = self.fpn_refine[0](e3)

            e3_to_e2 = self.fpn_convs[1](e3)
            e3_to_e2_up = F.interpolate(e3_to_e2, size=e2.shape[-2:], mode='bilinear', align_corners=False)
            e2 = e2 + e3_to_e2_up
            e2 = self.fpn_refine[1](e2)

            e2_to_e1 = self.fpn_convs[2](e2)
            e2_to_e1_up = F.interpolate(e2_to_e1, size=e1.shape[-2:], mode='bilinear', align_corners=False)
            e1 = e1 + e2_to_e1_up
            e1 = self.fpn_refine[2](e1)

        e1 = self.skip_transformers[0](e1)
        e2 = self.skip_transformers[1](e2)
        e3 = self.skip_transformers[2](e3)
        e4 = self.skip_transformers[3](e4)

        self._h_end = e4
        
        if return_fusion_weights and fusion_weights is not None:
            return (e1, e2, e3, e4), (e1, e2, e3, e4), fusion_weights
        
        return (e1, e2, e3, e4), (e1, e2, e3, e4)
    
    def bottleneck(self, h=None):
        if h is not None:
            if h.shape[1] == 512:
                xb = h
            else:
                xb = self.bottleneck_proj(h)
            self._cached_gating = xb
            return xb
        else:
            assert self._h_end is not None, "encode() must be called before bottleneck()"
            xb = self.bottleneck_proj(self._h_end)
            self._cached_gating = xb
            return xb
    
    def decode(self, xb, skips=None, return_features=False):
        h = xb
        
        if skips is None:
            skips = [None, None, None, None]
        
        edge_outputs = []
        self.aux_preds = []
        h_prev = h

        for ind in range(4):

            h = self.decoder_upsample[ind](h)
            if skips[-(ind+1)] is not None:
                skip_feat = skips[-(ind+1)]

                h_aligned = self.h_curr_align[ind](h)
                skip_channel_aligned = self.skip_align[ind](skip_feat)
                h_prev_channel_aligned = self.h_prev_align[ind](h_prev)

                h_h, h_w = h_aligned.shape[-2:]
                if skip_channel_aligned.shape[-2:] != (h_h, h_w):
                    skip_aligned = F.interpolate(
                        skip_channel_aligned, size=(h_h, h_w),
                        mode='bilinear', align_corners=False
                    )
                else:
                    skip_aligned = skip_channel_aligned
                
                if h_prev_channel_aligned.shape[-2:] != h_aligned.shape[-2:]:
                    h_prev_aligned = F.interpolate(
                        h_prev_channel_aligned, size=h_aligned.shape[-2:],
                        mode='bilinear', align_corners=False
                    )
                else:
                    h_prev_aligned = h_prev_channel_aligned
                

                skip_gated = self.gag_list[ind](g=h_aligned, x=skip_aligned)

                h = self.decoder_swfm[ind](h_prev_aligned, h_aligned, skip_gated)

            edge_logits = self.edge_heads[ind](h)
            edge_outputs.append(edge_logits)

            h = self.decoder_dropout(h)

            h_prev = h

        h = self.final_upsample(h)

        self.edge_preds = edge_outputs
        self._decoder_final_features = h

        out = self.out(h)
        
        if return_features:
            return out, self._decoder_final_features
        return out
    
    def forward(self, x, task_idx=None, return_features=False):
        skips, _ = self.encode(x, task_idx=task_idx)
        xb = self.bottleneck()
        out = self.decode(xb, skips=skips, return_features=return_features)
        return out
    
    def freeze_stages(self, num_stages=2):

        if num_stages > 0:
            for param in self.swin_encoder.patch_embed.parameters():
                param.requires_grad = False
            if self.swin_encoder.ape:
                self.swin_encoder.absolute_pos_embed.requires_grad = False

        for i in range(min(num_stages, len(self.swin_encoder.layers))):
            for param in self.swin_encoder.layers[i].parameters():
                param.requires_grad = False
    
    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True


def create_swin_backbone(
    img_size=128,
    in_chans=4,
    pretrained_path=None,
    use_checkpoint=False,
    model_size='tiny',
    use_modality_fusion=True,
    modality_fusion_type='advanced',
    num_tasks=3,
    gag_residual_alpha=0.6,
    swfm_use_residual=True,
    swfm_residual_alpha=0.2,
    use_fpn=True,
    adaptive_window_size=True,
    use_fab_align=False,
    align_type='fab',
    fab_reduction=4,
    fusion_mode='full',
    decoder_channels=None,
):

    model_configs = {
        'tiny': {
            'embed_dim': 96,
            'depths': [2, 2, 6, 2],
            'num_heads': [3, 6, 12, 24],
            'drop_path_rate': 0.2,
        },
        'small': {
            'embed_dim': 96,
            'depths': [2, 2, 18, 2],
            'num_heads': [3, 6, 12, 24],
            'drop_path_rate': 0.3,
        },
        'base': {
            'embed_dim': 128,
            'depths': [2, 2, 18, 2],
            'num_heads': [4, 8, 16, 32],
            'drop_path_rate': 0.5,
        }
    }
    
    if model_size not in model_configs:
        raise ValueError(f"model_size必须是 'tiny', 'small', 'base' 之一，但得到 '{model_size}'")
    
    config = model_configs[model_size]
    fusion_status = f"启用模态融合 ({modality_fusion_type})" if use_modality_fusion else "禁用模态融合"
    fpn_status = "启用FPN增强" if use_fpn else "禁用FPN"
    window_status = "阶梯式window_size" if adaptive_window_size else "固定window_size=4"
    print(f"创建Swin-{model_size.capitalize()}编码器: depths={config['depths']}, embed_dim={config['embed_dim']}")
    print(f"  - {fusion_status}")
    print(f"  - {fpn_status}")
    print(f"  - {window_status}")
    
    backbone_kwargs = dict(
        img_size=img_size,
        in_chans=in_chans,
        embed_dim=config['embed_dim'],
        depths=config['depths'],
        num_heads=config['num_heads'],
        window_size=4,
        drop_path_rate=config['drop_path_rate'],
        use_checkpoint=use_checkpoint,
        use_modality_fusion=use_modality_fusion,
        modality_fusion_type=modality_fusion_type,
        num_tasks=num_tasks,
        gag_residual_alpha=gag_residual_alpha,
        swfm_use_residual=swfm_use_residual,
        swfm_residual_alpha=swfm_residual_alpha,
        use_fpn=use_fpn,
        adaptive_window_size=adaptive_window_size,
        use_fab_align=use_fab_align,
        align_type=align_type,
        fab_reduction=fab_reduction,
        fusion_mode=fusion_mode,
    )
    if decoder_channels is not None:
        backbone_kwargs['decoder_channels'] = decoder_channels

    backbone = SwinBackboneAdapter(**backbone_kwargs)
    
    # 加载预训练权重
    if pretrained_path is not None:
        print(f"加载Swin-{model_size.capitalize()}预训练权重: {pretrained_path}")
        try:
            backbone.swin_encoder.load_from(pretrained_path)
            print("✓ 预训练权重加载成功")
        except Exception as e:
            print(f"预训练权重加载失败: {e}")
    else:
        print(f"未提供预训练权重，使用随机初始化Swin-{model_size.capitalize()}")
    
    return backbone

