import torch
import torch.nn as nn
import torch.nn.functional as F


class Different_Sum_Gate_Fusion(nn.Module):
    def __init__(self, in_chan, is_first=False, use_residual=True, residual_alpha=0.3):
        super(Different_Sum_Gate_Fusion, self).__init__()
        self.inchan = in_chan
        self.is_first = is_first
        self.use_residual = use_residual
        self.residual_alpha = residual_alpha

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

        self.weight_net = nn.Sequential(
            nn.Conv2d(in_chan * 2, in_chan, 1, bias=False),
            nn.BatchNorm2d(in_chan),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_chan, 2, 1, bias=False),
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=2 * in_chan, out_channels=2 * in_chan, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(2 * in_chan),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_chan, out_channels=in_chan, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_chan),
            nn.ReLU(inplace=True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=3 * in_chan, out_channels=in_chan, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_chan),
            nn.ReLU(inplace=True),
        )

        if not is_first:
            # 可学习的融合权重（x0和融合特征的权重）
            self.fusion_weight = nn.Parameter(torch.ones(2) * 0.5)
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels=2 * in_chan, out_channels=in_chan, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(in_chan),
                nn.ReLU(inplace=True),
            )
    
    def forward(self, x0, x1, x2):
        """
        Args:
            x0: 上一层特征（如果is_first=False）
            x1: 当前层特征
            x2: skip特征
        """
        x1_gap = self.gap(x1)
        x1_gmp = self.gmp(x1)
        x2_gap = self.gap(x2)
        x2_gmp = self.gmp(x2)

        x1_desc = x1_gap + x1_gmp
        x2_desc = x2_gap + x2_gmp
        feat_sum = x1_desc + x2_desc
        feat_diff = torch.abs(x1_desc - x2_desc)
        feat_cat = torch.cat([feat_sum, feat_diff], dim=1)
        
        weights = self.weight_net(feat_cat)
        weights = F.softmax(weights, dim=1)
        w1, w2 = weights[:, 0:1], weights[:, 1:2]

        x1_weighted = x1 * w1
        x2_weighted = x2 * w2

        x1_enhanced = x1_weighted
        x2_enhanced = x2_weighted

        Wr_cat = self.conv1(torch.cat((x1_enhanced, x2_enhanced), dim=1))
        Wr1, Wr2 = Wr_cat.chunk(2, dim=1)
        Wr1 = torch.sigmoid(Wr1)
        Wr2 = torch.sigmoid(Wr2)
        
        feat_x1 = x1_enhanced * Wr1
        feat_x2 = x2_enhanced * Wr2
        
        x = self.conv3(torch.cat((self.conv2(feat_x1 + feat_x2), x1_enhanced, x2_enhanced), dim=1))

        if not self.is_first:
            fusion_weights = F.softmax(self.fusion_weight, dim=0)
            x_weighted = fusion_weights[0] * x0 + fusion_weights[1] * x
            x = self.conv(torch.cat((x0, x_weighted), dim=1))

            if self.use_residual:
                x = x * (1.0 - self.residual_alpha) + (x + x0) * self.residual_alpha
        
        return x

if __name__ == '__main__':
    input1 = torch.randn((16, 16, 224, 224))
    input2 = torch.randn((16, 16, 224, 224))
    input3 = torch.randn((16, 16, 224, 224))
    SWFM = Different_Sum_Gate_Fusion(16, is_first=True)
    output = SWFM(input1, input2, input3)
    print(output.shape)