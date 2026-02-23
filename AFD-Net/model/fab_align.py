import torch
import torch.nn as nn
class FeatureAdaptiveBiasAlign(nn.Module):
    
    def __init__(self, in_channels, out_channels, reduction=4, init_bias_scale=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.projection = nn.Conv2d(in_channels, out_channels, 
                                   kernel_size=1, bias=False)

        hidden_dim = max(out_channels // reduction, 16)
        self.bias_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1),
        )

        self.bias_scale = nn.Parameter(torch.ones(1) * init_bias_scale)

        nn.init.zeros_(self.bias_generator[-1].weight)
        if self.bias_generator[-1].bias is not None:
            nn.init.zeros_(self.bias_generator[-1].bias)
    
    def forward(self, x):
        """
        Args:
            x: [B, in_channels, H, W]
        Returns:
            y: [B, out_channels, H, W]
        """
        y = self.projection(x)  # [B, out_channels, H, W]

        bias = self.bias_generator(x)  # [B, out_channels, 1, 1]

        bias = bias * self.bias_scale

        y = y + bias
        
        return y
    
    def extra_repr(self):
        return f'in_channels={self.in_channels}, out_channels={self.out_channels}'


def create_fab_align(in_channels, out_channels, reduction=4, init_bias_scale=0.1):

    return FeatureAdaptiveBiasAlign(in_channels, out_channels, reduction, init_bias_scale)


