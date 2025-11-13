"""Full assembly of the parts to form the complete network"""

from .unet_parts import *
import torch.utils.checkpoint


class UNet(nn.Module):

    def __init__(
        self,
        n_channels,
        n_classes,
        bilinear=False,
        scale=1.0,
        with_heatmap=False,
        heatmap_ups=1,
    ):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.with_heatmap = with_heatmap
        self.heatmap_ups = heatmap_ups

        self.inc = DoubleConv(n_channels, int(64 * scale))
        self.down1 = Down(int(64 * scale), int(128 * scale))
        self.down2 = Down(int(128 * scale), int(256 * scale))
        self.down3 = Down(int(256 * scale), int(512 * scale))
        factor = 2 if bilinear else 1
        self.down4 = Down(int(512 * scale), int(1024 * scale) // factor)
        self.up1 = Up(int(1024 * scale), int(512 * scale) // factor, bilinear)
        self.up2 = Up(int(512 * scale), int(256 * scale) // factor, bilinear)
        self.up3 = Up(int(256 * scale), int(128 * scale) // factor, bilinear)
        self.up4 = Up(int(128 * scale), int(64 * scale), bilinear)
        self.outc = OutConv(int(64 * scale), n_classes)
        # add heatmap branch
        if with_heatmap:
            if heatmap_ups == 2:
                self.up3_1 = Up(int(256 * scale),
                                int(128 * scale) // factor, bilinear)
            self.up4_1 = Up(int(128 * scale), int(64 * scale), bilinear)
            self.heatmap = OutConv(int(64 * scale), n_classes)

    def forward(self, x, use_checkpoint=False):
        if use_checkpoint:
            return self.forward_checkpointed(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x_ = self.up3(x, x2)
        # heatmap branch
        heatmap = None
        if self.with_heatmap:
            if self.heatmap_ups == 1:
                x_heatmap = self.up4_1(x_, x1)
            elif self.heatmap_ups == 2:
                x_heatmap = self.up3_1(x, x2)
                x_heatmap = self.up4_1(x_heatmap, x1)
            heatmap = self.heatmap(x_heatmap)
        x_ = self.up4(x_, x1)
        logits = self.outc(x_)
        return logits, heatmap
