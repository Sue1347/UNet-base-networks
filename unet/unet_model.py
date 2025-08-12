""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

# class UNet_7(nn.Module):
#     # 7 layers, add more layers, as the nnUNet structure
#     def __init__(self, n_channels, n_classes, bilinear=True):
#         super(UNet_7, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear

#         self.inc = (DoubleConv(n_channels, 32))
#         self.down1 = (Down(32, 64)) # /2
#         self.down2 = (Down(64, 128)) # /4
#         self.down3 = (Down(128, 256)) # /8
#         self.down4 = (Down(256, 512)) # /32 (256/32 = 8)
#         self.down5 = (DoubleConv(512, 512)) # since /64 (256/64 = 4) is too small
#         self.down6 = (DoubleConv(512, 512))
#         factor = 2
#         self.down7 = (Down(512, 512)) # /64 (4*4)
#         self.up1 = (Up(512 * factor, 512, bilinear))
#         self.up2 = (DoubleConv(512 * factor, 512))
#         self.up3 = (DoubleConv(512 * factor, 256))
#         self.up4 = (Up(512, 256 // factor, bilinear))
#         self.up5 = (Up(256, 128 // factor, bilinear))
#         self.up6 = (Up(128, 64, bilinear))
#         self.up7 = (Up(64, 32, bilinear))
#         self.outc = (OutConv(32, n_classes))

#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x6 = self.down5(x5)
#         x7 = self.down6(x6)
#         x8 = self.down7(x7)
#         o1 = self.up1(x8, x7)
#         o2 = self.up2(match_cat(o1, x6))
#         o3 = self.up3(match_cat(o2, x5))
#         o4 = self.up4(o3, x4)
#         o5 = self.up5(o4, x3)
#         o6 = self.up6(o5, x2)
#         o7 = self.up7(o6, x1)
#         logits = self.outc(o7)
#         return logits


class UNet_7(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_7, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        factor = 2

        # Encoder
        self.inc = DoubleConv(n_channels, 32)
        self.down_blocks = nn.ModuleList([
            Down(32, 64),         # /2
            Down(64, 128),        # /4
            Down(128, 256),       # /8
            Down(256, 512),       # /16
            DoubleConv(512, 512), # no pooling
            DoubleConv(512, 512), # no pooling
            Down(512, 512)        # /32
        ])

        # Decoder
        self.up_blocks = nn.ModuleList([
            Up(512 * factor, 512, bilinear),
            DoubleConv(512 * factor, 512),
            DoubleConv(512 * factor, 256),
            Up(512, 256 // factor, bilinear),
            Up(256, 128 // factor, bilinear),
            Up(128, 64 // factor, bilinear),
            Up(64, 32, bilinear)
        ])

        # Output
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        # Encoder
        enc_feats = []
        x = self.inc(x)
        enc_feats.append(x)
        for i, down in enumerate(self.down_blocks):
            x = down(x)
            enc_feats.append(x)

        # Decoder
        # reverse encoder features for skip connections (ignore last feature map)
        skips = enc_feats[:-1][::-1]
        x = enc_feats[-1]

        for i, up in enumerate(self.up_blocks):
            # print(f"now is {i}: {up}")
            if isinstance(up, Up):  # Up takes skip connection
                x = up(x, skips.pop(0))
            else:  # DoubleConv takes concatenated tensor
                x = up(match_cat(x, skips.pop(0)))

        return self.outc(x)