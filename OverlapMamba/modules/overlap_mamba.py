


import os
import sys
p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
if p not in sys.path:
    sys.path.append(p)
sys.path.append('../tools/')
sys.path.append('../mambapy/')
import time
import torch
import torch.nn as nn

from modules.netvlad import NetVLADLoupe
import torch.nn.functional as F
from tools.read_samples import read_one_need_from_seq
import yaml
from mambapy.mamba import Mamba, MambaConfig
from mambapy.mamba_lm import MambaLMConfig, MambaLM
# from mamba_ssm.modules.mamba_simple import Mamba


"""
    Feature extracter of OverlapTransformer.
    Args:
        height: the height of the range image (64 for KITTI sequences). 
                 This is an interface for other types LIDAR.
        width: the width of the range image (900, alone the lines of OverlapNet).
                This is an interface for other types LIDAR.
        channels: 1 for depth only in our work. 
                This is an interface for multiple cues.
        norm_layer: None in our work for better model.
        use_transformer: Whether to use MHSA.
        
"""


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, (1, 3), 1, p=(0, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, padding= p,groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SPPF(nn.Module): # [1, 128, 1, 900]
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1=128, c2=128, k=(1, 5)):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, (1, 1), (1, 1))
        self.cv2 = Conv(c_ * 4, c2, (1, 1), (1, 1))
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=(0, 2))

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, img_size=(64, 900), patch_size=(16, 4), stride=(16, 4), in_chans=1, embed_dim=128, norm_layer=None,
                 flatten=True):
        super().__init__()
        self.img_size = img_size  # (64, 900)
        self.patch_size = patch_size  # (16, 4)
        self.grid_size = (
        (img_size[0] - patch_size[0]) // stride[0] + 1, (img_size[1] - patch_size[1]) // stride[0] + 1)  # (4, 225)
        self.num_patches = self.grid_size[0] * self.grid_size[1]  # 900
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)  # [1, 1, 64, 900] -> [1, 128, 4, 255]
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape  # [1, 1, 64, 900]
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)  # [1, 256, 4, 255]
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC [1, 900, 256]
        x = self.norm(x)
        return x

class featureExtracter(nn.Module):
    def __init__(self, height=64, width=900, channels=5, norm_layer=None, use_transformer=False, use_mamba=True,
                 use_patch_embed=False, use_mamba_lm=False, use_conv=False):
        super(featureExtracter, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.use_transformer = use_transformer
        self.use_mamba = use_mamba
        self.ues_patch_embed = use_patch_embed
        self.use_mamba_lm = use_mamba_lm
        self.use_conv = use_conv

        if not self.ues_patch_embed:
            if not self.use_conv:
                self.conv1 = nn.Conv2d(channels, 16, kernel_size=(5,1), stride=(1,1), bias=False)
                self.bn1 = norm_layer(16)
                self.conv2 = nn.Conv2d(16, 32, kernel_size=(3,1), stride=(2,1), bias=False)
                self.bn2 = norm_layer(32)
                self.conv3 = nn.Conv2d(32, 64, kernel_size=(3,1), stride=(2,1), bias=False)
                self.bn3 = norm_layer(64)
                self.conv4 = nn.Conv2d(64, 64, kernel_size=(3,1), stride=(2,1), bias=False)
                self.bn4 = norm_layer(64)
                self.conv5 = nn.Conv2d(64, 128, kernel_size=(2,1), stride=(2,1), bias=False)
                self.bn5 = norm_layer(128)
                self.conv6 = nn.Conv2d(128, 128, kernel_size=(1,1), stride=(1,1), bias=False)
                self.bn6 = norm_layer(128)
                self.conv7 = nn.Conv2d(128, 128, kernel_size=(1,1), stride=(2,1), bias=False)
                self.bn7 = norm_layer(128)
                self.conv8 = nn.Conv2d(128, 128, kernel_size=(1,1), stride=(2,1), bias=False)
                self.bn8 = norm_layer(128)
                self.conv9 = nn.Conv2d(128, 128, kernel_size=(1,1), stride=(2,1), bias=False)
                self.bn9 = norm_layer(128)
                self.conv10 = nn.Conv2d(128, 128, kernel_size=(1,1), stride=(2,1), bias=False)
                self.bn10 = norm_layer(128)
                self.conv11 = nn.Conv2d(128, 128, kernel_size=(1,1), stride=(2,1), bias=False)
                self.bn11 = norm_layer(128)
                self.sppf = SPPF(128)
            else:
                self.conv1 = Conv(channels, 16, k=(6, 1), s=(2, 1), p=(2, 0))
                self.conv2 = Conv(16, 32, k=(3, 1), s=(2, 1))  # 16
                self.c3_1 = C3(32, 32, 3)
                self.conv3 = Conv(32, 64, k=(3, 1), s=(2, 1))
                self.c3_2 = C3(64, 64, 3)
                self.conv4 = Conv(64, 128, k=(3, 1), s=(2, 1))
                self.c3_3 = C3(128, 128, 3)
                self.conv5 = Conv(128, 128, k=(3, 1), s=(2, 1))
                self.c3_4 = C3(128, 128, 3)
                self.sppf = SPPF(128)
        else:
            self.patch_embed = PatchEmbed()

        self.relu = nn.ReLU(inplace=True)

        """
            MHSA
            num_layers=1 is suggested in our work.
        """
        if self.use_mamba:
            if self. use_mamba_lm:
                config = MambaLMConfig(d_model=256, n_layers=1, vocab_size=1000)
                self.mamba_encoder = MambaLM(config)

            else:
                config = MambaConfig(d_model=256, n_layers=1)
                self.mamba_encoder = Mamba(config)


        if self.use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=1024, activation='relu', batch_first=False,dropout=0.)
            self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.convLast1 = nn.Conv2d(128, 256, kernel_size=(1,1), stride=(1,1), bias=False)
        self.bnLast1 = norm_layer(256)
        self.convLast2 = nn.Conv2d(512, 1024, kernel_size=(1,1), stride=(1,1), bias=False)
        self.bnLast2 = norm_layer(1024)

        self.linear = nn.Linear(128*900, 256)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

        """
            NETVLAD
            add_batch_norm=False is needed in our work.
        """
        self.net_vlad = NetVLADLoupe(feature_size=1024, max_samples=900, cluster_size=64,
                                     output_dim=256, gating=True, add_batch_norm=False,
                                     is_training=True)

        """TODO: How about adding some dense layers?"""
        self.linear1 = nn.Linear(1 * 256, 256)
        self.bnl1 = norm_layer(256)
        self.linear2 = nn.Linear(1 * 256, 256)
        self.bnl2 = norm_layer(256)
        self.linear3 = nn.Linear(1 * 256, 256)
        self.bnl3 = norm_layer(256)

    def forward(self, x_l):
        if self.ues_patch_embed:
            out_l = self.patch_embed(x_l)
            out_l = out_l.unsqueeze(2).permute(0, 3, 2, 1)

        elif self.use_conv:
            out_l = self.conv1(x_l)
            out_l = self.conv2(out_l)
            out_l = self.c3_1(out_l)
            out_l = self.conv3(out_l)
            out_l = self.c3_2(out_l)
            out_l = self.conv4(out_l)
            out_l = self.c3_3(out_l)
            out_l = self.conv5(out_l)
            out_l = self.c3_4(out_l)
            out_l = self.sppf(out_l)

        else:

            out_l = self.relu(self.conv1(x_l))
            out_l = self.relu(self.conv2(out_l))
            out_l = self.relu(self.conv3(out_l))
            out_l = self.relu(self.conv4(out_l))
            out_l = self.relu(self.conv5(out_l))
            out_l = self.relu(self.conv6(out_l))
            out_l = self.relu(self.conv7(out_l))
            out_l = self.relu(self.conv8(out_l))
            out_l = self.relu(self.conv9(out_l))
            out_l = self.relu(self.conv10(out_l))
            out_l = self.relu(self.conv11(out_l))
            out_l = self.sppf(out_l)

        out_l_1 = out_l.permute(0, 1, 3, 2)  # [1, 128, 1, 900] -> [1, 128, 900, 1]
        out_l_1 = self.relu(self.convLast1(out_l_1))

        """Using transformer needs to decide whether batch_size first"""
        if self.use_transformer:
            out_l = out_l_1.squeeze(3)  # [1, 256, 900]
            out_l = out_l.permute(2, 0, 1)
            out_l = self.transformer_encoder(out_l)
            out_l = out_l.permute(1, 2, 0)
            out_l = out_l.unsqueeze(3)

            out_l = torch.cat((out_l_1, out_l), dim=1)
            out_l = self.relu(self.convLast2(out_l))
            out_l = F.normalize(out_l, dim=1)
            out_l = self.net_vlad(out_l)
            out_l = F.normalize(out_l, dim=1)

        elif self.use_mamba:
            out_l = out_l_1.squeeze(3)
            out_l = out_l.permute(0, 2, 1)
            out_l = self.mamba_encoder(out_l)
            out_l = out_l.permute(0, 2, 1)
            out_l = out_l.unsqueeze(3)

            out_l = torch.cat((out_l_1, out_l), dim=1)
            out_l = self.relu(self.convLast2(out_l))
            out_l = F.normalize(out_l, dim=1)
            out_l = self.net_vlad(out_l)
            out_l = F.normalize(out_l, dim=1)

        else:
            out_l = torch.cat((out_l_1, out_l_1), dim=1)
            out_l = F.normalize(out_l, dim=1)
            out_l = self.net_vlad(out_l)
            out_l = F.normalize(out_l, dim=1)

        # print(f'coast:{time.time() - t:.8f}s')
        return out_l


if __name__ == '__main__':
    # load config ================================================================
    config_filename = '../config/config.yml'
    config = yaml.safe_load(open(config_filename))
    seqs_root = config["data_root"]["data_root_folder"]
    # ============================================================================

    combined_tensor = read_one_need_from_seq(seqs_root, "000000","00")
    combined_tensor = torch.cat((combined_tensor,combined_tensor), dim=0)

    feature_extracter=featureExtracter(use_transformer=False, channels=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extracter.to(device)
    feature_extracter.eval()

    print("model architecture: \n")
    print(feature_extracter)

    gloabal_descriptor = feature_extracter(combined_tensor)
    print("size of gloabal descriptor: \n")
    print(gloabal_descriptor.size())
