import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log, ceil, floor
from typing import Union, List


# ----------------- 鏇挎崲浠庤繖閲屽紑濮?-----------------

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = kernel_size // 2 if dilation == 1 else dilation

        self.conv_h = nn.Conv2d(in_ch, out_ch, kernel_size=(1, kernel_size), padding=(0, padding),
                                dilation=(1, dilation), bias=False)
        self.conv_v = nn.Conv2d(in_ch, out_ch, kernel_size=(kernel_size, 1), padding=(padding, 0),
                                dilation=(dilation, 1), bias=False)

        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)  # 杩欓噷浣犱篃鍙互闅忔椂寰皟涓?nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_h(x) + self.conv_v(x)
        return self.relu(self.bn(out))


class DownConvBNReLU(ConvBNReLU):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, flag: bool = True):
        super().__init__(in_ch, out_ch, kernel_size, dilation)
        self.down_flag = flag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.down_flag:
            x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)
        # 銆愪慨澶嶇偣銆戯細涓嶈鑷繁璋?conv锛岀洿鎺ユ妸鐗瑰緛浜ょ粰鐖剁被鍘昏繃鍗佸瓧鍗风Н
        return super().forward(x)


class UpConvBNReLU(ConvBNReLU):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, flag: bool = True):
        super().__init__(in_ch, out_ch, kernel_size, dilation)
        self.up_flag = flag

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if self.up_flag:
            x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=False)
        # 銆愪慨澶嶇偣銆戯細灏嗘嫾鎺ュソ鐨勭壒寰佺洿鎺ヤ氦缁欑埗绫昏繃鍗佸瓧鍗风Н
        return super().forward(torch.cat([x1, x2], dim=1))


# ----------------- 鏇挎崲鍒拌繖閲岀粨鏉?-----------------
# =====================================================================
# 鏈枃鏍稿績鍒涙柊鐐?3锛氱壒寰佺骇鍏夋檿鎶戝埗棰勬祴澶?(Feature-level Halo Suppression Head)
# 鐗╃悊鍔ㄦ満锛氫笓闂ㄥ垏闄?SPD 鏃犳崯涓嬮噰鏍峰甫鏉ョ殑鈥滅儹杈愬皠鍏夋檿鈥濓紝閫氳繃缁撴瀯鍖栦晶鎶戝埗瀹炵幇杈圭紭閿愬寲
# =====================================================================
# =====================================================================
# 鏈枃鏍稿績鍒涙柊鐐?3 缁堟瀬鐗堬細甯﹀眬閮ㄦ瀬鍊间繚鎶ょ殑鍏夋檿鎶戝埗澶?(LPP-FHS)
# 鐗╃悊鍔ㄦ満锛氬埄鐢ㄧ孩澶栫洰鏍囩殑灞€閮ㄧ儹杈愬皠鏋佸€肩壒鎬э紝淇濇姢鏍稿績鐗瑰緛涓嶈婀伃锛屼粎鍒囬櫎澶栧洿鍏夋檿
# =====================================================================
class HaloSuppressionHead(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 1):
        super().__init__()
        self.core_path = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True)
        )

        self.halo_path = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=2, dilation=2, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.Sigmoid()
        )

        self.out_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        core_feat = self.core_path(x)
        halo_gate = self.halo_path(x)

        local_max = F.max_pool2d(core_feat, kernel_size=3, stride=1, padding=1)
        feat_mean = core_feat.mean(dim=(2, 3), keepdim=True)
        is_peak = ((core_feat >= local_max - 1e-5) & (local_max > feat_mean * 1.5)).float()

        safe_halo_gate = halo_gate * (1.0 - is_peak)
        sharpened_feat = core_feat - (core_feat * safe_halo_gate)

        return self.out_conv(sharpened_feat)

class RSU(nn.Module):
    """ Residual U-block """

    def __init__(self, height: int, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()

        assert height >= 2
        self.conv_in = ConvBNReLU(in_ch, out_ch)  # stem
        encode_list = [DownConvBNReLU(out_ch, mid_ch, flag=False)]
        decode_list = [UpConvBNReLU(mid_ch * 2, mid_ch, flag=False)]
        for i in range(height - 2):
            encode_list.append(DownConvBNReLU(mid_ch, mid_ch))
            decode_list.append(UpConvBNReLU(mid_ch * 2, mid_ch if i < height - 3 else out_ch))
        encode_list.append(ConvBNReLU(mid_ch, mid_ch, dilation=2))

        self.encode_modules = nn.ModuleList(encode_list)
        self.decode_modules = nn.ModuleList(decode_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.conv_in(x)

        x = x_in
        encode_outputs = []
        for m in self.encode_modules:
            x = m(x)
            encode_outputs.append(x)

        x = encode_outputs.pop()
        for m in self.decode_modules:
            x2 = encode_outputs.pop()
            x = m(x, x2)

        return x + x_in


class RSU4F(nn.Module):
    """
    寰皟浼樺寲锛氶噰鐢?HDC (Hybrid Dilated Convolution) 鍘熷垯銆?    灏嗗師鏈夌殑 [2, 4, 8] 鑶ㄨ儉鐜囨浛鎹负浜掍负璐ㄦ暟鐨?[2, 5, 7]銆?    鍦ㄤ笉澧炲姞浠讳綍鍙傛暟鍜岃绠楅噺鐨勫墠鎻愪笅锛屽交搴曟秷闄ゆ繁灞傜壒寰佺殑缃戞牸浼奖鐩插尯锛?    纭繚寰皬鐩爣鐨勭壒寰佷笉琚仐婕忋€?    """

    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.conv_in = ConvBNReLU(in_ch, out_ch)

        # 銆愪慨鏀圭偣 1銆戯細灏嗗師鏈殑 dilation=4 鏀逛负 5锛宒ilation=8 鏀逛负 7
        self.encode_modules = nn.ModuleList([ConvBNReLU(out_ch, mid_ch),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=2),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=5),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=7)])

        self.decode_modules = nn.ModuleList([ConvBNReLU(mid_ch * 2, mid_ch, dilation=5),
                                             ConvBNReLU(mid_ch * 2, mid_ch, dilation=2),
                                             ConvBNReLU(mid_ch * 2, out_ch)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.conv_in(x)

        x = x_in
        encode_outputs = []
        for m in self.encode_modules:
            x = m(x)
            encode_outputs.append(x)

        x = encode_outputs.pop()
        for m in self.decode_modules:
            x2 = encode_outputs.pop()
            x = m(torch.cat([x, x2], dim=1))

        return x + x_in


class CGDAF(nn.Module):
    """Contrast-Guided Dual Attention Fusion."""
    def __init__(self, in_channel, kernel_size=1, stride=1):
        super(CGDAF, self).__init__()
        self.in_channel = in_channel
        self.inter_channel = in_channel // 2
        self.out_channel = in_channel * 2
        ratio = 4

        self.conv_1 = nn.Sequential(
            nn.Conv2d(self.in_channel, self.inter_channel, 1, stride, 0, bias=False),
            nn.BatchNorm2d(self.inter_channel), nn.ReLU(True),
            nn.Conv2d(self.inter_channel, 1, 1, stride, 0, bias=False),
            nn.BatchNorm2d(1), nn.ReLU(True))
        self.conv_2 = nn.Sequential(
            nn.Conv2d(self.in_channel, self.inter_channel, 1, stride, 0, bias=False),
            nn.BatchNorm2d(self.inter_channel), nn.ReLU(True),
            nn.Conv2d(self.inter_channel, self.inter_channel, 1, stride, 0, bias=False),
            nn.BatchNorm2d(self.inter_channel), nn.ReLU(True))
        self.conv_up = nn.Sequential(
            nn.Conv2d(self.inter_channel, self.inter_channel // ratio, 1),
            nn.LayerNorm([self.inter_channel // ratio, 1, 1]), nn.ReLU(inplace=True),
            nn.Conv2d(self.inter_channel // ratio, self.in_channel, 1),
            nn.LayerNorm([self.in_channel, 1, 1]))
        self.conv_3 = nn.Sequential(
            nn.Conv2d(self.in_channel, self.inter_channel, 1, stride, 0, bias=False),
            nn.BatchNorm2d(self.inter_channel), nn.ReLU(True),
            nn.Conv2d(self.inter_channel, self.inter_channel, 1, stride, 0, bias=False),
            nn.BatchNorm2d(self.inter_channel), nn.ReLU(True))
        self.conv_4 = nn.Sequential(
            nn.Conv2d(self.in_channel, self.inter_channel, 1, stride, 0, bias=False),
            nn.BatchNorm2d(self.inter_channel), nn.ReLU(True),
            nn.Conv2d(self.inter_channel, self.inter_channel, 1, stride, 0, bias=False),
            nn.BatchNorm2d(self.inter_channel), nn.ReLU(True))
        self.DODA = DODA(self.inter_channel)
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Multi-scale contrast branch
        self.center_conv = nn.Conv2d(self.in_channel, 1, 1, 1, 0, bias=False)  # pointwise center intensity
        self.contrast_convs = nn.ModuleList([
            nn.Conv2d(self.in_channel, 1, 3, 1, 1, bias=False),  # 3x3 surround
            nn.Conv2d(self.in_channel, 1, 5, 1, 2, bias=False),  # 5x5 surround
            nn.Conv2d(self.in_channel, 1, 7, 1, 3, bias=False),  # 7x7 surround
        ])
        self.contrast_fuse = nn.Sequential(
            nn.Conv2d(3, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.contrast_alpha = nn.Parameter(torch.tensor([0.05]))

        self.out = nn.Sequential(
            nn.Conv2d(self.in_channel, self.out_channel, 3, 1, 1),
            nn.BatchNorm2d(self.out_channel), nn.ReLU(True))

    def forward(self, E, D):
        # Multi-scale contrast (center - surround)
        contrast_maps = []
        center = self.center_conv(E)  # (B, 1, H, W) 鈥?pointwise center intensity
        for conv in self.contrast_convs:
            surround = conv(E)  # (B, 1, H, W) 鈥?local mean at scale k
            diff = F.relu(center - surround)
            contrast_maps.append(diff)
        contrast = torch.cat(contrast_maps, dim=1)
        contrast_weight = self.contrast_fuse(contrast)

        # Channel attention
        D_ = self.conv_2(D)
        batch, channel, height, width = D_.size()
        D_ = D_.view(batch, channel, height * width)
        E_att = self.conv_1(E).view(batch, 1, height * width)
        E_att = self.softmax(E_att)
        ch_context = torch.matmul(D_, E_att.transpose(1, 2)).unsqueeze(-1)
        ch_context = self.conv_up(self.DODA(ch_context))
        ch_out = D * self.sigmoid(ch_context)

        # Spatial attention
        E_sp = self.conv_3(E)
        batch, channel, height, width = E_sp.size()
        E_sp_pool = self.DODA(self.avg_pool(E_sp))
        E_sp_pool = E_sp_pool.view(batch, channel, 1).permute(0, 2, 1)
        E_sp_pool = self.softmax(E_sp_pool)
        D_sp = self.conv_4(D).view(batch, self.inter_channel, height * width)
        sp_context = torch.matmul(E_sp_pool, D_sp).view(batch, 1, height, width)
        sp_context = F.layer_norm(sp_context, (1, height, width))
        sp_out = E * self.sigmoid(sp_context)

        # Contrast modulation
        modulation = 1.0 + self.contrast_alpha * contrast_weight
        ch_out = ch_out * modulation
        sp_out = sp_out * modulation

        return self.out(ch_out + sp_out)

class DODA(nn.Module):
    def __init__(self, channel, n=2, b=2, inter='ceil'):
        super(DODA, self).__init__()

        kernel_size = int(abs((log(channel, 2) + 1) / 2))
        self.kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.layer_size = 1 + (log(channel, 2) - b) / (2 * n)
        assert inter == 'ceil' or 'floor'
        if inter == 'ceil':
            self.layer_size = ceil(self.layer_size)
        else:
            self.layer_size = floor(self.layer_size)

        self.conv = nn.Conv1d(1, 1, kernel_size=self.kernel_size, padding=int(self.kernel_size / 2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        h = self.conv(x.squeeze(-1).transpose(-1, -2))
        for i in range(self.layer_size):
            h = self.conv(h)
        h = h.transpose(-1, -2).unsqueeze(-1)

        h = self.sigmoid(h)
        return x + h



class SkipConcatFusion(nn.Module):
    """Plain skip fusion used by ablation variants without CGDAF."""

    def forward(self, E, D):
        return torch.cat([E, D], dim=1)


ABLATION_CONFIGS = {
    "A0": {"use_cgdaf": False, "use_peak_head": False},
    "A1": {"use_cgdaf": True, "use_peak_head": False},
    "CEPNet": {"use_cgdaf": True, "use_peak_head": True},
}


def _resolve_ablation(ablation: str, use_cgdaf=None, use_peak_head=None):
    if ablation not in ABLATION_CONFIGS:
        raise ValueError(f"Unknown ablation variant: {ablation}. Expected one of {sorted(ABLATION_CONFIGS)}.")
    cfg = dict(ABLATION_CONFIGS[ablation])
    if use_cgdaf is not None:
        cfg["use_cgdaf"] = use_cgdaf
    if use_peak_head is not None:
        cfg["use_peak_head"] = use_peak_head
    return cfg


class CEPNetBase(nn.Module):

    def __init__(self, cfg: dict, out_ch: int = 1,
                 use_cgdaf=True, use_peak_head=True):
        super().__init__()
        assert "encode" and "decode" in cfg
        self.encode_num = len(cfg["encode"])
        self.use_cgdaf = use_cgdaf
        self.use_peak_head = use_peak_head

        encode_list = []
        side_list = []
        for c in cfg["encode"]:
            # c: [height, in_ch, mid_ch, out_ch, RSU4F, side]
            assert len(c) >= 6
            encode_list.append(RSU(*c[:4]) if c[4] is False else RSU4F(*c[1:4]))
            if c[5] is True:
                side_list.append(nn.Conv2d(c[3], out_ch, kernel_size=3, padding=1))
        self.encode_modules = nn.ModuleList(encode_list)

        decode_list = []
        ipof_list = []
        for c in cfg["decode"]:
            # c: [height, in_ch, mid_ch, out_ch, RSU4F, side]
            assert len(c) >= 6
            decode_list.append(RSU(*c[:4]) if c[4] is False else RSU4F(*c[1:4]))
            ipof_list.append(CGDAF(int(c[1] / 2)) if self.use_cgdaf else SkipConcatFusion())
            if c[5] is True:
                side_list.append(nn.Conv2d(c[3], out_ch, kernel_size=3, padding=1))

        self.decode_modules = nn.ModuleList(decode_list)
        self.side_modules = nn.ModuleList(side_list)
        self.ipof_modules = nn.ModuleList(ipof_list)

        out_in_ch = len(side_list) * out_ch
        if self.use_peak_head:
            self.out_conv = HaloSuppressionHead(in_ch=out_in_ch, out_ch=out_ch)
        else:
            self.out_conv = nn.Conv2d(out_in_ch, out_ch, kernel_size=1)

        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        _, _, h, w = x.shape

        encode_outputs = []
        for i, m in enumerate(self.encode_modules):
            x = m(x)
            encode_outputs.append(x)
            if i != self.encode_num - 1:
                x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)

        x = encode_outputs.pop()
        decode_outputs = [x]
        for m in zip(self.decode_modules, self.ipof_modules):
            x2 = encode_outputs.pop()
            x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=False)
            x = m[1](x, x2)
            x = m[0](x)
            decode_outputs.insert(0, x)

        side_outputs = []
        for m in self.side_modules:
            x = decode_outputs.pop()
            x = F.interpolate(m(x), size=[h, w], mode='bilinear', align_corners=False)
            side_outputs.insert(0, x)

        x = self.out_conv(torch.concat(side_outputs, dim=1))

        if self.training:
            return [x] + side_outputs
        else:
            return self.bn(x)


def CEPNet_L(out_ch: int = 1, ablation: str = "CEPNet",
             use_cgdaf=None, use_peak_head=None):
    ab_cfg = _resolve_ablation(ablation, use_cgdaf, use_peak_head)
    cfg = {
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "encode": [[7, 3, 16, 64, False, False],  # En1
                   [6, 64, 16, 64, False, False],  # En2
                   [5, 64, 32, 64, False, False],  # En3
                   [4, 64, 32, 128, False, False],  # En4
                   [4, 128, 32, 128, True, False],  # En5
                   [4, 128, 64, 128, True, True]],  # En6
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "decode": [[4, 256, 64, 128, True, True],  # De5
                   [4, 256, 32, 64, False, True],  # De4
                   [5, 128, 32, 64, False, True],  # De3
                   [6, 128, 16, 64, False, True],  # De2
                   [7, 128, 16, 64, False, True]]  # De1
    }

    return CEPNetBase(cfg, out_ch, **ab_cfg)


def CEPNet_M(out_ch: int = 1, ablation: str = "CEPNet",
             use_cgdaf=None, use_peak_head=None):
    ab_cfg = _resolve_ablation(ablation, use_cgdaf, use_peak_head)
    cfg = {
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "encode": [[7, 3, 16, 64, False, False],  # En1
                   [6, 64, 16, 64, False, False],  # En2
                   [5, 64, 16, 64, False, False],  # En3
                   [4, 64, 16, 64, False, False],  # En4
                   [4, 64, 16, 64, True, False],  # En5
                   [4, 64, 16, 64, True, True]],  # En6
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "decode": [[4, 128, 16, 64, True, True],  # De5
                   [4, 128, 16, 64, False, True],  # De4
                   [5, 128, 16, 64, False, True],  # De3
                   [6, 128, 16, 64, False, True],  # De2
                   [7, 128, 16, 64, False, True]]  # De1
    }

    return CEPNetBase(cfg, out_ch, **ab_cfg)


def CEPNet(out_ch: int = 1, ablation: str = "CEPNet",
           use_cgdaf=None, use_peak_head=None):
    ab_cfg = _resolve_ablation(ablation, use_cgdaf, use_peak_head)
    """
    榛勯噾姣斾緥寰皟鐗?(S+ 妯″瀷)锛?    灏嗗師鐗?S 鏋佸害骞茬槳鐨?8 閫氶亾鎷撳鑷?16 閫氶亾锛屼腑闂寸摱棰堜粠 4 鎷撳鑷?8銆?    鏃㈣兘鎻愪緵瓒冲鐨勫甫瀹芥潵鎵胯浇 SPD 鍜?鍗佸瓧鍗风Н鐨勮竟缂樼壒寰侊紝
    鍙堣兘灏嗗弬鏁伴噺姝绘鍘嬪埗鍦ㄦ瀬鍏惰交閲忕殑绾у埆锛堥璁＄害 0.1M ~ 0.2M锛夈€?    """
    cfg = {
        # 鏍煎紡: [height, in_ch, mid_ch, out_ch, RSU4F, side]
        "encode": [[7, 3, 8, 16, False, False],  # En1: 杈撳叆3(RGB), 鐡堕8, 杈撳嚭16
                   [6, 16, 8, 16, False, False],  # En2
                   [5, 16, 8, 16, False, False],  # En3
                   [4, 16, 8, 16, False, False],  # En4
                   [4, 16, 8, 16, True, False],  # En5
                   [4, 16, 8, 16, True, True]],  # En6

        # 瑙ｇ爜鍣ㄤ腑锛宨n_ch 閫氬父鏄笂涓€灞傜殑 out_ch 鍜岃烦璺冭繛鎺ョ殑鎷兼帴 (16+16=32)
        # 鏍煎紡: [height, in_ch, mid_ch, out_ch, RSU4F, side]
        "decode": [[4, 32, 8, 16, True, True],  # De5
                   [4, 32, 8, 16, False, True],  # De4
                   [5, 32, 8, 16, False, True],  # De3
                   [6, 32, 8, 16, False, True],  # De2
                   [7, 32, 8, 16, False, True]]  # De1
    }

    return CEPNetBase(cfg, out_ch, **ab_cfg)
