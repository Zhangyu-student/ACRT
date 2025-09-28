import torch
import torch.nn as nn
import numbers
from torch.nn import functional as F
from einops import rearrange
from thop import profile


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature1 = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv1 = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv2 = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1,
            padding=1, groups=dim * 3, bias=bias
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, level_factor):
        b, c, h, w = x.shape

        # 处理level_factor，确保是整数
        if isinstance(level_factor, torch.Tensor):
            if level_factor.numel() > 1:
                level_factor = level_factor.item()
            else:
                level_factor = level_factor.item()

        level_factor = max(1, int(level_factor))
        scale_factor = max(1, 64 // level_factor)

        # 第一分支
        qkv1 = self.qkv1(x)
        q1, k1, v1 = qkv1.chunk(3, dim=1)

        # 功能式下采样
        def downsample(tensor):
            return F.avg_pool2d(tensor, kernel_size=scale_factor, stride=scale_factor) if scale_factor > 1 else tensor

        q_down = downsample(q1)
        k_down = downsample(k1)
        v_down = downsample(v1)

        # 准备空间注意力张量
        q_down = rearrange(q_down, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_down = rearrange(k_down, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_down = rearrange(v_down, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # 空间注意力计算
        q_down = F.normalize(q_down, dim=-2)
        k_down = F.normalize(k_down, dim=-2)

        attn1 = (q_down.transpose(-2, -1) @ k_down) * self.temperature1
        attn1 = attn1.softmax(dim=-1)

        out1 = (attn1 @ v_down.transpose(-2, -1)).transpose(-2, -1)
        out1 = rearrange(out1, 'b head c (h w) -> b (head c) h w',
                         head=self.num_heads, h=h // scale_factor, w=w // scale_factor)

        # 功能式上采样
        if scale_factor > 1:
            out1 = F.interpolate(out1, scale_factor=scale_factor, mode='nearest')

        # 第二分支
        qkv2 = self.qkv_dwconv(self.qkv2(x))
        q2, k2, v2 = qkv2.chunk(3, dim=1)

        # 通道注意力
        q2 = rearrange(q2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k2 = rearrange(k2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v2 = rearrange(v2, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q2 = F.normalize(q2, dim=-1)
        k2 = F.normalize(k2, dim=-1)

        attn = (q2 @ k2.transpose(-2, -1)) * self.temperature2
        attn = attn.softmax(dim=-1)

        out = attn @ v2
        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        # 结合注意力输出
        out_mul = out * out1
        return self.project_out(out_mul)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, i):
        # 确保输入始终是张量，并分离出level因子
        if isinstance(x, (tuple, list)):
            input_tensor = x[0]
            level_factor = x[1]
        else:
            input_tensor = x
            level_factor = i

        # 主处理流程
        x_tmp = self.attn(self.norm1(input_tensor), level_factor)
        x_out = input_tensor + x_tmp
        x_out = x_out + self.ffn(self.norm2(x_out))

        # 只返回特征张量
        return x_out


# class ResNetBlock(nn.Module):
#     def __init__(self, dim, bias=True):
#         super(ResNetBlock, self).__init__()
#         self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)
#         self.bn1 = nn.BatchNorm2d(dim)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)
#         self.bn2 = nn.BatchNorm2d(dim)
#
#     def forward(self, x):
#         identity = x  # Save the input for the residual connection
#
#         # First convolutional layer
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         # Second convolutional layer
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         # Add the residual connection
#         out += identity
#         out = self.relu(out)
#
#         return out
#
# class TransformerBlock(nn.Module):
#     def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
#         super(TransformerBlock, self).__init__()
#
#         self.norm1 = LayerNorm(dim, LayerNorm_type)
#         # self.attn = Attention(dim, num_heads, bias)
#         self.res = ResNetBlock(dim, bias)
#         self.norm2 = LayerNorm(dim, LayerNorm_type)
#         self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
#
#     def forward(self, x , i):
#
#         if isinstance(x, (tuple, list)):
#             input_tensor = x[0]
#         else:
#             input_tensor = x
#
#         # ResBlock的消融实验
#         x = self.res(input_tensor)
#         # print(f"x的shape为{x.shape}")
#         x = x + self.ffn(self.norm2(x))
#
#         return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class DIRformer(nn.Module):
    def __init__(self,
                 inp_channels=7,
                 out_channels=3,
                 dim=64,
                 num_blocks=[1, 1, 1, 1],
                 num_refinement_blocks=4,
                 heads=[1, 1, 2, 4],
                 ffn_expansion_factor=1.6,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 dual_pixel_task=False  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 ):

        super(DIRformer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        # 使用ModuleList替代Sequential以便收集attention map
        self.encoder_level1 = nn.ModuleList([
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks[0])
        ])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.ModuleList([
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks[1])
        ])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3

        self.encoder_level3 = nn.ModuleList([
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks[2])
        ])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.ModuleList([
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks[3])
        ])

        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.ModuleList([
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks[2])
        ])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.ModuleList([
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks[1])
        ])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1 (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.ModuleList([
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks[0])
        ])

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        # Encoder Level 1
        inp_enc_level1 = self.patch_embed(inp_img)
        x = inp_enc_level1
        for blk in self.encoder_level1:
            x = blk(x, 1)  # Level factor=1
        out_enc_level1 = x

        # Downsample to Level 2
        inp_enc_level2 = self.down1_2(out_enc_level1)
        x = inp_enc_level2
        for blk in self.encoder_level2:
            x = blk(x, 2)  # Level factor=2
        out_enc_level2 = x

        # Downsample to Level 3
        inp_enc_level3 = self.down2_3(out_enc_level2)
        x = inp_enc_level3
        for blk in self.encoder_level3:
            x = blk(x, 4)  # Level factor=4
        out_enc_level3 = x

        # Downsample to Level 4 (Latent)
        inp_enc_level4 = self.down3_4(out_enc_level3)
        x = inp_enc_level4
        for blk in self.latent:
            x = blk(x, 8)  # Level factor=8
        latent = x

        # Upsample from Level 4 to Level 3
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        x = inp_dec_level3
        for blk in self.decoder_level3:
            x = blk(x, 4)  # Level factor=2
        out_dec_level3 = x

        # Upsample from Level 3 to Level 2
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        x = inp_dec_level2
        for blk in self.decoder_level2:
            x = blk(x, 2)  # Level factor=4
        out_dec_level2 = x

        # Upsample from Level 2 to Level 1
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        x = inp_dec_level1
        for blk in self.decoder_level1:
            x = blk(x, 1)  # Level factor=1
        out_dec_level1 = x

        # Final output
        out_dec_level1 = self.output(out_dec_level1)

        return out_dec_level1


class DiffIRS1(nn.Module):
    def __init__(self,
                 inp_channels=7,
                 out_channels=3,
                 dim=64,
                 num_blocks=[1, 1, 2, 4],
                 heads=[1, 1, 1, 1],
                 ffn_expansion_factor=1.6,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 dual_pixel_task=False  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 ):
        super(DiffIRS1, self).__init__()
        self.G = DIRformer(
            inp_channels=inp_channels,
            out_channels=out_channels,
            dim=dim,
            num_blocks=num_blocks,
            heads=heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
            dual_pixel_task=dual_pixel_task
        )

    def forward(self, x, ref_img):
        x = torch.cat((x, ref_img), dim=1)
        sr = self.G(x)
        return sr


def compute_params_and_flops():
    # 表格中的配置列表
    configs = [
        # Block配置 [1,1,1,1] 对应三组Heads
        {"blocks": [1, 1, 1, 1], "heads_list": [
            [1, 1, 1, 1],
            [1, 1, 2, 4],
            [1, 2, 4, 8]
        ]},
        # Block配置 [1,1,2,4] 对应三组Heads
        {"blocks": [1, 1, 2, 4], "heads_list": [
            [1, 1, 1, 1],
            [1, 1, 2, 4],
            [1, 2, 4, 8]
        ]},
        # Block配置 [1,2,4,8] 对应三组Heads
        {"blocks": [1, 2, 4, 8], "heads_list": [
            [1, 1, 1, 1],
            [1, 1, 2, 4],
            [1, 2, 4, 8]
        ]}
    ]

    # 创建结果表格
    results = []

    # 输入尺寸 (保持固定)
    input_x = torch.randn(1, 4, 512, 512)
    ref_img = torch.randn(1, 3, 512, 512)

    # 遍历所有配置
    for config in configs:
        blocks = config["blocks"]

        for heads in config["heads_list"]:
            # 创建对应配置的模型
            model = DiffIRS1(
                inp_channels=7,
                out_channels=3,
                dim=64,
                num_blocks=blocks,
                heads = heads
            )

            # 计算参数量
            total_params = sum(p.numel() for p in model.parameters())
            params_m = total_params / 1e6  # 转换为百万

            # 计算MACs
            macs, _ = profile(model, inputs=(input_x, ref_img), verbose=False)
            macs_g = macs / 1e9  # 转换为Giga

            # 存储结果
            results.append({
                "Blocks": blocks,
                "Heads": heads,
                "Params": params_m,
                "MACs": macs_g
            })

    # 打印表格
    print("\n{:<15} {:<15} {:<15} {:<15}".format("Blocks", "Heads", "Params (M)", "MACs (G)"))
    print("-" * 60)
    for res in results:
        blocks_str = str(res["Blocks"])
        heads_str = ' '.join(map(str, res["Heads"]))
        print("{:<15} {:<15} {:<15.3f} {:<15.3f}".format(
            blocks_str, heads_str, res["Params"], res["MACs"]
        ))


if __name__ == "__main__":
    compute_params_and_flops()