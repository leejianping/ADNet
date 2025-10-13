import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers

from einops import rearrange


# Layer Normalization
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


# Differentiable Binarization for QR codes
class DifferentiableBinarization(nn.Module):
    def __init__(self, temperature=2.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(float(temperature)))

    def forward(self, x):
        return torch.sigmoid((x - 0.5) / self.temperature)


# GDFN with Binarization
class GDFN(nn.Module):
    def __init__(self, dim, expansion_factor=2.66, use_binarization=True):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_dim * 2, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_dim * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False)
        self.use_binarization = use_binarization
        if use_binarization:
            self.binarization = DifferentiableBinarization(temperature=2.0)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1 = F.gelu(x1)
        x2 = torch.sigmoid(x2)
        x = x1 * x2
        x = self.project_out(x)
        if self.use_binarization:
            x = self.binarization(x)
        return x


class EdgeGuidedMDTA(nn.Module):
    
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=False)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

        # �෽���Ե��� - ����������Ҫ����
        self.setup_multi_directional_edge_detection()
        
    #simplified version, only multiply direction
    def setup_multi_directional_edge_detection(self):
        #�����ĸ�����ı�Ե�������
        
        # ˮƽ��Ե��� (��ⴹֱ�˶�ģ���е�ˮƽ�ṹ)
        self.edge_horizontal = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        horizontal_kernel = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.edge_horizontal.weight = nn.Parameter(horizontal_kernel, requires_grad=False)
        
        # ��ֱ��Ե��� (���ˮƽ�˶�ģ���еĴ�ֱ�ṹ)
        self.edge_vertical = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        vertical_kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        self.edge_vertical.weight = nn.Parameter(vertical_kernel, requires_grad=False)
        
        # �ԽǱ�Ե���1 (45�ȷ���)
        self.edge_diagonal1 = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        float32)
        self.edge_diagonal1.weight = nn.Parameter(diagonal1_kernel, requires_grad=False)
        
        # �ԽǱ�Ե���2 (135�ȷ���)
        self.edge_diagonal2 = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        diagonal2_kernel = torch.tensor([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]], dtype=torch.float32)
        self.edge_diagonal2.weight = nn.Parameter(diagonal2_kernel, requires_grad=False)

    def extract_multi_directional_edges(self, x):
        x_mean = x.mean(dim=1, keepdim=True)
    
        # ˳����㲢�����ͷ��м���
        edge_map = self.edge_horizontal(x_mean).abs()
        edge_map = torch.max(edge_map, self.edge_vertical(x_mean).abs())
        edge_map = torch.max(edge_map, self.edge_diagonal1(x_mean).abs())
        edge_map = torch.max(edge_map, self.edge_diagonal2(x_mean).abs())
    
        return torch.sigmoid(edge_map)

    def forward(self, x):
        b, c, h, w = x.shape

        # �෽���Ե��ȡ
        edge_map = self.extract_multi_directional_edges(x)
        

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        # ʹ�ñ�Ե��Ϣ����q��k
        # ���ӵ���ǿ�ȣ�ȷ����Ե��Ϣ�õ��������
        q = q*(1 + 0.1 * edge_map)  # ��0.1���ӵ�0.2
        k = k*(1 + 0.1 * edge_map)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out



#########################################################################
# EGRestormer Transformer Block
#########################################################################
class QRTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, use_binarization, LayerNorm_type):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = EdgeGuidedMDTA(dim, num_heads=num_heads)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = GDFN(dim, use_binarization=use_binarization)

    def forward(self, x):
        attn_out = self.attn(self.norm1(x))
        x = x + attn_out
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out
        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


#########################################################################
# EGRestormer Architecture
#########################################################################
class EGRestormer(nn.Module):
    def __init__(self, inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 apply_binarization=True
                 ):
        super().__init__()
        self.apply_binarization = apply_binarization
        self.conv_in = nn.Conv2d(inp_channels, dim, kernel_size=3, padding=1)

        # Encoder with downsampling layers
        self.encoder_level1 = nn.Sequential(*[
            QRTransformerBlock(int(dim), heads[0], use_binarization= False, LayerNorm_type=LayerNorm_type) for _ in
            range(num_blocks[0])])

        self.down1_2 = Downsample(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            QRTransformerBlock(int(dim * (2 ** 1)), heads[1], use_binarization= False, LayerNorm_type=LayerNorm_type) for _ in
            range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            QRTransformerBlock(int(dim * (2 ** 2)), heads[2], use_binarization= False, LayerNorm_type=LayerNorm_type) for _ in
            range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[
            QRTransformerBlock(int(dim * (2 ** 3)), heads[3], use_binarization= False, LayerNorm_type=LayerNorm_type) for _ in
            range(num_blocks[3])])

        # Decoder with Upsampling layers
        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1)
        self.decoder_level3 = nn.Sequential(*[
            QRTransformerBlock(int(dim * (2 ** 2)), heads[2], use_binarization= False, LayerNorm_type=LayerNorm_type) for _ in
            range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1)
        self.decoder_level2 = nn.Sequential(*[
            QRTransformerBlock(int(dim * (2 ** 1)), heads[1], use_binarization= False, LayerNorm_type=LayerNorm_type) for _ in
            range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[
            QRTransformerBlock(int(dim * (2 ** 1)), heads[0], use_binarization= False, LayerNorm_type=LayerNorm_type) for _ in
            range(num_blocks[0])])

        self.refinement = nn.Sequential(
            *[QRTransformerBlock(dim * 2 ** 1, heads[0], use_binarization= False, LayerNorm_type=LayerNorm_type) for _ in
              range(num_refinement_blocks)])

        self.conv_out = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, padding=1)

        if self.apply_binarization:
            self.final_binarizer = DifferentiableBinarization(temperature=2.0)

    def forward(self, x):
        inp_enc_level1 = self.conv_in(x)  # Shape: [B, dim, H, W]
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)

        output = self.conv_out(out_dec_level1) + x
        
        #return output

        if self.apply_binarization:
            final_output = self.final_binarizer(output)
            return final_output
        else:
            return output
