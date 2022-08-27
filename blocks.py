from fastai.vision.all import *
import fastai


def get_layer_name(layer, idx):
    if isinstance(layer, torch.nn.Conv2d):
        layer_name = 'Conv2d_{}_{}x{}'.format(
            idx, layer.in_channels, layer.out_channels
        )
    elif isinstance(layer, torch.nn.ConvTranspose2d):
        layer_name = 'ConvT2d_{}_{}x{}'.format(
            idx, layer.in_channels, layer.out_channels
        )
    elif isinstance(layer, torch.nn.BatchNorm2d):
        layer_name = 'BatchNorm2D_{}_{}'.format(
            idx, layer.num_features)
    elif isinstance(layer, torch.nn.Linear):
        layer_name = 'Linear_{}_{}x{}'.format(
            idx, layer.in_features, layer.out_features
        )
    elif isinstance(layer, fastai.layers.Identity):
        layer_name = 'Identity'
    else:
        layer_name = "Activation_{}".format(idx)
    return '_'.join(layer_name.split('_')[:2]).lower()
    
    
# convolutional block
class ConvBlock(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()

        layers = [
            nn.BatchNorm2d(n_in, momentum=0.01, eps=0.001),
            nn.ELU(),
            nn.Conv2d(n_in, n_out, kernel_size, stride, padding),
            nn.BatchNorm2d(n_out, momentum=0.01, eps=0.001),
            nn.ELU(),
            nn.Conv2d(n_out, n_out, kernel_size, stride, padding),
        ]
        self._init_block(layers)

    def forward(self, x):
        for layer in self.conv_block.values():
            x = layer(x)
        return x

    def _init_block(self, layers):
        self.conv_block = nn.ModuleDict()
        for idx, layer in enumerate(layers):
            self.conv_block.add_module(get_layer_name(layer, idx), layer)
            
# residual block
class Add(nn.Module):
    def __init__(self):
        super(Add, self).__init__()
        self.add = torch.add

    def forward(self, x1, x2):
        return self.add(x1, x2)


class IdentityPath(nn.Module):
    def __init__(self, n_in: int, n_out: int, is_conv: bool = True, upsample: bool = False):
        super(IdentityPath, self).__init__()

        self.is_conv = is_conv
        self.upsample = upsample

        if upsample:
            layer = nn.ConvTranspose2d(
                n_in, n_out, kernel_size=2, stride=2, padding=0)
        elif is_conv:
            layer = nn.Conv2d(n_in, n_out, kernel_size=1, padding=0)
        else:
            layer = Identity()
        self.layer_name = get_layer_name(layer, 1)
        self.add_module(self.layer_name, layer)

    def forward(self, x):
        return getattr(self, self.layer_name)(x)


class ResidualBlock(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, padding=1, is_conv=True):
        super(ResidualBlock, self).__init__()

        self.is_conv = is_conv
        self.conv_path = ConvBlock(
            n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding)
        self.id_path = IdentityPath(n_in, n_out, is_conv=is_conv)
        self.add = Add()

    def forward(self, x):
        conv_path = self.conv_path(x)
        short_connect = self.id_path(x)
        return self.add(conv_path, short_connect)
        
# bottleneck
class Bottleneck(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=5, stride=1, padding=2):
        super(Bottleneck, self).__init__()

        self.residual_block1 = ResidualBlock(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding,
                                             is_conv=True)
        self.residual_block2 = ResidualBlock(n_out, n_out, kernel_size=kernel_size, stride=stride, padding=padding,
                                             is_conv=False)

    def forward(self, x):
        return self.residual_block2(self.residual_block1(x))

        
# residual block (decode path)
class Concatenate(nn.Module):
    def __init__(self, dim):
        super(Concatenate, self).__init__()
        self.cat = partial(torch.cat, dim=dim)

    def forward(self, x):
        return self.cat(x)


class UpResidualBlock(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, padding=1, concat_dim=1):
        super(UpResidualBlock, self).__init__()
        self.id_path = nn.ModuleDict({
            "up_conv": nn.ConvTranspose2d(n_in, n_out, kernel_size=2, stride=2, padding=0),
            "concat": Concatenate(dim=concat_dim)
        })
        self.conv_path = ConvBlock(
            n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding)
        self.add = Add()

    def forward(self, x, long_connect):
        short_connect = self.id_path.up_conv(x)
        concat = self.id_path.concat([short_connect, long_connect])
        return self.add(self.conv_path(concat), short_connect)
        

# heatmap
class Heatmap(nn.Module):
    def __init__(self, n_in, n_out=1, kernel_size=1, stride=1, padding=0):
        super(Heatmap, self).__init__()
        self.conv2d = nn.Conv2d(n_in, n_out, kernel_size, stride, padding)
        self.act = nn.Sigmoid()

    def forward(self, x):
        return self.act(self.conv2d(x))
    

class Heatmap2d(nn.Module):
    def __init__(self, n_in, n_out=2, kernel_size=1, stride=1, padding=0, concat_dim=1):
        super(Heatmap2d, self).__init__()
        self.heatmap = Heatmap(n_in, n_out - 1, kernel_size, stride, padding)
        self.concat = Concatenate(dim=concat_dim)

    def forward(self, x):
        heatmap1 = self.heatmap(x)
        heatmap0 = torch.ones_like(heatmap1) - heatmap1
        return self.concat([heatmap0, heatmap1])
        

# cell resunet
class cResUnet(nn.Module):
    def __init__(self, n_features_start=16, n_out=2):
        super(cResUnet, self).__init__()
        pool_ks, pool_stride, pool_pad = 2, 2, 0

        self.encoder = nn.ModuleDict({
            # colorspace transform
            'colorspace': nn.Conv2d(3, 1, kernel_size=1, padding=0),

            # block 1
            'conv_block': ConvBlock(1, n_features_start),
            'pool1': nn.MaxPool2d(pool_ks, pool_stride, pool_pad),

            # block 2
            'residual_block1': ResidualBlock(n_features_start, 2 * n_features_start, is_conv=True),
            'pool2': nn.MaxPool2d(pool_ks, pool_stride, pool_pad),

            # block 3
            'residual_block2': ResidualBlock(2 * n_features_start, 4 * n_features_start, is_conv=True),
            'pool3': nn.MaxPool2d(pool_ks, pool_stride, pool_pad),

            # bottleneck
            'bottleneck': Bottleneck(4 * n_features_start, 8 * n_features_start, kernel_size=5, padding=2),
        })

        self.decoder = nn.ModuleDict({
            # block 6
            'upconv_block1': UpResidualBlock(n_in=8 * n_features_start, n_out=4 * n_features_start),

            # block 7
            'upconv_block2': UpResidualBlock(4 * n_features_start, 2 * n_features_start),

            # block 8
            'upconv_block3': UpResidualBlock(2 * n_features_start, n_features_start),
        })

        # output
        self.head = Heatmap2d(
            n_features_start, n_out, kernel_size=1, stride=1, padding=0)

    def _forward_impl(self, x: Tensor) -> Tensor:
        downblocks = []
        for lbl, layer in self.encoder.items():
            x = layer(x)
            if 'block' in lbl:
                downblocks.append(x)
        for layer, long_connect in zip(self.decoder.values(), reversed(downblocks)):
            x = layer(x, long_connect)
        return self.head(x)

    def init_kaiming_normal(self, mode='fan_in'):
        print('Initializing conv2d weights with Kaiming He normal')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode=mode)
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
