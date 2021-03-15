import paddle
from paddle import nn
from paddle import ParamAttr
from paddle.nn import functional as F

#-----------------------------------------------
#                Normal ConvBlock
#-----------------------------------------------
class Conv2dLayer(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, pad_type = 'zero', activation = 'lrelu', norm = 'none', sn = False):
        super(Conv2dLayer, self).__init__()
        # Initialize the padding scheme
        if pad_type == 'reflect':
            self.pad = nn.Pad2D([padding]*4,mode=pad_type)
        elif pad_type == 'replicate':
            self.pad = nn.Pad2D([padding]*4,mode=pad_type)
        elif pad_type == 'zero':
            self.pad = nn.Pad2D([padding]*4,mode='constant')
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)
        
        # Initialize the normalization type
        if norm == 'bn':
            self.norm = nn.BatchNorm2D(out_channels,momentum=0.1,weight_attr=False, bias_attr=False)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2D(out_channels,momentum=0.1,weight_attr=False, bias_attr=False)
        elif norm == 'ln':
            self.norm = nn.LayerNorm(out_channels,momentum=0.1,weight_attr=False, bias_attr=False) #LayerNorm(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)
        
        # Initialize the activation funtion
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # Initialize the convolution layers
        #self.sn_flag = sn
        #self.conv2d = nn.Conv2D(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation)
        if sn:
            self.conv2d = SpectralNorm(nn.Conv2D(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation))
        else:
            self.conv2d = nn.Conv2D(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation)
    
    def forward(self, x):
        x = self.pad(x)
        x = self.conv2d(x)
        # #if self.sn_flag:
        #     spectral_norm = paddle.nn.SpectralNorm(x.shape, dim=1, power_iters=2)
        #     x = spectral_norm(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class TransposeConv2dLayer(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, pad_type = 'zero', activation = 'lrelu', norm = 'none', sn = False, scale_factor = 2):
        super(TransposeConv2dLayer, self).__init__()
        # Initialize the conv scheme
        self.scale_factor = scale_factor
        self.conv2d = Conv2dLayer(in_channels, out_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, sn)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor = self.scale_factor, mode = 'nearest')
        x = self.conv2d(x)
        return x

#-----------------------------------------------
#                Gated ConvBlock
#-----------------------------------------------
class GatedConv2d(nn.Layer):
    def __init__(self,in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, pad_type = 'reflect', activation = 'lrelu', norm = 'none', sn = False):
        super(GatedConv2d, self).__init__()
        # Initialize the padding scheme
        if pad_type == 'reflect':
            self.pad = nn.Pad2D([padding]*4,mode=pad_type)
        elif pad_type == 'replicate':
            self.pad = nn.Pad2D([padding]*4,mode=pad_type)
        elif pad_type == 'zero':
            self.pad = nn.Pad2D([padding]*4,mode='constant')
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)
        
        # Initialize the normalization type
        if norm == 'bn':
            self.norm = nn.BatchNorm2D(out_channels,momentum=0.1,weight_attr=False,bias_attr=False)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2D(out_channels,momentum=0.1,weight_attr=False,bias_attr=False)
        elif norm == 'ln':
            self.norm = nn.LayerNorm(out_channels,momentum=0.1,weight_attr=False,bias_attr=False)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)
        
        # Initialize the activation funtion
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # Initialize the convolution layers

        # Use paddle.SpectralNorm
        # self.sn_flag = sn
        # self.sn_1 = None
        # self.sn_2 = None

        if sn:
            self.conv2d = SpectralNorm(nn.Conv2D(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation))
            self.mask_conv2d = SpectralNorm(nn.Conv2D(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation))
        else:
            self.conv2d = nn.Conv2D(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation)
            self.mask_conv2d = nn.Conv2D(in_channels, out_channels, kernel_size, stride, padding = 0, dilation = dilation)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.pad(x)

        # Use paddle.SpectralNorm
        # conv = self.conv2d(x)
        # mask = self.mask_conv2d(x)
        # if self.sn_flag:
        #     if self.sn_1 is None:
        #         self.sn_1 = paddle.nn.SpectralNorm(conv.shape, dim=1, power_iters=2)
        #         self.sn_2 = paddle.nn.SpectralNorm(mask.shape, dim=1, power_iters=2)
        #     conv = self.sn_1(x)
        #     mask = self.sn_2(x)

        conv = self.conv2d(x)
        mask = self.mask_conv2d(x)
        gated_mask = self.sigmoid(mask)
        x = conv * gated_mask
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class TransposeGatedConv2d(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, dilation = 1, pad_type = 'zero', activation = 'lrelu', norm = 'none', sn = True, scale_factor = 2):
        super(TransposeGatedConv2d, self).__init__()
        # Initialize the conv scheme
        self.scale_factor = scale_factor
        self.gated_conv2d = GatedConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, pad_type, activation, norm, sn)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor = self.scale_factor, mode = 'nearest')
        x = self.gated_conv2d(x)
        return x

# ----------------------------------------
#               Layer Norm
# ----------------------------------------
class LayerNorm(nn.Layer):
    def __init__(self, num_features, eps = 1e-8, affine = True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = self.create_parameter(paddle.uniform(num_features))
            self.beta = self.create_parameter(paddle.zeros(num_features))

    def forward(self, x):
        # layer norm
        shape = [-1] + [1] * (x.ndim - 1)                                  # for 4d input: [-1, 1, 1, 1]
        if x.shape(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.reshape((-1,)).mean().reshape(*shape)
            std = x.reshape((-1,)).std().reshape(*shape)
        else:
            mean = x.reshape(x.shape[0], -1).mean(1).reshape(*shape)
            std = x.reshape(x.shape(0), -1).std(1).reshape(*shape)
        x = (x - mean) / (std + self.eps)
        # if it is learnable
        if self.affine:
            shape = [1, -1] + [1] * (x.ndim - 2)                          # for 4d input: [1, -1, 1, 1]
            x = x * self.gamma.reshape(*shape) + self.beta.reshape(*shape)
        return x

#-----------------------------------------------
#                  SpectralNorm
#-----------------------------------------------
def l2normalize(v, eps = 1e-12):
    return v / (v.norm() + eps)

class SpectralNorm(nn.Layer):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.shape[0]
        for _ in range(self.power_iterations):
            v = l2normalize(paddle.mv(paddle.t(w.reshape((height,-1))), u))
            u = l2normalize(paddle.mv(w.reshape((height,-1)), v))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.reshape((height, -1)).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)
        w_ = w.numpy()
        height = w_.shape[0]
        width = w_.reshape(height,-1).shape[-1]

        u = self.create_parameter((height,))
        v = self.create_parameter((width,))
        u.data = l2normalize(u)
        v.data = l2normalize(v)
        w_bar = self.create_parameter(w.shape)

        del self.module._parameters[self.name]

        self.module.add_parameter(self.name + "_u",u)
        self.module.add_parameter(self.name + "_v",v)
        self.module.add_parameter(self.name + "_bar",w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)
