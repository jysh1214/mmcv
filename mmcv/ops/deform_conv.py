# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair, _single

from mmcv.utils import deprecated_api_warning
from ..cnn import CONV_LAYERS
from ..utils import ext_loader, print_log

ext_module = ext_loader.load_ext('_ext', [
    'deform_conv_backward_input',
    'deform_conv_backward_parameters'
])

torch.ops.load_library("/home/BEVDepth/mmcv/build/lib.linux-x86_64-3.9/deform_conv_forward.cpython-39-x86_64-linux-gnu.so")


def register_custom_op():
    def _deform_conv_forward(
        g,
        input,
        weight,
        offset,
        output,
        columns,
        ones,
        kW,
        kH,
        dW,
        dH,
        padW,
        padH,
        dilationW,
        dilationH,
        group,
        deformable_group,
        im2col_step,
    ):
        return g.op("sifive::DeformConv2d",
            input,
            weight,
            offset,
            output,
            columns,
            ones,
            kW,
            kH,
            dW,
            dH,
            padW,
            padH,
            dilationW,
            dilationH,
            group,
            deformable_group,
            im2col_step,
        )
    from torch.onnx import register_custom_op_symbolic
    register_custom_op_symbolic("mmcv::deform_conv_forward", _deform_conv_forward, 11)


register_custom_op()


def deform_conv2d(input: Tensor,
                  offset: Tensor,
                  weight: Tensor,
                  stride: Union[int, Tuple[int, ...]] = 1,
                  padding: Union[int, Tuple[int, ...]] = 0,
                  dilation: Union[int, Tuple[int, ...]] = 1,
                  groups: int = 1,
                  deform_groups: int = 1,
                  bias: bool = False,
                  im2col_step: int = 32) -> Tensor:
    if input is not None and input.dim() != 4:
        raise ValueError(f'Expected 4D tensor as input, got {input.dim()}D tensor instead.')
    assert bias is False, 'Only support bias is False.'
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    input = input.type_as(offset)
    weight = weight.type_as(input)

    def _output_size(padding, dilation, stride, input, weight):
        channels = weight.size(0)
        output_size = (input.size(0), channels)
        for d in range(input.dim() - 2):
            in_size = input.size(d + 2)
            pad = padding[d]
            kernel = dilation[d] * (weight.size(d + 2) - 1) + 1
            stride_ = stride[d]
            output_size += ((in_size + (2 * pad) - kernel) // stride_ + 1, )
        if not all(map(lambda s: s > 0, output_size)):
            raise ValueError(
                'convolution input is too small (output would be ' +
                'x'.join(map(str, output_size)) + ')')
        return output_size

    output = input.new_empty(_output_size(padding, dilation, stride, input, weight))
    bufs_ = [input.new_empty(0), input.new_empty(0)]  # columns, ones
    cur_im2col_step = min(im2col_step, input.size(0))
    assert (input.size(0) % cur_im2col_step) == 0, 'batch size must be divisible by im2col_step'
    return torch.ops.mmcv.deform_conv_forward(
        input,
        weight,
        offset,
        output,
        bufs_[0],
        bufs_[1],
        weight.size(3),
        weight.size(2),
        stride[1],
        stride[0],
        padding[1],
        padding[0],
        dilation[1],
        dilation[0],
        groups,
        deform_groups,
        cur_im2col_step,
    )


class DeformConv2d(nn.Module):
    r"""Deformable 2D convolution.

    Applies a deformable 2D convolution over an input signal composed of
    several input planes. DeformConv2d was described in the paper
    `Deformable Convolutional Networks
    <https://arxiv.org/pdf/1703.06211.pdf>`_

    Note:
        The argument ``im2col_step`` was added in version 1.3.17, which means
        number of samples processed by the ``im2col_cuda_kernel`` per call.
        It enables users to define ``batch_size`` and ``im2col_step`` more
        flexibly and solved `issue mmcv#1440
        <https://github.com/open-mmlab/mmcv/issues/1440>`_.

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size(int, tuple): Size of the convolving kernel.
        stride(int, tuple): Stride of the convolution. Default: 1.
        padding (int or tuple): Zero-padding added to both sides of the input.
            Default: 0.
        dilation (int or tuple): Spacing between kernel elements. Default: 1.
        groups (int): Number of blocked connections from input.
            channels to output channels. Default: 1.
        deform_groups (int): Number of deformable group partitions.
        bias (bool): If True, adds a learnable bias to the output.
            Default: False.
        im2col_step (int): Number of samples processed by im2col_cuda_kernel
            per call. It will work when ``batch_size`` > ``im2col_step``, but
            ``batch_size`` must be divisible by ``im2col_step``. Default: 32.
            `New in version 1.3.17.`
    """

    @deprecated_api_warning({'deformable_groups': 'deform_groups'},
                            cls_name='DeformConv2d')
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, ...]],
                 stride: Union[int, Tuple[int, ...]] = 1,
                 padding: Union[int, Tuple[int, ...]] = 0,
                 dilation: Union[int, Tuple[int, ...]] = 1,
                 groups: int = 1,
                 deform_groups: int = 1,
                 bias: bool = False,
                 im2col_step: int = 32) -> None:
        super().__init__()

        assert not bias, \
            f'bias={bias} is not supported in DeformConv2d.'
        assert in_channels % groups == 0, \
            f'in_channels {in_channels} cannot be divisible by groups {groups}'
        assert out_channels % groups == 0, \
            f'out_channels {out_channels} cannot be divisible by groups \
              {groups}'

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deform_groups = deform_groups
        self.im2col_step = im2col_step
        # enable compatibility with nn.Conv2d
        self.transposed = False
        self.output_padding = _single(0)

        # only weight, no bias
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // self.groups,
                         *self.kernel_size))

        self.reset_parameters()

    def reset_parameters(self):
        # switch the initialization of `self.weight` to the standard kaiming
        # method described in `Delving deep into rectifiers: Surpassing
        # human-level performance on ImageNet classification` - He, K. et al.
        # (2015), using a uniform distribution
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')

    def forward(self, x: Tensor, offset: Tensor) -> Tensor:
        """Deformable Convolutional forward function.

        Args:
            x (Tensor): Input feature, shape (B, C_in, H_in, W_in)
            offset (Tensor): Offset for deformable convolution, shape
                (B, deform_groups*kernel_size[0]*kernel_size[1]*2,
                H_out, W_out), H_out, W_out are equal to the output's.

                An offset is like `[y0, x0, y1, x1, y2, x2, ..., y8, x8]`.
                The spatial arrangement is like:

                .. code:: text

                    (x0, y0) (x1, y1) (x2, y2)
                    (x3, y3) (x4, y4) (x5, y5)
                    (x6, y6) (x7, y7) (x8, y8)

        Returns:
            Tensor: Output of the layer.
        """
        # To fix an assert error in deform_conv_cuda.cpp:128
        # input image is smaller than kernel
        input_pad = (x.size(2) < self.kernel_size[0]) or (x.size(3) <
                                                          self.kernel_size[1])
        if input_pad:
            pad_h = max(self.kernel_size[0] - x.size(2), 0)
            pad_w = max(self.kernel_size[1] - x.size(3), 0)
            x = F.pad(x, (0, pad_w, 0, pad_h), 'constant', 0).contiguous()
            offset = F.pad(offset, (0, pad_w, 0, pad_h), 'constant', 0)
            offset = offset.contiguous()
        out = deform_conv2d(x, offset, self.weight, self.stride, self.padding,
                            self.dilation, self.groups, self.deform_groups,
                            False, self.im2col_step)
        if input_pad:
            out = out[:, :, :out.size(2) - pad_h, :out.size(3) -
                      pad_w].contiguous()
        return out

    def __repr__(self):
        s = self.__class__.__name__
        s += f'(in_channels={self.in_channels},\n'
        s += f'out_channels={self.out_channels},\n'
        s += f'kernel_size={self.kernel_size},\n'
        s += f'stride={self.stride},\n'
        s += f'padding={self.padding},\n'
        s += f'dilation={self.dilation},\n'
        s += f'groups={self.groups},\n'
        s += f'deform_groups={self.deform_groups},\n'
        # bias is not supported in DeformConv2d.
        s += 'bias=False)'
        return s


@CONV_LAYERS.register_module('DCN')
class DeformConv2dPack(DeformConv2d):
    """A Deformable Conv Encapsulation that acts as normal Conv layers.

    The offset tensor is like `[y0, x0, y1, x1, y2, x2, ..., y8, x8]`.
    The spatial arrangement is like:

    .. code:: text

        (x0, y0) (x1, y1) (x2, y2)
        (x3, y3) (x4, y4) (x5, y5)
        (x6, y6) (x7, y7) (x8, y8)

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
    """

    _version = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deform_groups * 2 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=_pair(self.stride),
            padding=_pair(self.padding),
            dilation=_pair(self.dilation),
            bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        offset = self.conv_offset(x)
        return deform_conv2d(x, offset, self.weight, self.stride, self.padding,
                             self.dilation, self.groups, self.deform_groups,
                             False, self.im2col_step)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)

        if version is None or version < 2:
            # the key is different in early versions
            # In version < 2, DeformConvPack loads previous benchmark models.
            if (prefix + 'conv_offset.weight' not in state_dict
                    and prefix[:-1] + '_offset.weight' in state_dict):
                state_dict[prefix + 'conv_offset.weight'] = state_dict.pop(
                    prefix[:-1] + '_offset.weight')
            if (prefix + 'conv_offset.bias' not in state_dict
                    and prefix[:-1] + '_offset.bias' in state_dict):
                state_dict[prefix +
                           'conv_offset.bias'] = state_dict.pop(prefix[:-1] +
                                                                '_offset.bias')

        if version is not None and version > 1:
            print_log(
                f'DeformConv2dPack {prefix.rstrip(".")} is upgraded to '
                'version 2.',
                logger='root')

        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)
