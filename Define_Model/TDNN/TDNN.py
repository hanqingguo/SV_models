#!/usr/bin/env python
# encoding: utf-8

"""
@Author: yangwenhao
@Contact: 874681044@qq.com
@Software: PyCharm
@File: TDNN.py
@Time: 2019/8/28 上午10:54
@Overview: Implement TDNN

fork from:
https://github.com/jonasvdd/TDNN/blob/master/tdnn.py
"""

import math
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from Define_Model.FilterLayer import L2_Norm, Mean_Norm, TimeMaskLayer, FreqMaskLayer
from Define_Model.FilterLayer import fDLR, fBLayer, fBPLayer, fLLayer
from Define_Model.Pooling import AttentionStatisticPooling, StatisticPooling, GhostVLAD_v2, GhostVLAD_v3, \
    SelfAttentionPooling


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batchsize, num_channels, time_len = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, time_len)

    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, time_len)

    return x


"""Time Delay Neural Network as mentioned in the 1989 paper by Waibel et al. (Hinton) and the 2015 paper by Peddinti et al. (Povey)"""


class TimeDelayLayer_v1(nn.Module):
    def __init__(self, context, input_dim, output_dim, full_context=True):
        """
        Definition of context is the same as the way it's defined in the Peddinti paper. It's a list of integers, eg: [-2,2]
        By deault, full context is chosen, which means: [-2,2] will be expanded to [-2,-1,0,1,2] i.e. range(-2,3)
        """
        super(TimeDelayLayer_v1, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.check_valid_context(context)
        self.kernel_width, context = self.get_kernel_width(context, full_context)
        self.register_buffer('context',torch.LongTensor(context))
        self.full_context = full_context
        stdv = 1./math.sqrt(input_dim)
        self.kernel = nn.Parameter(torch.Tensor(output_dim, input_dim, self.kernel_width).normal_(0,stdv))
        self.bias = nn.Parameter(torch.Tensor(output_dim).normal_(0,stdv))
        # self.cuda_flag = False

    def forward(self, x):
        """
        x is one batch of data
        x.size(): [batch_size, sequence_length, input_dim]
        sequence length is the length of the input spectral data (number of frames) or if already passed through the convolutional network, it's the number of learned features
        output size: [batch_size, output_dim, len(valid_steps)]
        """
        # Check if parameters are cuda type and change context
        # if type(self.bias.data) == torch.cuda.FloatTensor and self.cuda_flag == False:
        #     self.context = self.context.cuda()
        #     self.cuda_flag = True
        conv_out = self.special_convolution(x, self.kernel, self.context, self.bias)
        return conv_out

    def special_convolution(self, x, kernel, context, bias):
        """
        This function performs the weight multiplication given an arbitrary context. Cannot directly use convolution because in case of only particular frames of context, one needs to select only those frames and perform a convolution across all batch items and all output dimensions of the kernel.
        """
        # pdb.set_trace()
        x = x.squeeze(1)
        input_size = x.size()

        assert len(input_size) == 3, 'Input tensor dimensionality is incorrect. Should be a 3D tensor'
        [batch_size, input_dim, input_sequence_length] = input_size
        #x = x.transpose(1,2).contiguous() # [batch_size, input_dim, input_length]

        # Allocate memory for output
        valid_steps = self.get_valid_steps(self.context, input_sequence_length)
        #xs = torch.Tensor(self.bias.data.new(batch_size, kernel.size()[0], len(valid_steps)))
        xs = torch.zeros((batch_size, kernel.size()[0], len(valid_steps)))

        if torch.cuda.is_available():
            xs = Variable(xs.cuda())
        # Perform the convolution with relevant input frames
        # pdb.set_trace()
        for c, i in enumerate(valid_steps):
            features = torch.index_select(x, 2, Variable(context+i))
            # torch.index_selec:
            # Returns a new tensor which indexes the input tensor along dimension dim using the entries in index which is a LongTensor.
            # The returned tensor has the same number of dimensions as the original tensor (input). The dim th dimension has the same
            # size as the length of index; other dimensions have the same size as in the original tensor.
            xs[:,:,c] = F.conv1d(features, kernel, bias=bias)[:,:,0]

        return xs

    @staticmethod
    def check_valid_context(context): #检查context是否合理
        # here context is still a list
        assert context[0] <= context[-1], 'Input tensor dimensionality is incorrect. Should be a 3D tensor'

    @staticmethod
    def get_kernel_width(context, full_context):
        if full_context:
            context = range(context[0],context[-1]+1) #确定一个context的范围
        return len(context), context

    @staticmethod
    def get_valid_steps(context, input_sequence_length):
        """
        Return the valid index frames considering the context.
        确定给定长度的序列，卷积之后的长度，及其帧
        :param context:
        :param input_sequence_length:
        :return:
        """

        start = 0 if context[0] >= 0 else -1*context[0]
        end = input_sequence_length if context[-1] <= 0 else input_sequence_length - context[-1]
        return range(start, end)


# Implement of 'https://github.com/cvqluu/TDNN/blob/master/tdnn.py'
class TimeDelayLayer_v2(nn.Module):

    def __init__(self, input_dim=40, output_dim=512, context_size=5, stride=1, dilation=1,
                 batch_norm=True, dropout_p=0.0, activation='relu'):
        '''
        TDNN as defined by https://www.danielpovey.com/files/2015_interspeech_multisplice.pdf
        Affine transformation not applied globally to all frames but smaller windows with local context
        batch_norm: True to include batch normalisation after the non linearity

        Context size and dilation determine the frames selected
        (although context size is not really defined in the traditional sense)
        For example:
            context size 5 and dilation 1 is equivalent to [-2,-1,0,1,2]
            context size 3 and dilation 2 is equivalent to [-2, 0, 2]
            context size 1 and dilation 1 is equivalent to [0]
        '''
        super(TimeDelayLayer_v2, self).__init__()
        self.context_size = context_size
        self.stride = stride
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dilation = dilation
        self.dropout_p = dropout_p
        self.batch_norm = batch_norm

        self.kernel = nn.Linear(input_dim * context_size, output_dim)
        if activation == 'relu':
            self.nonlinearity = nn.ReLU()
        elif activation == 'leakyrelu':
            self.nonlinearity = nn.LeakyReLU()

        if self.batch_norm:
            self.bn = nn.BatchNorm1d(output_dim)
            self.bn.weight.data.fill_(1)
            self.bn.bias.data.zero_()
        if self.dropout_p:
            self.drop = nn.Dropout(p=self.dropout_p)

    def set_dropout(self, dropout_p):
        self.dropout_p = dropout_p
        self.drop.p = self.dropout_p

    def forward(self, x):
        '''
        input: size (batch, seq_len, input_features)
        outpu: size (batch, new_seq_len, output_features)
        '''

        _, _, d = x.shape
        assert (d == self.input_dim), 'Input dimension was wrong. Expected ({}), got ({}) in ({})'.format(
            self.input_dim, d, str(x.shape))
        x = x.unsqueeze(1)

        # Unfold input into smaller temporal contexts
        x = F.unfold(
            x,
            (self.context_size, self.input_dim),
            stride=(1, self.input_dim),
            dilation=(self.dilation, 1)
        )

        # N, output_dim*context_size, new_t = x.shape
        x = x.transpose(1, 2)
        x = self.kernel(x)

        x = self.nonlinearity(x)

        if self.batch_norm:
            x = x.transpose(1, 2)
            x = self.bn(x)
            x = x.transpose(1, 2)

        if self.dropout_p:
            x = self.drop(x)

        return x


class TimeDelayLayer_v3(nn.Module):

    def __init__(self, input_dim=23, output_dim=512, context_size=5, stride=1, context=[-2, -1, 0, 1, 2],
                 batch_norm=True, dropout_p=0.0, activation='relu'):
        '''
        TDNN as defined by https://www.danielpovey.com/files/2015_interspeech_multisplice.pdf
        Affine transformation not applied globally to all frames but smaller windows with local context
        batch_norm: True to include batch normalisation after the non linearity

        Context size and dilation determine the frames selected
        (although context size is not really defined in the traditional sense)
        For example:
            context size 5 and dilation 1 is equivalent to [-2,-1,0,1,2]
            context size 3 and dilation 2 is equivalent to [-2, 0, 2]
            context size 1 and dilation 1 is equivalent to [0]
        '''
        super(TimeDelayLayer_v3, self).__init__()
        self.context_size = context_size
        self.stride = stride
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context = torch.tensor(context) + int((context_size - 1) / 2)
        self.dropout_p = dropout_p
        self.batch_norm = batch_norm

        self.kernel = nn.Linear(input_dim * len(context), output_dim)
        if activation == 'relu':
            self.nonlinearity = nn.ReLU()
        elif activation == 'leakyrelu':
            self.nonlinearity = nn.LeakyReLU()

        if self.batch_norm:
            self.bn = nn.BatchNorm1d(output_dim)
            self.bn.weight.data.fill_(1)
            self.bn.bias.data.zero_()
        if self.dropout_p:
            self.drop = nn.Dropout(p=self.dropout_p)

    def set_dropout(self, dropout_p):
        self.dropout_p = dropout_p
        self.drop.p = self.dropout_p

    def forward(self, x):
        '''
        input: size (batch, seq_len, input_features)
        outpu: size (batch, new_seq_len, output_features)
        '''

        b, l, d = x.shape
        assert (d == self.input_dim), 'Input dimension was wrong. Expected ({}), got ({}) in ({})'.format(
            self.input_dim, d, str(x.shape))
        x = x.unsqueeze(1)

        # Unfold input into smaller temporal contexts
        x = F.unfold(
            x,
            (self.context_size, self.input_dim),
            stride=(1, self.input_dim),
            dilation=(1, 1)
        )
        if self.context.shape[0] != self.context_size:
            _, _, new_t = x.shape

            x = x.view(b, new_t, -1, self.context_size, 1)
            x = x.index_select(3, self.context)
            x = x.view(b, -1, new_t)

        x = x.transpose(1, 2)
        x = self.kernel(x)

        x = self.nonlinearity(x)

        if self.batch_norm:
            x = x.transpose(1, 2)
            x = self.bn(x)
            x = x.transpose(1, 2)

        if self.dropout_p:
            x = self.drop(x)

        return x


# My implement TDNN using 2dConv Layer
class TimeDelayLayer_v4(nn.Module):

    def __init__(self, input_dim=23, output_dim=512, context_size=5, stride=1, dilation=1,
                 batch_norm=True, dropout_p=0.0, activation='relu'):
        '''
        TDNN as defined by https://www.danielpovey.com/files/2015_interspeech_multisplice.pdf
        Affine transformation not applied globally to all frames but smaller windows with local context
        batch_norm: True to include batch normalisation after the non linearity

        Context size and dilation determine the frames selected
        (although context size is not really defined in the traditional sense)
        For example:
            context size 5 and dilation 1 is equivalent to [-2,-1,0,1,2]
            context size 3 and dilation 2 is equivalent to [-2, 0, 2]
            context size 1 and dilation 1 is equivalent to [0]
        '''
        super(TimeDelayLayer_v4, self).__init__()
        self.context_size = context_size
        self.stride = stride
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dilation = dilation
        self.dropout_p = dropout_p
        self.batch_norm = batch_norm

        self.tdnn_layer = nn.Conv2d(1, output_dim, kernel_size=(context_size, input_dim), padding=(0, 0),
                                    dilation=(self.dilation, 1))
        if activation == 'relu':
            self.nonlinearity = nn.ReLU()
        elif activation == 'leakyrelu':
            self.nonlinearity = nn.LeakyReLU()
        elif activation == 'prelu':
            self.nonlinearity = nn.PReLU()

        if self.batch_norm:
            self.bn = nn.BatchNorm2d(output_dim)

        if self.dropout_p:
            self.drop = nn.Dropout(p=self.dropout_p)

    def set_dropout(self, dropout_p):
        self.dropout_p = dropout_p
        self.drop.p = self.dropout_p

    def forward(self, x):
        '''
        input: size (batch, seq_len, input_features)
        outpu: size (batch, new_seq_len, output_features)
        '''
        assert len(x.shape) == 4, print(x.shape)
        b, c, l, d = x.shape
        assert (d == self.input_dim), 'Input dimension ({})'.format(str(x.shape))

        x = self.tdnn_layer(x)
        x = self.nonlinearity(x)
        if self.batch_norm:
            x = self.bn(x)

        if self.dropout_p:
            x = self.drop(x)

        return x.transpose(1, 3)

# My implement TDNN using 1dConv Layer
class TimeDelayLayer_v5(nn.Module):

    def __init__(self, input_dim=23, output_dim=512, context_size=5, stride=1, dilation=1,
                 dropout_p=0.0, padding=0, groups=1, activation='relu'):
        super(TimeDelayLayer_v5, self).__init__()
        self.context_size = context_size
        self.stride = stride
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dilation = dilation
        self.dropout_p = dropout_p
        self.padding = padding
        self.groups = groups

        self.kernel = nn.Conv1d(self.input_dim, self.output_dim, self.context_size, stride=self.stride,
                                padding=self.padding, dilation=self.dilation, groups=self.groups)

        if activation == 'relu':
            self.nonlinearity = nn.ReLU(inplace=True)
        elif activation in ['leakyrelu', 'leaky_relu']:
            self.nonlinearity = nn.LeakyReLU()
        elif activation == 'prelu':
            self.nonlinearity = nn.PReLU()

        self.bn = nn.BatchNorm1d(output_dim)

        # self.drop = nn.Dropout(p=self.dropout_p)
    def forward(self, x):
        '''
        input: size (batch, seq_len, input_features)
        outpu: size (batch, new_seq_len, output_features)
        '''
        # _, _, d = x.shape
        # assert (d == self.input_dim), 'Input dimension was wrong. Expected ({}), got ({})'.format(
        #     self.input_dim, d)
        x = self.kernel(x.transpose(1, 2))
        x = self.nonlinearity(x)
        x = self.bn(x)

        return x.transpose(1, 2)


class TimeDelayLayer_v6(nn.Module):

    def __init__(self, input_dim=23, output_dim=512, context_size=5, stride=1, dilation=1,
                 batch_norm=True, dropout_p=0.0, padding=0, groups=1, activation='relu'):
        super(TimeDelayLayer_v6, self).__init__()
        self.context_size = context_size
        self.stride = stride
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dilation = dilation
        self.dropout_p = dropout_p
        self.padding = padding
        self.groups = groups

        self.kernel = nn.Sequential(nn.Conv1d(self.input_dim, self.output_dim,
                                              self.context_size, stride=self.stride,
                                              padding=self.padding, dilation=self.dilation, groups=self.groups))

        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation == 'leakyrelu':
            act_fn = nn.LeakyReLU
        elif activation == 'prelu':
            act_fn = nn.PReLU

        self.kernel.add_module('tdnn_act', act_fn())

        if batch_norm:
            self.kernel.add_module('tdnn_bn', nn.BatchNorm1d(output_dim))

    def forward(self, x):
        '''
        input: size (batch, seq_len, input_features)
        outpu: size (batch, new_seq_len, output_features)
        '''

        # x = self.kernel(x.transpose(1, 2))
        x = self.kernel(x)

        return x  # .transpose(1, 2)



class Conv2DLayer(nn.Module):

    def __init__(self, input_dim=40, output_dim=512, context_size=5, stride=1, dilation=1,
                 batch_norm=True, dropout_p=0.0, padding=0, groups=8, activation='relu'):
        super(Conv2DLayer, self).__init__()
        self.context_size = context_size
        self.stride = stride
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dilation = dilation
        self.dropout_p = dropout_p
        self.padding = padding
        self.groups = groups

        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=5, stride=(2, 1), padding=(2, 2)),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(),
                                   nn.Conv2d(32, 64, kernel_size=5, stride=(2, stride), padding=(2, 2)),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU())
        concat_channels = int(np.ceil(input_dim / 4) * 64)

        real_group = 196. * input_dim / (1625 - 5 * input_dim)
        if int(2 ** np.ceil(np.log2(real_group))) > groups:
            groups = min(int(2 ** np.ceil(np.log2(real_group))), 64)
            print('number of Group is set to %d' % groups)
        self.conv2 = nn.Sequential(
            nn.Conv1d(concat_channels, output_dim, kernel_size=1, stride=1, groups=groups, bias=False),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        '''
        input: size (batch, seq_len, input_features)
        outpu: size (batch, new_seq_len, output_features)
        '''
        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        x = self.conv1(x.transpose(2, 3))
        x_shape = x.shape
        x = x.reshape((x_shape[0], -1, x_shape[-1]))
        x = self.conv2(x)
        return x.transpose(1, 2)


class ShuffleTDLayer(nn.Module):

    def __init__(self, input_dim=23, output_dim=512, context_size=5, stride=1, dilation=1,
                 dropout_p=0.0, padding=0, groups=1, activation='relu', ) -> None:
        super(ShuffleTDLayer, self).__init__()
        self.context_size = context_size
        self.stride = stride
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dilation = dilation
        self.dropout_p = dropout_p
        self.padding = padding
        self.groups = groups

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = self.output_dim // 2
        assert (self.stride != 1) or (self.input_dim == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(self.input_dim, self.input_dim, kernel_size=3, stride=self.stride, padding=1,
                                    dilation=dilation),
                nn.BatchNorm1d(self.input_dim),
                nn.Conv1d(self.input_dim, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm1d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv1d(self.input_dim if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=self.context_size,
                                dilation=self.dilation, stride=self.stride, padding=1),
            nn.BatchNorm1d(branch_features),
            nn.Conv1d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i: int, o: int, kernel_size: int, stride: int = 1, padding: int = 0,
                       dilation: int = 1, bias: bool = False) -> nn.Conv1d:
        return nn.Conv1d(i, o, kernel_size, stride, padding, dilation=dilation, bias=bias, groups=i)

    def forward(self, x):
        x = x.transpose(1, 2)
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out.transpose(1, 2)


class TDNN_v1(nn.Module):
    def __init__(self, context, input_dim, output_dim, node_num, full_context):
        super(TDNN_v1, self).__init__()
        self.tdnn1 = TimeDelayLayer_v1(context[0], input_dim, node_num[0], full_context[0])
        self.tdnn2 = TimeDelayLayer_v1(context[1], node_num[0], node_num[1], full_context[1])
        self.tdnn3 = TimeDelayLayer_v1(context[2], node_num[1], node_num[2], full_context[2])
        self.tdnn4 = TimeDelayLayer_v1(context[3], node_num[2], node_num[3], full_context[3])
        self.tdnn5 = TimeDelayLayer_v1(context[4], node_num[3], node_num[4], full_context[4])
        self.fc1 = nn.Linear(node_num[5], node_num[6])
        self.fc2 = nn.Linear(node_num[6], node_num[7])
        self.fc3 = nn.Linear(node_num[7], output_dim)
        self.batch_norm1 = nn.BatchNorm1d(node_num[0])
        self.batch_norm2 = nn.BatchNorm1d(node_num[1])
        self.batch_norm3 = nn.BatchNorm1d(node_num[2])
        self.batch_norm4 = nn.BatchNorm1d(node_num[3])
        self.batch_norm5 = nn.BatchNorm1d(node_num[4])
        self.batch_norm6 = nn.BatchNorm1d(node_num[6])
        self.batch_norm7 = nn.BatchNorm1d(node_num[7])
        self.input_dim = input_dim
        self.output_dim = output_dim

        for m in self.modules():  # 对于各层参数的初始化
            if isinstance(m, nn.BatchNorm1d):  # weight设置为1，bias为0
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def statistic_pooling(self, x):
        mean_x = x.mean(dim=2)
        std_x = x.std(dim=2)
        mean_std = torch.cat((mean_x, std_x), 1)
        return mean_std

    def pre_forward(self, x):
        a1 = F.relu(self.batch_norm1(self.tdnn1(x)))
        a2 = F.relu(self.batch_norm2(self.tdnn2(a1)))
        a3 = F.relu(self.batch_norm3(self.tdnn3(a2)))
        a4 = F.relu(self.batch_norm4(self.tdnn4(a3)))
        a5 = F.relu(self.batch_norm5(self.tdnn5(a4)))

        a6 = self.statistic_pooling(a5)
        x_vectors = F.relu(self.batch_norm6(self.fc1(a6)))

        return x_vectors

    def forward(self, x):
        # a7 = self.pre_forward(x)
        a8 = F.relu(self.batch_norm7(self.fc2(x)))
        output = self.fc3(a8)
        return output


class TDNN_v2(nn.Module):
    def __init__(self, num_classes, embedding_size, input_dim, alpha=0., input_norm='',
                 dropout_p=0.0, encoder_type='STAP', **kwargs):
        super(TDNN_v2, self).__init__()
        self.num_classes = num_classes
        self.dropout_p = dropout_p
        self.input_dim = input_dim
        self.alpha = alpha

        if input_norm == 'Instance':
            self.inst_layer = nn.InstanceNorm1d(input_dim)
        elif input_norm == 'Mean':
            self.inst_layer = Mean_Norm()
        else:
            self.inst_layer = None

        self.frame1 = TimeDelayLayer_v2(input_dim=self.input_dim, output_dim=512, context_size=5, dilation=1)
        self.frame2 = TimeDelayLayer_v2(input_dim=512, output_dim=512, context_size=3, dilation=2)
        self.frame3 = TimeDelayLayer_v2(input_dim=512, output_dim=512, context_size=3, dilation=3)
        self.frame4 = TimeDelayLayer_v2(input_dim=512, output_dim=512, context_size=1, dilation=1)
        self.frame5 = TimeDelayLayer_v2(input_dim=512, output_dim=1500, context_size=1, dilation=1)
        self.drop = nn.Dropout(p=self.dropout_p)
        if encoder_type == 'STAP':
            self.encoder = StatisticPooling(input_dim=1500)
        elif encoder_type == 'SASP':
            self.encoder = AttentionStatisticPooling(input_dim=1500, hidden_dim=512)
        else:
            raise ValueError(encoder_type)

        self.segment6 = nn.Sequential(
            nn.Linear(3000, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512)
        )

        self.segment7 = nn.Sequential(
            nn.Linear(512, embedding_size),
            nn.ReLU(),
            nn.BatchNorm1d(embedding_size)
        )

        if self.alpha:
            self.l2_norm = L2_Norm(self.alpha)

        self.classifier = nn.Linear(embedding_size, num_classes)
        # self.bn = nn.BatchNorm1d(num_classes)

        for m in self.modules():  # 对于各层参数的初始化
            if isinstance(m, nn.BatchNorm1d):  # weight设置为1，bias为0
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, TimeDelayLayer_v2):
                # nn.init.normal(m.kernel.weight, mean=0., std=1.)
                nn.init.kaiming_normal_(m.kernel.weight, mode='fan_out', nonlinearity='relu')

    def set_global_dropout(self, dropout_p):
        self.dropout_p = dropout_p
        self.drop.p = dropout_p

    def forward(self, x):
        # pdb.set_trace()
        x = x.squeeze(1).float()
        if self.inst_layer != None:
            x = self.inst_layer(x)
        x = self.frame1(x)
        x = self.frame2(x)
        x = self.frame3(x)
        x = self.frame4(x)
        x = self.frame5(x)

        if self.dropout_p:
            x = self.drop(x)

        # print(x.shape)
        x = self.encoder(x)

        x = self.segment6(x)
        embedding_b = self.segment7(x)

        if self.alpha:
            embedding_b = self.l2_norm(embedding_b)

        logits = self.classifier(embedding_b)
        # logits = self.out_act(x)

        return logits, embedding_b


class TDNN_v4(nn.Module):
    def __init__(self, num_classes, embedding_size, input_dim, alpha=0., input_norm='',
                 dropout_p=0.0, encoder_type='STAP', **kwargs):
        super(TDNN_v4, self).__init__()
        self.num_classes = num_classes
        self.dropout_p = dropout_p
        self.input_dim = input_dim
        self.alpha = alpha

        if input_norm == 'Instance':
            self.inst_layer = nn.InstanceNorm1d(input_dim)
        elif input_norm == 'Mean':
            self.inst_layer = Mean_Norm()
        else:
            self.inst_layer = None

        self.frame1 = TimeDelayLayer_v4(input_dim=self.input_dim, output_dim=512, context_size=5, dilation=1)
        self.frame2 = TimeDelayLayer_v4(input_dim=512, output_dim=512, context_size=3, dilation=2)
        self.frame3 = TimeDelayLayer_v4(input_dim=512, output_dim=512, context_size=3, dilation=3)
        self.frame4 = TimeDelayLayer_v4(input_dim=512, output_dim=512, context_size=1, dilation=1)
        self.frame5 = TimeDelayLayer_v4(input_dim=512, output_dim=1500, context_size=1, dilation=1)
        self.drop = nn.Dropout(p=self.dropout_p)
        if encoder_type == 'STAP':
            self.encoder = StatisticPooling(input_dim=1500)
        elif encoder_type == 'SASP':
            self.encoder = AttentionStatisticPooling(input_dim=1500, hidden_dim=512)
        else:
            raise ValueError(encoder_type)

        self.segment6 = nn.Sequential(
            nn.Linear(3000, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512)
        )

        self.segment7 = nn.Sequential(
            nn.Linear(512, embedding_size),
            nn.ReLU(),
            nn.BatchNorm1d(embedding_size)
        )

        if self.alpha:
            self.l2_norm = L2_Norm(self.alpha)

        self.classifier = nn.Linear(embedding_size, num_classes)
        # self.bn = nn.BatchNorm1d(num_classes)

        for m in self.modules():  # 对于各层参数的初始化
            if isinstance(m, nn.BatchNorm1d):  # weight设置为1，bias为0
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, TimeDelayLayer_v2):
                # nn.init.normal(m.kernel.weight, mean=0., std=1.)
                nn.init.kaiming_normal_(m.kernel.weight, mode='fan_out', nonlinearity='relu')

    def set_global_dropout(self, dropout_p):
        self.dropout_p = dropout_p
        self.drop.p = dropout_p

    def forward(self, x):
        # pdb.set_trace()
        # x = x.squeeze(1).float()
        if self.inst_layer != None:
            x = self.inst_layer(x)

        x = self.frame1(x)
        x = self.frame2(x)
        x = self.frame3(x)
        x = self.frame4(x)
        x = self.frame5(x)

        if self.dropout_p:
            x = self.drop(x)

        # print(x.shape)
        x = self.encoder(x)
        embedding_a = self.segment6(x)
        embedding_b = self.segment7(embedding_a)

        if self.alpha:
            embedding_b = self.l2_norm(embedding_b)

        logits = self.classifier(embedding_b)

        return logits, embedding_b


class TDNN_v5(nn.Module):
    def __init__(self, num_classes, embedding_size, input_dim, alpha=0., input_norm='',
                 filter=None, sr=16000, feat_dim=64, exp=False, filter_fix=False,
                 dropout_p=0.0, dropout_layer=False, encoder_type='STAP',
                 num_classes_b=0, block_type='basic', first_2d=False, stride=[1],
                 mask='None', mask_len=20, channels=[512, 512, 512, 512, 1500], **kwargs):
        super(TDNN_v5, self).__init__()
        self.num_classes = num_classes
        self.num_classes_b = num_classes_b
        self.dropout_p = dropout_p
        self.dropout_layer = dropout_layer
        self.input_dim = input_dim
        self.channels = channels
        self.alpha = alpha
        self.mask = mask
        self.filter = filter
        self.feat_dim = feat_dim
        self.block_type = block_type.lower()
        self.stride = stride
        if len(self.stride) == 1:
            while len(self.stride) < 4:
                self.stride.append(self.stride[0])
        if np.sum((self.stride)) > 4:
            print('The stride for tdnn layers are: ', str(self.stride))

        if self.filter == 'fDLR':
            self.filter_layer = fDLR(input_dim=input_dim, sr=sr, num_filter=feat_dim, exp=exp, filter_fix=filter_fix)
        elif self.filter == 'fBLayer':
            self.filter_layer = fBLayer(input_dim=input_dim, sr=sr, num_filter=feat_dim, exp=exp, filter_fix=filter_fix)
        elif self.filter == 'fBPLayer':
            self.filter_layer = fBPLayer(input_dim=input_dim, sr=sr, num_filter=feat_dim, exp=exp,
                                         filter_fix=filter_fix)
        elif self.filter == 'fLLayer':
            self.filter_layer = fLLayer(input_dim=input_dim, num_filter=feat_dim, exp=exp)
        elif self.filter == 'Avg':
            self.filter_layer = nn.AvgPool2d(kernel_size=(1, 7), stride=(1, 3))
        else:
            self.filter_layer = None

        if input_norm == 'Instance':
            self.inst_layer = nn.InstanceNorm1d(input_dim)
        elif input_norm == 'Mean':
            self.inst_layer = Mean_Norm()
        else:
            self.inst_layer = None

        if self.mask == "time":
            self.maks_layer = TimeMaskLayer(mask_len=mask_len)
        elif self.mask == "freq":
            self.mask = FreqMaskLayer(mask_len=mask_len)
        elif self.mask == "time_freq":
            self.mask_layer = nn.Sequential(
                TimeMaskLayer(mask_len=mask_len),
                FreqMaskLayer(mask_len=mask_len)
            )
        else:
            self.mask_layer = None

        if self.filter_layer != None:
            self.input_dim = feat_dim
        if self.block_type == 'basic':
            TDlayer = TimeDelayLayer_v5
        elif self.block_type == 'basic_v6':
            TDlayer = TimeDelayLayer_v6
        elif self.block_type == 'shuffle':
            TDlayer = ShuffleTDLayer
        else:
            raise ValueError(self.block_type)

        if not first_2d:
            self.frame1 = TimeDelayLayer_v5(input_dim=self.input_dim, output_dim=self.channels[0],
                                            context_size=5, stride=self.stride[0], dilation=1)
        else:
            self.frame1 = Conv2DLayer(input_dim=self.input_dim, output_dim=self.channels[0], stride=self.stride[0])
        self.frame2 = TDlayer(input_dim=self.channels[0], output_dim=self.channels[1],
                              context_size=3, stride=self.stride[1], dilation=2)
        self.frame3 = TDlayer(input_dim=self.channels[1], output_dim=self.channels[2],
                              context_size=3, stride=self.stride[2], dilation=3)
        self.frame4 = TDlayer(input_dim=self.channels[2], output_dim=self.channels[3],
                              context_size=1, stride=self.stride[0], dilation=1)
        self.frame5 = TimeDelayLayer_v5(input_dim=self.channels[3], output_dim=self.channels[4],
                                        context_size=1, stride=self.stride[3], dilation=1)

        self.drop = nn.Dropout(p=self.dropout_p)

        if encoder_type == 'STAP':
            self.encoder = StatisticPooling(input_dim=self.channels[4])
            self.encoder_output = self.channels[4] * 2
        elif encoder_type == 'ASP':
            self.encoder = AttentionStatisticPooling(input_dim=self.channels[4], hidden_dim=self.channels[4])
            self.encoder_output = self.channels[4] * 2
        elif encoder_type == 'SAP':
            self.encoder = SelfAttentionPooling(input_dim=self.channels[4], hidden_dim=self.channels[4])
            self.encoder_output = self.channels[4]
        elif encoder_type == 'Ghos_v3':
            self.encoder = GhostVLAD_v3(num_clusters=self.num_classes_b, gost=1, dim=self.channels[4])
            self.encoder_output = self.channels[4] * 2
        else:
            raise ValueError(encoder_type)

        self.segment6 = nn.Sequential(
            nn.Linear(self.encoder_output, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512)
        )

        self.segment7 = nn.Sequential(
            nn.Linear(512, embedding_size),
            nn.ReLU(),
            nn.BatchNorm1d(embedding_size)
        )

        if self.alpha:
            self.l2_norm = L2_Norm(self.alpha)

        self.classifier = nn.Linear(embedding_size, num_classes)
        # self.bn = nn.BatchNorm1d(num_classes)

        for m in self.modules():  # 对于各层参数的初始化
            if isinstance(m, nn.BatchNorm1d):  # weight设置为1，bias为0
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, TimeDelayLayer_v5):
                # nn.init.normal(m.kernel.weight, mean=0., std=1.)
                nn.init.kaiming_normal_(m.kernel.weight, mode='fan_out', nonlinearity='relu')

    def set_global_dropout(self, dropout_p):
        self.dropout_p = dropout_p
        self.drop.p = dropout_p

    def forward(self, x):
        # pdb.set_trace()
        if self.filter_layer != None:
            x = self.filter_layer(x)

        if len(x.shape) == 4:
            x = x.squeeze(1).float()

        if self.inst_layer != None:
            x = self.inst_layer(x)

        if self.mask_layer != None:
            x = self.mask_layer(x)

        x = self.frame1(x)
        x = self.frame2(x)
        x = self.frame3(x)
        x = self.frame4(x)
        x = self.frame5(x)

        if self.dropout_layer:
            x = self.drop(x)

        # print(x.shape)
        x = self.encoder(x)
        embedding_a = self.segment6(x)
        embedding_b = self.segment7(embedding_a)

        if self.alpha:
            embedding_b = self.l2_norm(embedding_b)

        logits = self.classifier(embedding_b)

        return logits, embedding_b

    def xvector(self, x):
        # pdb.set_trace()
        if self.filter_layer != None:
            x = self.filter_layer(x)

        if len(x.shape) == 4:
            x = x.squeeze(1).float()

        if self.inst_layer != None:
            x = self.inst_layer(x)

        if self.mask_layer != None:
            x = self.mask_layer(x)
        # x = x.transpose(1, 2)
        x = self.frame1(x)
        x = self.frame2(x)
        x = self.frame3(x)
        x = self.frame4(x)
        x = self.frame5(x)

        if self.dropout_layer:
            x = self.drop(x)

        # print(x.shape)
        # x = self.encoder(x.transpose(1, 2))
        x = self.encoder(x)

        embedding_a = self.segment6[0](x)

        return "", embedding_a


class TDNN_v6(nn.Module):
    def __init__(self, num_classes, embedding_size, input_dim, alpha=0., input_norm='',
                 dropout_p=0.0, dropout_layer=False, encoder_type='STAP',
                 mask='None', mask_len=20, **kwargs):
        super(TDNN_v6, self).__init__()
        self.num_classes = num_classes
        self.dropout_p = dropout_p
        self.dropout_layer = dropout_layer
        self.input_dim = input_dim
        self.alpha = alpha
        self.mask = mask

        if input_norm == 'Instance':
            self.inst_layer = nn.InstanceNorm1d(input_dim)
        elif input_norm == 'Mean':
            self.inst_layer = Mean_Norm()
        else:
            self.inst_layer = None

        if self.mask == "time":
            self.maks_layer = TimeMaskLayer(mask_len=mask_len)
        elif self.mask == "freq":
            self.mask = FreqMaskLayer(mask_len=mask_len)
        elif self.mask == "time_freq":
            self.mask_layer = nn.Sequential(
                TimeMaskLayer(mask_len=mask_len),
                FreqMaskLayer(mask_len=mask_len)
            )
        else:
            self.mask_layer = None

        self.frame1 = TimeDelayLayer_v6(input_dim=self.input_dim, output_dim=512, context_size=5,
                                        dilation=1)
        self.frame2 = TimeDelayLayer_v6(input_dim=512, output_dim=512, context_size=3,
                                        dilation=2)
        self.frame3 = TimeDelayLayer_v6(input_dim=512, output_dim=512, context_size=3,
                                        dilation=3)
        self.frame4 = TimeDelayLayer_v6(input_dim=512, output_dim=512, context_size=1,
                                        dilation=1)
        self.frame5 = TimeDelayLayer_v6(input_dim=512, output_dim=1500, context_size=1,
                                        dilation=1)

        self.drop = nn.Dropout(p=self.dropout_p)

        if encoder_type == 'STAP':
            self.encoder = StatisticPooling(input_dim=1500)
        elif encoder_type == 'SASP':
            self.encoder = AttentionStatisticPooling(input_dim=1500, hidden_dim=512)
        else:
            raise ValueError(encoder_type)

        self.segment6 = nn.Sequential(
            nn.Linear(3000, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512)
        )

        self.segment7 = nn.Sequential(
            nn.Linear(512, embedding_size),
            nn.ReLU(),
            nn.BatchNorm1d(embedding_size)
        )

        if self.alpha:
            self.l2_norm = L2_Norm(self.alpha)

        self.classifier = nn.Linear(embedding_size, num_classes)
        # self.bn = nn.BatchNorm1d(num_classes)

        for m in self.modules():  # 对于各层参数的初始化
            if isinstance(m, nn.BatchNorm1d):  # weight设置为1，bias为0
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, TimeDelayLayer_v6):
                # nn.init.normal(m.kernel.weight, mean=0., std=1.)
                nn.init.kaiming_normal_(m.kernel[0].weight, mode='fan_out', nonlinearity='relu')

    def set_global_dropout(self, dropout_p):
        self.dropout_p = dropout_p
        self.drop.p = dropout_p

    def forward(self, x):
        # pdb.set_trace()
        if len(x.shape) == 4:
            x = x.squeeze(1).float()

        if self.inst_layer != None:
            x = self.inst_layer(x)

        if self.mask_layer != None:
            x = self.mask_layer(x)

        x = self.frame1(x)
        x = self.frame2(x)
        x = self.frame3(x)
        x = self.frame4(x)
        x = self.frame5(x)

        if self.dropout_layer:
            x = self.drop(x)

        # print(x.shape)
        x = self.encoder(x)
        embedding_a = self.segment6(x)
        embedding_b = self.segment7(embedding_a)

        if self.alpha:
            embedding_b = self.l2_norm(embedding_b)

        logits = self.classifier(embedding_b)

        return logits, embedding_b