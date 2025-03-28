"""
DQN网络模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DQN(nn.Module):
    """
    深度Q网络
    """
    def __init__(self, input_shape, n_actions):
        """
        初始化DQN网络
        
        Args:
            input_shape: 输入状态形状 (C, H, W)
            n_actions: 动作数量
        """
        super(DQN, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        conv_out_size = self._get_conv_output(input_shape)
        
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        
    def _get_conv_output(self, shape):
        """
        计算卷积层输出大小
        """
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
        
    def forward(self, x):
        """
        前向传播
        """
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


class DuelingDQN(nn.Module):
    """
    Dueling DQN网络
    """
    def __init__(self, input_shape, n_actions):
        """
        初始化Dueling DQN网络
        
        Args:
            input_shape: 输入状态形状 (C, H, W)
            n_actions: 动作数量
        """
        super(DuelingDQN, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        conv_out_size = self._get_conv_output(input_shape)
        
        # 价值流
        self.value_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
        # 优势流
        self.advantage_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        
    def _get_conv_output(self, shape):
        """
        计算卷积层输出大小
        """
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
        
    def forward(self, x):
        """
        前向传播
        """
        conv_out = self.conv(x).view(x.size()[0], -1)
        
        # 计算状态价值
        value = self.value_stream(conv_out)
        
        # 计算动作优势
        advantage = self.advantage_stream(conv_out)
        
        # 合并价值和优势
        # Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum(A(s,a')))
        return value + advantage - advantage.mean(dim=1, keepdim=True)
