"""
PPO网络模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical


class PPONetwork(nn.Module):
    """
    PPO网络，包含共享的特征提取器、策略网络和价值网络
    """
    def __init__(self, input_shape, n_actions):
        """
        初始化PPO网络
        
        Args:
            input_shape: 输入状态形状 (C, H, W)
            n_actions: 动作数量
        """
        super(PPONetwork, self).__init__()
        
        # 共享特征提取器
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        conv_out_size = self._get_conv_output(input_shape)
        
        # 策略网络（演员）
        self.policy = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        
        # 价值网络（评论家）
        self.value = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
    def _get_conv_output(self, shape):
        """
        计算卷积层输出大小
        """
        o = self.features(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
        
    def forward(self, x):
        """
        前向传播，返回动作概率和状态价值
        """
        features = self.features(x)
        features = features.view(x.size(0), -1)
        
        # 计算动作概率
        action_logits = self.policy(features)
        action_probs = F.softmax(action_logits, dim=1)
        
        # 计算状态价值
        state_value = self.value(features)
        
        return action_probs, state_value
        
    def evaluate(self, x, action):
        """
        评估给定状态和动作，返回动作概率分布、动作对数概率、熵和状态价值
        """
        action_probs, state_value = self.forward(x)
        
        # 创建分类分布
        dist = Categorical(action_probs)
        
        # 计算动作对数概率
        action_log_probs = dist.log_prob(action)
        
        # 计算熵
        entropy = dist.entropy()
        
        return action_log_probs, state_value, entropy
        
    def act(self, x, deterministic=False):
        """
        根据给定状态选择动作，可选择确定性或随机性策略
        """
        with torch.no_grad():
            action_probs, state_value = self.forward(x)
            
            if deterministic:
                # 确定性策略：选择概率最高的动作
                action = torch.argmax(action_probs, dim=1)
            else:
                # 随机性策略：根据概率采样动作
                dist = Categorical(action_probs)
                action = dist.sample()
                
            # 计算动作对数概率
            log_prob = torch.log(action_probs.gather(1, action.unsqueeze(1)) + 1e-10).squeeze(1)
            
        return action, log_prob, state_value
