"""
DQN智能体
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
from datetime import datetime

from models.dqn_model import DQN, DuelingDQN
from utils.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


class DQNAgent:
    """
    DQN智能体
    """
    def __init__(self, state_shape, action_space, device, config):
        """
        初始化DQN智能体
        
        Args:
            state_shape: 状态空间形状
            action_space: 动作空间
            device: 计算设备
            config: 配置参数
        """
        self.device = device
        self.action_space = action_space
        self.n_actions = action_space.n
        self.config = config
        
        # 创建网络
        if config["dueling_dqn"]:
            self.policy_net = DuelingDQN(state_shape, self.n_actions).to(device)
            self.target_net = DuelingDQN(state_shape, self.n_actions).to(device)
        else:
            self.policy_net = DQN(state_shape, self.n_actions).to(device)
            self.target_net = DQN(state_shape, self.n_actions).to(device)
            
        # 复制参数到目标网络
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # 目标网络不需要梯度
        
        # 创建优化器
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), 
            lr=config["learning_rate"]
        )
        
        # 创建经验回放缓冲区
        self.buffer = ReplayBuffer(config["buffer_size"], device)
        
        # 探索参数
        self.epsilon = config["epsilon_start"]
        self.epsilon_end = config["epsilon_end"]
        self.epsilon_decay = config["epsilon_decay"]
        
        # 其他参数
        self.gamma = config["gamma"]  # 折扣因子
        self.batch_size = config["batch_size"]
        self.target_update = config["target_update_interval"]
        self.learning_starts = config["learning_starts"]
        self.double_dqn = config["double_dqn"]
        
        # 训练计数器
        self.step_counter = 0
        self.update_counter = 0
        self.episode_counter = 0
        
        # 日志
        self.training_info = {
            "losses": [],
            "q_values": [],
            "epsilons": []
        }
        
    def select_action(self, state, evaluate=False):
        """
        选择动作，使用epsilon-greedy策略
        
        Args:
            state: 环境状态
            evaluate: 是否为评估模式
            
        Returns:
            action: 选择的动作
        """
        # 在评估模式下，使用贪婪策略
        if evaluate:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state)
                action = q_values.max(1)[1].item()
                return action
                
        # 计算当前探索率
        self.epsilon = self.epsilon_end + (self.epsilon - self.epsilon_end) * \
                       np.exp(-1. * self.step_counter / self.epsilon_decay)
                       
        # 随机探索
        if random.random() < self.epsilon:
            return self.action_space.sample()
            
        # 贪婪选择
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state)
            action = q_values.max(1)[1].item()
            
            # 记录Q值
            self.training_info["q_values"].append(q_values.max().item())
            self.training_info["epsilons"].append(self.epsilon)
            
            return action
            
    def store_transition(self, state, action, next_state, reward, done):
        """
        存储经验到回放缓冲区
        
        Args:
            state: 当前状态
            action: 执行的动作
            next_state: 下一状态
            reward: 获得的奖励
            done: 是否结束
        """
        self.buffer.push(state, action, next_state, reward, done)
        self.step_counter += 1
        
    def optimize_model(self):
        """
        优化模型
        
        Returns:
            loss: 损失值
        """
        # 如果经验不足，不进行优化
        if len(self.buffer) < self.batch_size or self.step_counter < self.learning_starts:
            return None
            
        # 从缓冲区中采样
        state, action, next_state, reward, done = self.buffer.sample(self.batch_size)
        
        # 计算当前Q值
        q_values = self.policy_net(state)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        
        # 计算目标Q值
        if self.double_dqn:
            # Double DQN: 使用当前网络选择动作，使用目标网络估计Q值
            next_q_values = self.policy_net(next_state)
            next_actions = next_q_values.max(1)[1].unsqueeze(1)
            next_q_value = self.target_net(next_state).gather(1, next_actions).squeeze(1)
        else:
            # 标准DQN
            next_q_value = self.target_net(next_state).max(1)[0]
            
        # 计算目标值
        target = reward + (1 - done) * self.gamma * next_q_value
        
        # 计算损失
        loss = nn.functional.smooth_l1_loss(q_value, target)
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        # 记录损失
        self.training_info["losses"].append(loss.item())
        
        # 更新目标网络
        self.update_counter += 1
        if self.update_counter % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        return loss.item()
        
    def save_model(self, path):
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
            
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step_counter': self.step_counter,
            'update_counter': self.update_counter,
            'episode_counter': self.episode_counter,
            'epsilon': self.epsilon,
            'config': self.config
        }, path)
        
    def load_model(self, path):
        """
        加载模型
        
        Args:
            path: 模型路径
        """
        checkpoint = torch.load(path)
        
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.step_counter = checkpoint['step_counter']
        self.update_counter = checkpoint['update_counter']
        self.episode_counter = checkpoint['episode_counter']
        self.epsilon = checkpoint['epsilon']
        
        # 确保目标网络处于评估模式
        self.target_net.eval()
