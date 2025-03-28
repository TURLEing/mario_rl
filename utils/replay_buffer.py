"""
经验回放缓冲区，用于DQN算法
"""
import numpy as np
import random
from collections import deque, namedtuple
import torch

# 定义经验转移元组
Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayBuffer:
    """
    经验回放缓冲区
    """
    def __init__(self, capacity, device):
        """
        初始化经验回放缓冲区
        
        Args:
            capacity: 缓冲区容量
            device: 计算设备
        """
        self.memory = deque(maxlen=capacity)
        self.device = device
        
    def push(self, state, action, next_state, reward, done):
        """
        添加经验到缓冲区
        """
        self.memory.append(Transition(state, action, next_state, reward, done))
        
    def sample(self, batch_size):
        """
        随机采样一批经验
        
        Args:
            batch_size: 批次大小
            
        Returns:
            batch: 经验批次
        """
        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))
        
        # 转换为张量
        state = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action = torch.LongTensor(np.array(batch.action)).to(self.device)
        reward = torch.FloatTensor(np.array(batch.reward)).to(self.device)
        next_state = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done = torch.FloatTensor(np.array(batch.done)).to(self.device)
        
        return state, action, next_state, reward, done
        
    def __len__(self):
        """
        返回当前缓冲区大小
        """
        return len(self.memory)


class PrioritizedReplayBuffer:
    """
    优先经验回放缓冲区
    """
    def __init__(self, capacity, device, alpha=0.6, beta=0.4, beta_increment=0.001):
        """
        初始化优先经验回放缓冲区
        
        Args:
            capacity: 缓冲区容量
            device: 计算设备
            alpha: 优先级指数
            beta: 重要性采样指数
            beta_increment: beta增量
        """
        self.capacity = capacity
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.memory = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0
        
    def push(self, state, action, next_state, reward, done):
        """
        添加经验到缓冲区
        """
        max_priority = self.max_priority if len(self.memory) > 0 else 1.0
        
        if len(self.memory) < self.capacity:
            self.memory.append(Transition(state, action, next_state, reward, done))
        else:
            self.memory[self.position] = Transition(state, action, next_state, reward, done)
            
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        """
        按优先级采样一批经验
        
        Args:
            batch_size: 批次大小
            
        Returns:
            batch: 经验批次
            indices: 采样索引
            weights: 重要性采样权重
        """
        if len(self.memory) < self.capacity:
            probs = self.priorities[:len(self.memory)]
        else:
            probs = self.priorities
            
        # 计算采样概率
        probs = probs ** self.alpha
        probs = probs / np.sum(probs)
        
        # 采样索引
        indices = np.random.choice(len(probs), batch_size, p=probs)
        
        # 获取样本
        samples = [self.memory[idx] for idx in indices]
        batch = Transition(*zip(*samples))
        
        # 计算重要性采样权重
        weights = (len(self.memory) * probs[indices]) ** (-self.beta)
        weights = weights / np.max(weights)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # 增加beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # 转换为张量
        state = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action = torch.LongTensor(np.array(batch.action)).to(self.device)
        reward = torch.FloatTensor(np.array(batch.reward)).to(self.device)
        next_state = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done = torch.FloatTensor(np.array(batch.done)).to(self.device)
        
        return (state, action, next_state, reward, done), indices, weights
        
    def update_priorities(self, indices, priorities):
        """
        更新优先级
        
        Args:
            indices: 索引
            priorities: 优先级
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            
        self.max_priority = max(self.max_priority, np.max(priorities))
        
    def __len__(self):
        """
        返回当前缓冲区大小
        """
        return len(self.memory)
