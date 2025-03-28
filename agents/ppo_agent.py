"""
PPO智能体
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from datetime import datetime

from models.ppo_model import PPONetwork


class PPOMemory:
    """
    PPO记忆缓冲区，用于存储轨迹
    """
    def __init__(self, batch_size):
        """
        初始化PPO记忆缓冲区
        
        Args:
            batch_size: 批次大小
        """
        self.states = []
        self.actions = []
        self.probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.returns = []  # 添加returns属性
        self.batch_size = batch_size
        
    def push(self, state, action, prob, value, reward, done):
        """
        存储经验
        """
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        
    def clear(self):
        """
        清空缓冲区
        """
        self.states = []
        self.actions = []
        self.probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        
    def compute_returns(self, last_value, gamma, gae_lambda):
        """
        计算广义优势估计(GAE)和回报
        
        Args:
            last_value: 最后状态的价值
            gamma: 折扣因子
            gae_lambda: GAE lambda参数
            
        Returns:
            returns: 计算的回报
            advantage: 计算的优势
        """
        rewards = self.rewards
        values = self.values + [last_value]
        dones = self.dones
        
        returns = []
        gae = 0
        
        # 反向计算GAE
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + gamma * gae_lambda * (1 - dones[step]) * gae
            returns.insert(0, gae + values[step])
            
        return returns
        
    def get_minibatch(self, device):
        """
        获取批数据
        
        Args:
            device: 计算设备
            
        Returns:
            mini_batch: 批数据
        """
        batch_size = len(self.states)
        indices = np.random.permutation(batch_size)
        
        # 将所有数据转换为张量
        states = torch.FloatTensor(np.array(self.states)).to(device)
        actions = torch.LongTensor(np.array(self.actions)).to(device)
        old_probs = torch.FloatTensor(np.array(self.probs)).to(device)
        values = torch.FloatTensor(np.array(self.values)).to(device)
        returns = torch.FloatTensor(np.array(self.returns)).to(device)
        advantage = returns - values
        
        # 标准化优势
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        
        # 创建批次
        mini_batch_size = self.batch_size
        num_mini_batches = batch_size // mini_batch_size
        
        for i in range(num_mini_batches):
            start = i * mini_batch_size
            end = (i + 1) * mini_batch_size
            batch_indices = indices[start:end]
            
            yield {
                'states': states[batch_indices],
                'actions': actions[batch_indices],
                'old_probs': old_probs[batch_indices],
                'values': values[batch_indices],
                'returns': returns[batch_indices],
                'advantage': advantage[batch_indices]
            }
            
    def generate_batches(self, device):
        """
        生成批数据生成器
        
        Args:
            device: 计算设备
            
        Returns:
            generator: 批数据生成器
        """
        states = torch.FloatTensor(np.array(self.states)).to(device)
        actions = torch.LongTensor(np.array(self.actions)).to(device)
        old_probs = torch.FloatTensor(np.array(self.probs)).to(device)
        returns = torch.FloatTensor(np.array(self.returns)).to(device)
        
        # 计算优势
        values = torch.FloatTensor(np.array(self.values)).to(device)
        advantage = returns - values
        
        # 标准化优势
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        
        return states, actions, old_probs, returns, advantage


class PPOAgent:
    """
    PPO智能体
    """
    def __init__(self, state_shape, action_space, device, config):
        """
        初始化PPO智能体
        
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
        self.policy = PPONetwork(state_shape, self.n_actions).to(device)
        
        # 创建优化器
        self.optimizer = optim.Adam(
            self.policy.parameters(), 
            lr=config["learning_rate"]
        )
        
        # 创建记忆缓冲区
        self.memory = PPOMemory(config["batch_size"])
        
        # PPO参数
        self.gamma = config["gamma"]  # 折扣因子
        self.gae_lambda = config["gae_lambda"]  # GAE lambda
        self.clip_range = config["clip_range"]  # PPO裁剪范围
        self.n_epochs = config["n_epochs"]  # 训练轮数
        self.n_steps = config["n_steps"]  # 每轮步数
        self.entropy_coef = config["entropy_coef"]  # 熵系数
        self.value_coef = config["value_coef"]  # 价值系数
        self.max_grad_norm = config["max_grad_norm"]  # 梯度裁剪
        
        # 训练计数器
        self.step_counter = 0
        self.update_counter = 0
        self.episode_counter = 0
        
        # 日志
        self.training_info = {
            "policy_losses": [],
            "value_losses": [],
            "entropy": [],
            "approx_kl": []
        }
        
    def select_action(self, state, evaluate=False):
        """
        选择动作
        
        Args:
            state: 环境状态
            evaluate: 是否为评估模式
            
        Returns:
            action: 选择的动作
            log_prob: 动作对数概率
            value: 状态价值
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # 使用策略网络选择动作
        action, log_prob, value = self.policy.act(state, deterministic=evaluate)
        
        return action.item(), log_prob.item(), value.item()
        
    def store_transition(self, state, action, prob, value, reward, done):
        """
        存储经验到记忆缓冲区
        
        Args:
            state: 当前状态
            action: 执行的动作
            prob: 动作概率
            value: 状态价值
            reward: 获得的奖励
            done: 是否结束
        """
        self.memory.push(state, action, prob, value, reward, done)
        self.step_counter += 1
        
    def learn(self):
        """
        学习更新策略
        
        Returns:
            loss_info: 损失信息
        """
        # 检查记忆缓冲区是否为空
        if len(self.memory.states) == 0:
            return {
                "policy_loss": 0,
                "value_loss": 0,
                "entropy": 0,
                "approx_kl": 0,
                "clip_fraction": 0,
                "explained_variance": 0
            }
            
        # 获取最后状态的价值估计
        with torch.no_grad():
            state = torch.FloatTensor(self.memory.states[-1]).unsqueeze(0).to(self.device)
            _, last_value = self.policy(state)
            last_value = last_value.item()
            
        # 计算GAE和回报
        self.memory.returns = self.memory.compute_returns(last_value, self.gamma, self.gae_lambda)
        
        # 生成批数据
        states, actions, old_probs, returns, advantages = self.memory.generate_batches(self.device)
        
        # 多轮训练
        loss_info = {
            "policy_loss": 0,
            "value_loss": 0,
            "entropy": 0,
            "approx_kl": 0,
            "clip_fraction": 0,
            "explained_variance": 0
        }
        
        # 训练多轮
        for _ in range(self.n_epochs):
            # 评估动作
            new_log_probs, state_values, entropy = self.policy.evaluate(states, actions)
            state_values = state_values.squeeze()
            
            # 计算比率
            ratios = torch.exp(new_log_probs - old_probs)
            
            # 计算策略损失
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_range, 1 + self.clip_range) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 计算价值损失
            value_loss = nn.functional.mse_loss(state_values, returns)
            
            # 计算总损失
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()
            
            # 优化
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            # 计算近似KL散度
            approx_kl = ((ratios - 1) - torch.log(ratios)).mean().item()
            
            # 计算裁剪比例
            clip_fraction = ((ratios - 1).abs() > self.clip_range).float().mean().item()
            
            # 计算解释方差
            explained_var = 1 - (returns - state_values).var() / returns.var()
            if torch.isnan(explained_var):
                explained_var = torch.tensor(0.0)
                
            # 累加损失信息
            loss_info["policy_loss"] += policy_loss.item() / self.n_epochs
            loss_info["value_loss"] += value_loss.item() / self.n_epochs
            loss_info["entropy"] += entropy.mean().item() / self.n_epochs
            loss_info["approx_kl"] += approx_kl / self.n_epochs
            loss_info["clip_fraction"] += clip_fraction / self.n_epochs
            loss_info["explained_variance"] = explained_var.item()
            
        # 记录损失
        self.training_info["policy_losses"].append(loss_info["policy_loss"])
        self.training_info["value_losses"].append(loss_info["value_loss"])
        self.training_info["entropy"].append(loss_info["entropy"])
        self.training_info["approx_kl"].append(loss_info["approx_kl"])
        
        # 清空记忆缓冲区
        self.memory.clear()
        
        # 更新计数器
        self.update_counter += 1
        
        return loss_info
        
    def save_model(self, path):
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
            
        torch.save({
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step_counter': self.step_counter,
            'update_counter': self.update_counter,
            'episode_counter': self.episode_counter,
            'config': self.config
        }, path)
        
    def load_model(self, path):
        """
        加载模型
        
        Args:
            path: 模型路径
        """
        checkpoint = torch.load(path)
        
        self.policy.load_state_dict(checkpoint['policy'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.step_counter = checkpoint['step_counter']
        self.update_counter = checkpoint['update_counter']
        self.episode_counter = checkpoint['episode_counter']