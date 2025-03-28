"""
环境包装器，用于处理和转换游戏环境
"""
import gym
import numpy as np
from gym import spaces
import cv2
from collections import deque


class SkipFrame(gym.Wrapper):
    """
    跳帧包装器，用于跳过n帧，只处理每n帧中的最后一帧
    """
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        info = {}
        
        # 重复同一动作skip次，累计奖励
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            total_reward += reward
            if done:
                break
                
        return obs, total_reward, terminated, truncated, info


class GrayScaleObservation(gym.ObservationWrapper):
    """
    将RGB图像转换为灰度图像
    """
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
        )

    def observation(self, observation):
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    """
    调整观测图像的大小
    """
    def __init__(self, env, shape=(84, 84)):
        super().__init__(env)
        self.shape = tuple(shape)
        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
        )

    def observation(self, observation):
        observation = cv2.resize(
            observation, self.shape, interpolation=cv2.INTER_AREA
        )
        return observation


class NormalizeObservation(gym.ObservationWrapper):
    """
    将像素值归一化到[0, 1]范围
    """
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0, high=1.0, shape=self.observation_space.shape, dtype=np.float32
        )

    def observation(self, observation):
        return np.array(observation).astype(np.float32) / 255.0


class FrameStack(gym.Wrapper):
    """
    堆叠最近的n帧
    """
    def __init__(self, env, num_stack=4):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)
        
        low = np.repeat(
            self.observation_space.low[np.newaxis, ...], num_stack, axis=0
        )
        high = np.repeat(
            self.observation_space.high[np.newaxis, ...], num_stack, axis=0
        )
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=self.observation_space.dtype
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(obs)
        return self._get_observation(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self):
        return np.array(list(self.frames))


class RewardScaler(gym.RewardWrapper):
    """
    缩放奖励值
    """
    def __init__(self, env, scale=0.1):
        super().__init__(env)
        self.scale = scale

    def reward(self, reward):
        return reward * self.scale


class EpisodeInfo(gym.Wrapper):
    """
    记录每个episode的信息
    """
    def __init__(self, env):
        super().__init__(env)
        self.episode_reward = 0.0
        self.episode_length = 0
        self.x_pos = 0

    def reset(self, **kwargs):
        self.episode_reward = 0.0
        self.episode_length = 0
        self.x_pos = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_reward += reward
        self.episode_length += 1
        
        # 记录马里奥的x坐标位置
        if 'x_pos' in info:
            self.x_pos = max(self.x_pos, info['x_pos'])
            info['max_x_pos'] = self.x_pos
        
        info['episode_reward'] = self.episode_reward
        info['episode_length'] = self.episode_length
        
        return obs, reward, terminated, truncated, info
