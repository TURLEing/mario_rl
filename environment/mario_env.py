"""
超级马里奥环境适配器
"""
import gym
import numpy as np
from gym import spaces
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros import SuperMarioBrosEnv

from utils.wrappers import (
    SkipFrame,
    GrayScaleObservation,
    ResizeObservation,
    NormalizeObservation,
    FrameStack,
    RewardScaler,
    EpisodeInfo
)


class MarioEnv:
    """
    马里奥环境包装器，整合所有需要的处理
    """
    def __init__(
        self,
        env_name="SuperMarioBros-v0",
        world_stage="1-1",
        movement="simple",
        frame_skip=4,
        frame_stack=4,
        reward_scale=0.1,
        resize_shape=(84, 84),
        render_mode="rgb_array",
        record_video=False
    ):
        """
        初始化马里奥环境
        
        Args:
            env_name: 环境名称
            world_stage: 游戏关卡 (format: "W-S")
            movement: 动作空间类型 ("simple", "complex", "right_only")
            frame_skip: 跳帧数量
            frame_stack: 堆叠帧数量
            reward_scale: 奖励缩放
            resize_shape: 图像调整大小
            render_mode: 渲染模式
            record_video: 是否记录视频
        """
        # 创建基础环境
        if world_stage:
            # 指定关卡
            env = gym.make(
                f"{env_name}-{world_stage}",
                render_mode=render_mode,
                apply_api_compatibility=True
            )
        else:
            # 使用默认关卡
            env = gym.make(
                env_name,
                render_mode=render_mode,
                apply_api_compatibility=True
            )
            
        # 设置动作空间
        if movement == "simple":
            action_space = SIMPLE_MOVEMENT
        elif movement == "complex":
            action_space = COMPLEX_MOVEMENT
        elif movement == "right_only":
            action_space = RIGHT_ONLY
        else:
            raise ValueError(f"Unknown movement type: {movement}")
            
        env = JoypadSpace(env, action_space)
        
        # 应用包装器
        env = SkipFrame(env, skip=frame_skip)
        env = GrayScaleObservation(env)
        env = ResizeObservation(env, shape=resize_shape)
        env = NormalizeObservation(env)
        env = FrameStack(env, num_stack=frame_stack)
        env = RewardScaler(env, scale=reward_scale)
        env = EpisodeInfo(env)
        
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        
    def reset(self):
        """重置环境"""
        return self.env.reset()
        
    def step(self, action):
        """执行动作"""
        return self.env.step(action)
        
    def render(self):
        """渲染环境"""
        return self.env.render()
        
    def close(self):
        """关闭环境"""
        self.env.close()
        
    @staticmethod
    def get_custom_reward(info):
        """
        自定义奖励函数，根据游戏状态信息提供更有意义的奖励
        
        Args:
            info: 游戏信息字典
            
        Returns:
            additional_reward: 额外奖励值
        """
        additional_reward = 0
        
        # 根据x位置前进给予奖励
        if "x_pos" in info and "x_pos_previous" in info:
            x_progress = info["x_pos"] - info["x_pos_previous"]
            additional_reward += x_progress * 0.1
            
        # 根据时间剩余给予奖励（鼓励速度）
        if "time" in info and "time_previous" in info:
            time_penalty = (info["time_previous"] - info["time"]) * 0.01
            additional_reward -= time_penalty
            
        # 收集金币奖励
        if "coins" in info and "coins_previous" in info:
            coins_collected = info["coins"] - info["coins_previous"]
            additional_reward += coins_collected * 5
            
        # 消灭敌人奖励
        if "status" in info and "status_previous" in info:
            if info["status"] == "fireball" and info["status_previous"] != "fireball":
                additional_reward += 10
                
        # 生命损失惩罚
        if "life" in info and "life_previous" in info:
            life_lost = info["life_previous"] - info["life"]
            if life_lost > 0:
                additional_reward -= 25
                
        return additional_reward
