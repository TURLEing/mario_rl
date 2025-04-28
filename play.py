"""
使用训练好的模型玩游戏
"""
import os
import argparse
import time
import numpy as np
import torch
from tqdm import tqdm
import gymnasium as gym

from config import DEVICE, ENV_NAME, WORLD_STAGE, FRAME_SKIP, FRAME_STACK, RESIZE_SHAPE, REWARD_SCALE
from environment.mario_env import MarioEnv
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from utils.visualization import VideoRecorder


def play_game(agent, env, episodes=5, max_steps=10000, render=True, record=True, fps=30):
    """
    使用训练好的智能体玩游戏
    
    Args:
        agent: 智能体实例
        env: 环境实例
        episodes: 游戏回合数
        max_steps: 每回合最大步数
        render: 是否渲染
        record: 是否录制视频
        fps: 录制视频的帧率
    """
    recorder = None
    if record:
        recorder = VideoRecorder()
    
    for episode in range(episodes):
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        max_x_pos = 0
        done = False
        
        if record:
            recorder.start_recording()
        
        print(f"Episode {episode+1}/{episodes}")
        
        step = 0
        while not done and step < max_steps:
            # 选择动作
            action = agent.select_action(state, evaluate=True)
            
            # 对于不同的智能体类型，select_action可能返回不同的格式
            if isinstance(action, tuple):
                action = action[0]
            
            # 执行动作
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # 更新指标
            episode_reward += reward
            episode_length += 1
            step += 1
            if 'x_pos' in info:
                max_x_pos = max(max_x_pos, info['x_pos'])
            
            # 渲染和录制
            if render or record:
                frame = env.render()
                if record:
                    recorder.add_frame(frame)
                
                # 如果渲染，则添加一点延迟使游戏可观看
                if render:
                    time.sleep(1/fps)
            
            # 打印进度
            if step % 100 == 0:
                print(f"Step: {step}, X-Position: {info.get('x_pos', 0)}, Reward: {episode_reward:.2f}")
            
            state = next_state
        
        # 回合结束，打印统计信息
        print(f"Episode {episode+1} finished:")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Length: {episode_length}")
        print(f"  Max x position: {max_x_pos}")
        
        # 保存视频
        if record:
            video_path = recorder.save_video(f"mario_episode_{episode+1}")
            print(f"  Video saved to: {video_path}")

def main():
        # 解析命令行参数
    parser = argparse.ArgumentParser(description="Play Super Mario Bros with a trained agent")
    parser.add_argument("--agent", type=str, default="dqn", choices=["dqn", "ppo"], help="Agent type (dqn or ppo)")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--world", type=str, default=WORLD_STAGE, help="World-stage (e.g., '1-1')")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to play")
    parser.add_argument("--max-steps", type=int, default=10000, help="Maximum steps per episode")
    parser.add_argument("--no-render", action="store_true", help="Do not render environment")
    parser.add_argument("--no-record", action="store_true", help="Do not record video")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for rendering and recording")
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建环境
    env = MarioEnv(
        env_name=ENV_NAME,
        world_stage=args.world,
        frame_skip=FRAME_SKIP,
        frame_stack=FRAME_STACK,
        reward_scale=REWARD_SCALE,
        resize_shape=RESIZE_SHAPE,
        render_mode="human" # if not args.no_render else "rgb_array"
    )
    
    # 加载模型
    print(f"Loading model from: {args.model}")
    
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    
    # 创建智能体并加载模型
    if args.agent == "dqn":
        # 从checkpoint获取配置
        config = checkpoint.get('config', {})
        # 创建状态和动作空间形状
        state_shape = env.observation_space.shape
        action_space = env.action_space
        # 创建智能体
        agent = DQNAgent(state_shape, action_space, device, config)
        agent.load_model(args.model)
    elif args.agent == "ppo":
        # 从checkpoint获取配置
        config = checkpoint.get('config', {})
        # 创建状态和动作空间形状
        state_shape = env.observation_space.shape
        action_space = env.action_space
        # 创建智能体
        agent = PPOAgent(state_shape, action_space, device, config)
        agent.load_model(args.model)
    else:
        raise ValueError(f"Unknown agent type: {args.agent}")
    
    # 开始游戏
    print(f"Starting to play with {args.agent.upper()} agent...")
    play_game(
        agent, 
        env, 
        episodes=args.episodes, 
        max_steps=args.max_steps, 
        render=not args.no_render, 
        record=not args.no_record,
        fps=args.fps
    )
    
    # 关闭环境
    env.close()
    print("Game completed!")


if __name__ == "__main__":
    main()