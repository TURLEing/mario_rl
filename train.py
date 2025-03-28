"""
训练脚本，用于训练智能体
"""
import os
import argparse
import time
import numpy as np
import torch
import random
from tqdm import tqdm
import gymnasium as gym
from datetime import datetime

from config import (
    DEVICE, ENV_NAME, WORLD_STAGE, RENDER_MODE, FRAME_SKIP, 
    FRAME_STACK, RESIZE_SHAPE, REWARD_SCALE, MAX_EPISODE_STEPS,
    TOTAL_TIMESTEPS, SAVE_INTERVAL, LOG_INTERVAL, EVAL_INTERVAL, 
    EVAL_EPISODES, SEED, DQN_CONFIG, PPO_CONFIG, VISUALIZATION_CONFIG,
    CHECKPOINT_DIR
)
from environment.mario_env import MarioEnv
from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from utils.visualization import Visualizer, VideoRecorder


def set_seed(seed=SEED):
    """
    设置随机种子
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_agent(agent_type, env, device):
    """
    创建智能体
    
    Args:
        agent_type: 智能体类型 ('dqn' 或 'ppo')
        env: 环境实例
        device: 计算设备
    
    Returns:
        agent: 智能体实例
    """
    state_shape = env.observation_space.shape
    action_space = env.action_space
    
    if agent_type == 'dqn':
        return DQNAgent(state_shape, action_space, device, DQN_CONFIG)
    elif agent_type == 'ppo':
        return PPOAgent(state_shape, action_space, device, PPO_CONFIG)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def evaluate_agent(agent, env, n_episodes=EVAL_EPISODES, render=False):
    """
    评估智能体性能
    
    Args:
        agent: 智能体实例
        env: 环境实例
        n_episodes: 评估回合数
        render: 是否渲染
    
    Returns:
        eval_info: 评估信息
    """
    total_rewards = []
    total_lengths = []
    max_x_positions = []
    recorder = None
    
    if render:
        recorder = VideoRecorder()
        recorder.start_recording()
    
    for episode in range(n_episodes):
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        max_x_pos = 0
        done = False
        
        while not done:
            action = agent.select_action(state, evaluate=True)
            
            # 对于不同的智能体类型，select_action可能返回不同的格式
            if isinstance(action, tuple):
                action = action[0]
                
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # 更新指标
            episode_reward += reward
            episode_length += 1
            if 'x_pos' in info:
                max_x_pos = max(max_x_pos, info['x_pos'])
                
            # 录制视频帧
            if render and recorder is not None:
                frame = env.render()
                recorder.add_frame(frame)
                
            state = next_state
            
        # 保存指标
        total_rewards.append(episode_reward)
        total_lengths.append(episode_length)
        max_x_positions.append(max_x_pos)
    
    # 保存评估视频
    video_path = None
    if render and recorder is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = recorder.save_video(f"eval_{timestamp}")
    
    # 计算平均指标
    eval_info = {
        "mean_reward": np.mean(total_rewards),
        "std_reward": np.std(total_rewards),
        "mean_length": np.mean(total_lengths),
        "mean_x_pos": np.mean(max_x_positions) if max_x_positions else 0,
        "max_x_pos": np.max(max_x_positions) if max_x_positions else 0,
        "video_path": video_path
    }
    
    return eval_info


def train_dqn(agent, env, timesteps, visualizer=None):
    """
    训练DQN智能体
    
    Args:
        agent: DQN智能体
        env: 环境实例
        timesteps: 总训练步数
        visualizer: 可视化器
    """
    global_step = agent.step_counter
    episode = agent.episode_counter
    best_reward = -float('inf')
    
    # 进度条
    pbar = tqdm(total=timesteps)
    pbar.update(global_step)
    
    while global_step < timesteps:
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        max_x_pos = 0
        episode_frames = []
        
        done = False
        while not done:
            # 选择动作
            action = agent.select_action(state)
            
            # 执行动作
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # 存储经验
            agent.store_transition(state, action, next_state, reward, done)
            
            # 学习
            loss = agent.optimize_model()
            
            # 记录指标
            episode_reward += reward
            episode_length += 1
            global_step += 1
            if 'x_pos' in info:
                max_x_pos = max(max_x_pos, info['x_pos'])
                
            # 记录视频帧
            if visualizer is not None and visualizer.save_video:
                frame = env.render()
                episode_frames.append(frame)
                
            # 记录训练损失
            if loss is not None and visualizer is not None:
                visualizer.log_train(global_step, {"loss": loss})
                
            # 评估智能体
            if global_step % EVAL_INTERVAL == 0:
                # 创建单独的评估环境，避免影响训练环境
                eval_env = MarioEnv(
                    env_name=ENV_NAME,
                    world_stage=WORLD_STAGE,
                    frame_skip=FRAME_SKIP,
                    frame_stack=FRAME_STACK,
                    reward_scale=REWARD_SCALE,
                    resize_shape=RESIZE_SHAPE,
                    render_mode="rgb_array"
                )
                
                eval_info = evaluate_agent(agent, eval_env, n_episodes=EVAL_EPISODES)
                
                # 关闭评估环境
                eval_env.close()
                
                print(f"\nEvaluation at step {global_step}:")
                print(f"  Mean reward: {eval_info['mean_reward']:.2f}")
                print(f"  Mean episode length: {eval_info['mean_length']:.2f}")
                print(f"  Mean x position: {eval_info['mean_x_pos']:.2f}")
                print(f"  Max x position: {eval_info['max_x_pos']:.2f}")
                
                # 可视化评估结果
                if visualizer is not None:
                    visualizer.log_step(global_step, {
                        "eval_reward": eval_info['mean_reward'],
                        "eval_length": eval_info['mean_length'],
                        "eval_x_pos": eval_info['mean_x_pos'],
                        "eval_max_x_pos": eval_info['max_x_pos']
                    })
                
                # 保存最佳模型
                if eval_info['mean_reward'] > best_reward:
                    best_reward = eval_info['mean_reward']
                    agent.save_model(os.path.join(CHECKPOINT_DIR, "best_dqn_model.pt"))
                    print(f"  New best model with reward: {best_reward:.2f}")
                    
            # 保存模型
            if global_step % SAVE_INTERVAL == 0:
                agent.save_model(os.path.join(CHECKPOINT_DIR, f"dqn_model_step_{global_step}.pt"))
                
            # 更新进度条
            pbar.update(1)
            
            # 检查是否达到总步数
            if global_step >= timesteps:
                break
                
            state = next_state
        
        # 回合结束，记录回合统计信息
        episode += 1
        agent.episode_counter = episode
        
        # 记录回合信息
        episode_info = {
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            "max_x_pos": max_x_pos,
            "epsilon": agent.epsilon
        }
        
        # 打印回合信息
        if episode % LOG_INTERVAL == 0:
            print(f"\nEpisode {episode}:")
            print(f"  Reward: {episode_reward:.2f}")
            print(f"  Length: {episode_length}")
            print(f"  Max x position: {max_x_pos}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            
        # 可视化回合信息
        if visualizer is not None:
            avg_metrics = visualizer.log_episode(episode, episode_info)
            
            # 保存回合视频
            if visualizer.save_video and len(episode_frames) > 0:
                visualizer.save_episode_video(episode, episode_frames)
                
    # 训练结束，保存最终模型
    agent.save_model(os.path.join(CHECKPOINT_DIR, "final_dqn_model.pt"))
    pbar.close()


def train_ppo(agent, env, timesteps, visualizer=None):
    """
    训练PPO智能体
    
    Args:
        agent: PPO智能体
        env: 环境实例
        timesteps: 总训练步数
        visualizer: 可视化器
    """
    global_step = agent.step_counter
    episode = agent.episode_counter
    best_reward = -float('inf')
    
    # 进度条
    pbar = tqdm(total=timesteps)
    pbar.update(global_step)
    
    # 每n_steps更新一次
    n_steps = PPO_CONFIG["n_steps"]
    
    while global_step < timesteps:
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        max_x_pos = 0
        episode_frames = []
        
        done = False
        step_count = 0
        
        while not done:
            # 选择动作
            action, log_prob, value = agent.select_action(state)
            
            # 执行动作
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # 存储经验
            agent.store_transition(state, action, log_prob, value, reward, done)
            
            # 记录指标
            episode_reward += reward
            episode_length += 1
            global_step += 1
            step_count += 1
            if 'x_pos' in info:
                max_x_pos = max(max_x_pos, info['x_pos'])
                
            # 记录视频帧
            if visualizer is not None and visualizer.save_video:
                frame = env.render()
                episode_frames.append(frame)
                
            # 达到n_steps或回合结束时更新策略
            if step_count % n_steps == 0 or done:
                loss_info = agent.learn()
                
                # 记录训练损失
                if visualizer is not None:
                    visualizer.log_train(global_step, loss_info)
                    
            # 评估智能体
            if global_step % EVAL_INTERVAL == 0:
                # 创建单独的评估环境，避免影响训练环境
                eval_env = MarioEnv(
                    env_name=ENV_NAME,
                    world_stage=WORLD_STAGE,
                    frame_skip=FRAME_SKIP,
                    frame_stack=FRAME_STACK,
                    reward_scale=REWARD_SCALE,
                    resize_shape=RESIZE_SHAPE,
                    render_mode="rgb_array"
                )
                
                eval_info = evaluate_agent(agent, eval_env, n_episodes=EVAL_EPISODES)
                
                # 关闭评估环境
                eval_env.close()
                
                print(f"\nEvaluation at step {global_step}:")
                print(f"  Mean reward: {eval_info['mean_reward']:.2f}")
                print(f"  Mean episode length: {eval_info['mean_length']:.2f}")
                print(f"  Mean x position: {eval_info['mean_x_pos']:.2f}")
                print(f"  Max x position: {eval_info['max_x_pos']:.2f}")
                
                # 可视化评估结果
                if visualizer is not None:
                    visualizer.log_step(global_step, {
                        "eval_reward": eval_info['mean_reward'],
                        "eval_length": eval_info['mean_length'],
                        "eval_x_pos": eval_info['mean_x_pos'],
                        "eval_max_x_pos": eval_info['max_x_pos']
                    })
                
                # 保存最佳模型
                if eval_info['mean_reward'] > best_reward:
                    best_reward = eval_info['mean_reward']
                    agent.save_model(os.path.join(CHECKPOINT_DIR, "best_ppo_model.pt"))
                    print(f"  New best model with reward: {best_reward:.2f}")
                    
            # 保存模型
            if global_step % SAVE_INTERVAL == 0:
                agent.save_model(os.path.join(CHECKPOINT_DIR, f"ppo_model_step_{global_step}.pt"))
                
            # 更新进度条
            pbar.update(1)
            
            # 检查是否达到总步数
            if global_step >= timesteps:
                break
                
            state = next_state
        
        # 回合结束，记录回合统计信息
        episode += 1
        agent.episode_counter = episode
        
        # 记录回合信息
        episode_info = {
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            "max_x_pos": max_x_pos
        }
        
        # 打印回合信息
        if episode % LOG_INTERVAL == 0:
            print(f"\nEpisode {episode}:")
            print(f"  Reward: {episode_reward:.2f}")
            print(f"  Length: {episode_length}")
            print(f"  Max x position: {max_x_pos}")
            
        # 可视化回合信息
        if visualizer is not None:
            avg_metrics = visualizer.log_episode(episode, episode_info)
            
            # 保存回合视频
            if visualizer.save_video and len(episode_frames) > 0:
                visualizer.save_episode_video(episode, episode_frames)
                
    # 训练结束，保存最终模型
    agent.save_model(os.path.join(CHECKPOINT_DIR, "final_ppo_model.pt"))
    pbar.close()

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Train an agent to play Super Mario Bros")
    parser.add_argument("--agent", type=str, default="dqn", choices=["dqn", "ppo"], help="Agent type (dqn or ppo)")
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS, help="Total timesteps for training")
    parser.add_argument("--world", type=str, default=WORLD_STAGE, help="World-stage (e.g., '1-1')")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--render", action="store_true", help="Render environment")
    parser.add_argument("--load-model", type=str, default=None, help="Load model from path")
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
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
        render_mode=RENDER_MODE if args.render else "rgb_array"
    )
    
    # 创建智能体
    agent = create_agent(args.agent, env, device)
    
    # 加载模型（如果指定）
    if args.load_model is not None:
        print(f"Loading model from: {args.load_model}")
        agent.load_model(args.load_model)
    
    # 创建可视化器
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("logs", f"{args.agent}_{timestamp}")
    visualizer = Visualizer(VISUALIZATION_CONFIG, log_dir=log_dir)
    
    # 打印训练配置
    print("Training configuration:")
    print(f"  Agent type: {args.agent}")
    print(f"  World-stage: {args.world}")
    print(f"  Total timesteps: {args.timesteps}")
    print(f"  Seed: {args.seed}")
    
    # 开始训练
    print(f"Starting training with {args.agent.upper()} agent...")
    
    try:
        if args.agent == "dqn":
            train_dqn(agent, env, args.timesteps, visualizer)
        elif args.agent == "ppo":
            train_ppo(agent, env, args.timesteps, visualizer)
    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        # 关闭环境和可视化器
        env.close()
        visualizer.close()
        
    print("Training completed!")
    

if __name__ == "__main__":
    main()