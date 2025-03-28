"""
可视化工具，用于实时监控训练进度和结果
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from collections import deque
import torch
import cv2
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class Visualizer:
    """
    训练可视化器
    """
    def __init__(self, config, log_dir="logs"):
        """
        初始化可视化器
        
        Args:
            config: 可视化配置
            log_dir: 日志目录
        """
        self.config = config
        self.log_dir = log_dir
        
        # 创建日志目录
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # TensorBoard
        if config["use_tensorboard"]:
            self.writer = SummaryWriter(log_dir=log_dir)
            
        # Weights & Biases
        if config["use_wandb"] and WANDB_AVAILABLE:
            wandb.init(project=config["project_name"])
            
        # 视频录制
        self.save_video = config["save_video"]
        self.video_interval = config["video_interval"]
        self.video_length = config["video_length"]
        
        # 指标跟踪
        self.metrics = {
            "rewards": deque(maxlen=100),
            "lengths": deque(maxlen=100),
            "x_pos": deque(maxlen=100),
            "time": deque(maxlen=100),
            "losses": deque(maxlen=100),
            "q_values": deque(maxlen=100)
        }
        
    def log_step(self, step, infos):
        """
        记录单步日志
        
        Args:
            step: 全局步数
            infos: 信息字典
        """
        # 更新TensorBoard
        if self.config["use_tensorboard"]:
            for key, value in infos.items():
                self.writer.add_scalar(f"step/{key}", value, step)
                
        # 更新Weights & Biases
        if self.config["use_wandb"] and WANDB_AVAILABLE:
            wandb.log(infos, step=step)
            
    def log_episode(self, episode, infos):
        """
        记录单个回合日志
        
        Args:
            episode: 回合数
            infos: 信息字典
        """
        # 更新指标
        self.metrics["rewards"].append(infos.get("episode_reward", 0))
        self.metrics["lengths"].append(infos.get("episode_length", 0))
        self.metrics["x_pos"].append(infos.get("max_x_pos", 0))
        self.metrics["time"].append(infos.get("time", 0))
        
        # 计算平均指标
        avg_metrics = {
            "avg_reward": np.mean(self.metrics["rewards"]),
            "avg_length": np.mean(self.metrics["lengths"]),
            "avg_x_pos": np.mean(self.metrics["x_pos"]),
            "max_x_pos": np.max(self.metrics["x_pos"]) if len(self.metrics["x_pos"]) > 0 else 0
        }
        
        # 将单个回合信息和平均指标合并
        log_info = {**infos, **avg_metrics}
        
        # 更新TensorBoard
        if self.config["use_tensorboard"]:
            for key, value in log_info.items():
                self.writer.add_scalar(f"episode/{key}", value, episode)
                
        # 更新Weights & Biases
        if self.config["use_wandb"] and WANDB_AVAILABLE:
            wandb.log(log_info, step=episode)
            
        return avg_metrics
        
    def log_train(self, step, loss_info):
        """
        记录训练日志
        
        Args:
            step: 全局步数
            loss_info: 损失信息字典
        """
        # 更新指标
        if "loss" in loss_info:
            self.metrics["losses"].append(loss_info["loss"])
            
        if "q_value" in loss_info:
            self.metrics["q_values"].append(loss_info["q_value"])
            
        # 更新TensorBoard
        if self.config["use_tensorboard"]:
            for key, value in loss_info.items():
                self.writer.add_scalar(f"train/{key}", value, step)
                
        # 更新Weights & Biases
        if self.config["use_wandb"] and WANDB_AVAILABLE:
            wandb.log({f"train/{k}": v for k, v in loss_info.items()}, step=step)
            
    def save_episode_video(self, step, frames, fps=30):
        """
        保存回合视频
        
        Args:
            step: 全局步数
            frames: 视频帧列表
            fps: 帧率
        """
        if not self.save_video or step % self.video_interval != 0:
            return
            
        # 创建视频目录
        video_dir = os.path.join(self.log_dir, "videos")
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
            
        # 保存视频
        video_path = os.path.join(video_dir, f"episode_{step}.mp4")
        
        # 确保所有帧的形状一致
        for i in range(len(frames)):
            if frames[i].shape[-1] != 3:  # 如果不是RGB图像
                frames[i] = cv2.cvtColor(frames[i], cv2.COLOR_GRAY2RGB)
            frames[i] = cv2.resize(frames[i], (640, 480))
            
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, fps, (frames[0].shape[1], frames[0].shape[0]))
        
        # 写入帧
        for frame in frames:
            out.write(frame)
            
        out.release()
        
        # 将视频添加到TensorBoard
        if self.config["use_tensorboard"]:
            self.writer.add_video("episode_video", np.array(frames).transpose(0, 3, 1, 2)[None], step, fps=fps)
            
        # 将视频添加到Weights & Biases
        if self.config["use_wandb"] and WANDB_AVAILABLE:
            wandb.log({"episode_video": wandb.Video(video_path, fps=fps, format="mp4")}, step=step)
            
    def plot_learning_curve(self, x, scores, figure_file, title="Learning Curve"):
        """
        绘制学习曲线
        
        Args:
            x: x轴数据
            scores: 分数数据
            figure_file: 图像保存路径
            title: 图像标题
        """
        # 计算平均值
        running_avg = np.zeros(len(scores))
        for i in range(len(running_avg)):
            running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
            
        # 绘制图像
        plt.figure(figsize=(12, 8))
        plt.plot(x, scores, color='blue', alpha=0.3)
        plt.plot(x, running_avg, color='red')
        plt.title(title)
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.savefig(figure_file)
        plt.close()
        
        return figure_file
        
    def plot_mario_progress(self, episodes, x_positions, figure_file, title="Mario Progress"):
        """
        绘制马里奥进度图
        
        Args:
            episodes: 回合数据
            x_positions: x位置数据
            figure_file: 图像保存路径
            title: 图像标题
        """
        # 绘制图像
        plt.figure(figsize=(12, 8))
        plt.plot(episodes, x_positions)
        plt.title(title)
        plt.xlabel('Episode')
        plt.ylabel('Max X Position')
        plt.savefig(figure_file)
        plt.close()
        
        return figure_file
        
    def close(self):
        """
        关闭可视化器
        """
        if self.config["use_tensorboard"]:
            self.writer.close()
            
        if self.config["use_wandb"] and WANDB_AVAILABLE:
            wandb.finish()


class VideoRecorder:
    """
    视频录制器
    """
    def __init__(self, video_folder="videos"):
        """
        初始化视频录制器
        
        Args:
            video_folder: 视频保存文件夹
        """
        self.video_folder = video_folder
        
        # 创建视频文件夹
        if not os.path.exists(video_folder):
            os.makedirs(video_folder)
            
        self.frames = []
        self.recording = False
        
    def start_recording(self):
        """
        开始录制
        """
        self.frames = []
        self.recording = True
        
    def add_frame(self, frame):
        """
        添加帧
        
        Args:
            frame: 视频帧
        """
        if self.recording:
            # 确保帧是RGB格式
            if frame.shape[-1] != 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            self.frames.append(frame)
            
    def save_video(self, filename, fps=30):
        """
        保存视频
        
        Args:
            filename: 文件名
            fps: 帧率
        """
        if not self.recording or len(self.frames) == 0:
            return None
            
        # 确保所有帧的形状一致
        for i in range(len(self.frames)):
            self.frames[i] = cv2.resize(self.frames[i], (640, 480))
            
        # 创建视频写入器
        video_path = os.path.join(self.video_folder, f"{filename}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, fps, (self.frames[0].shape[1], self.frames[0].shape[0]))
        
        # 写入帧
        for frame in self.frames:
            out.write(frame)
            
        out.release()
        self.recording = False
        
        return video_path
        
    def get_frames(self):
        """
        获取所有帧
        
        Returns:
            frames: 帧列表
        """
        return self.frames
