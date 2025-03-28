"""
配置文件，包含训练和模型的各种参数设置
"""
import torch

# 通用配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENV_NAME = "SuperMarioBros"
WORLD_STAGE = "1-1-v0"  # 游戏关卡，格式为"世界-关卡"
RENDER_MODE = "rgb_array"  # 渲染模式，训练时通常为rgb_array

# 环境预处理配置
FRAME_SKIP = 4      # 跳帧数量，每n帧执行一次动作
FRAME_STACK = 4     # 堆叠帧数量，将连续n帧作为状态输入
REWARD_SCALE = 0.1  # 奖励缩放
MAX_EPISODE_STEPS = 1000000  # 单轮游戏最大步数
RESIZE_SHAPE = (84, 84)  # 图像调整大小

# 训练配置
TOTAL_TIMESTEPS = 1000000  # 总训练步数
SAVE_INTERVAL = 50000      # 模型保存间隔
LOG_INTERVAL = 1000        # 日志记录间隔
EVAL_INTERVAL = 10000      # 评估间隔
EVAL_EPISODES = 5          # 每次评估的回合数
SEED = 42                  # 随机种子

# DQN配置
DQN_CONFIG = {
    "learning_rate": 1e-4,             # 学习率
    "gamma": 0.99,                     # 折扣因子
    "batch_size": 32,                  # 批次大小
    "buffer_size": 100000,             # 经验回放缓冲区大小
    "learning_starts": 10000,          # 开始学习前的步数
    "target_update_interval": 1000,    # 目标网络更新间隔
    "epsilon_start": 1.0,              # 初始探索率
    "epsilon_end": 0.1,                # 最终探索率
    "epsilon_decay": 100000,           # 探索率衰减步数
    "double_dqn": True,                # 是否使用Double DQN
    "dueling_dqn": True                # 是否使用Dueling DQN
}

# PPO配置
PPO_CONFIG = {
    "learning_rate": 3e-4,             # 学习率
    "gamma": 0.99,                     # 折扣因子
    "n_steps": 2048,                   # 每次更新的步数
    "batch_size": 64,                  # 批次大小
    "n_epochs": 10,                    # 每批数据的训练轮数
    "gae_lambda": 0.95,                # GAE lambda参数
    "clip_range": 0.2,                 # PPO裁剪范围
    "entropy_coef": 0.01,              # 熵系数
    "value_coef": 0.5,                 # 价值函数系数
    "max_grad_norm": 0.5               # 梯度裁剪
}

# 可视化配置
VISUALIZATION_CONFIG = {
    "use_tensorboard": True,           # 是否使用TensorBoard
    "use_wandb": False,                # 是否使用Weights & Biases
    "project_name": "mario-rl",        # 项目名称
    "save_video": True,                # 是否保存视频
    "video_interval": 50000,           # 视频保存间隔
    "video_length": 1000               # 视频长度
}

# 模型保存路径
CHECKPOINT_DIR = "checkpoints"
