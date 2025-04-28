# 超级马里奥强化学习

这是一个使用深度强化学习算法（DQN和PPO）训练智能体玩超级马里奥游戏的项目。项目包含完整的训练框架、可视化工具和评估脚本，目标是训练出能够通关游戏的AI智能体。

*注意：本项目框架基于 Claude 3.7 生成。*

## 特点

- 支持DQN（Deep Q-Network）和PPO（Proximal Policy Optimization）算法
- 内置训练过程可视化，包括TensorBoard和视频录制
- 易于使用的命令行接口，用于训练和评估
- 可配置的超参数和环境设置

## 安装

1. 克隆此仓库：

```bash
git clone https://github.com/your-username/mario-rl.git
cd mario-rl
```

2. 安装依赖项：

```bash
pip install -r requirements.txt
```

## 项目结构

```
mario-rl/
├── requirements.txt        # 项目依赖
├── main.py                 # 主程序入口
├── train.py                # 训练脚本
├── play.py                 # 使用训练好的模型玩游戏
├── config.py               # 配置文件
├── models/                 # 模型定义
│   ├── __init__.py
│   ├── dqn.py              # DQN模型
│   └── ppo.py              # PPO模型
├── agents/                 # 智能体实现
│   ├── __init__.py
│   ├── base_agent.py       # 基础智能体类
│   ├── dqn_agent.py        # DQN智能体
│   └── ppo_agent.py        # PPO智能体
├── environment/            # 环境包装器
│   ├── __init__.py
│   └── mario_env.py        # 超级马里奥环境适配器
├── utils/                  # 工具函数
│   ├── __init__.py
│   ├── replay_buffer.py    # 经验回放缓冲区
│   ├── wrappers.py         # 环境包装器
│   └── visualization.py    # 可视化工具
└── checkpoints/            # 模型保存目录
    └── README.md
```

## 使用方法

### 训练智能体

使用DQN算法训练智能体：

```bash
python main.py train --agent dqn --world 1-1 --timesteps 1000000
```

使用PPO算法训练智能体：

```bash
python main.py train --agent ppo --world 1-1 --timesteps 1000000
```

可选参数：
- `--agent`: 选择算法 (`dqn` 或 `ppo`)
- `--world`: 选择游戏关卡 (例如 `1-1`, `1-2` 等)
- `--timesteps`: 总训练步数
- `--seed`: 随机种子
- `--render`: 训练时渲染游戏画面
- `--load-model`: 从现有模型继续训练

### 使用训练好的模型玩游戏

```bash
python main.py play --agent ppo --model checkpoints/best_ppo_model.pt --world 1-1 --no-record
```

可选参数：
- `--agent`: 选择算法 (`dqn` 或 `ppo`)
- `--model`: 模型文件路径
- `--world`: 选择游戏关卡
- `--episodes`: 游戏回合数
- `--max-steps`: 每回合最大步数
- `--no-render`: 不渲染游戏画面
- `--no-record`: 不录制视频
- `--fps`: 渲染和录制的帧率

## 可视化

训练过程中会自动生成日志和可视化数据，可以通过TensorBoard查看：

```bash
tensorboard --logdir logs
```

这将显示以下指标：
- 奖励曲线
- 游戏进度（玩家在关卡中的最远位置）
- 损失函数曲线
- 回合长度
- 模型评估结果

## 模型保存

模型在训练过程中会自动保存：
- 每隔一段时间保存一次模型（可在config.py中配置）
- 保存性能最好的模型
- 训练结束时保存最终模型

所有模型都保存在`checkpoints`目录下。

## 配置

可以通过修改`config.py`文件来调整各种参数，如：
- 环境参数（帧跳过、帧堆叠、奖励缩放等）
- DQN参数（学习率、探索率、批次大小等）
- PPO参数（学习率、GAE lambda、裁剪范围等）
- 可视化设置（是否使用TensorBoard、是否保存视频等）

## 自定义

您可以通过以下方式自定义项目：
- 在`models/`目录下添加新的神经网络结构
- 在`agents/`目录下实现新的强化学习算法
- 修改`environment/mario_env.py`中的奖励函数
- 在`utils/wrappers.py`中添加新的环境包装器

## 贡献

欢迎贡献代码、报告问题或提出改进建议！

## 许可证

本项目采用MIT许可证。详见LICENSE文件。
