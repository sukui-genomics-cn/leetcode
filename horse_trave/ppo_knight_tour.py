import os

import torch as th
import torch.nn as nn
from typing import Dict, Tuple, Type, Union, List
from gym import spaces

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

from knight_tour_env_gym import KnightsTourEnv


class DynamicCNN(BaseFeaturesExtractor):
    """
    自适应棋盘尺寸的CNN特征提取器
    """
    def __init__(self, observation_space, features_dim: int = 256):
        n_channels = observation_space.shape[0]
        board_size = observation_space.shape[1]
        
        super().__init__(observation_space, features_dim)
        
        # 动态构建卷积层
        layers = []
        in_channels = n_channels
        out_channels = 32
        
        while board_size >= 4:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2) if board_size > 6 else nn.Identity()
            ])
            in_channels = out_channels
            out_channels *= 2
            board_size = board_size // 2 if board_size > 6 else board_size
        
        self.cnn = nn.Sequential(*layers)
        
        # 计算全连接层输入尺寸
        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).view(1, -1).shape[1]
        
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations).flatten(1))

class SeparateNetworksPolicy(nn.Module):
    """
    替代方案：通过net_arch实现非共享网络
    """
    def __init__(self, 
                 observation_space, 
                 action_space,
                 lr_schedule,
                 net_arch: Union[List[int], Dict[str, List[int]]] = None,
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 *args, **kwargs):
        
        super().__init__()
        
        if net_arch is None:
            # 默认非共享架构
            net_arch = [
                dict(vf=[256, 128], pi=[256, 128])
            ]
        
        # 确保net_arch格式正确
        if isinstance(net_arch, list) and len(net_arch) > 0:
            net_arch = net_arch[0]
        
        # 特征提取器（共享或不共享由share_features_extractor控制）
        self.features_extractor = DynamicCNN(observation_space, 512)
        
        # 构建policy和value网络
        self.policy_net = self._build_mlp(512, action_space.n, net_arch["pi"], activation_fn)
        self.value_net = self._build_mlp(512, 1, net_arch["vf"], activation_fn)
    
    def _build_mlp(self, input_dim, output_dim, net_arch, activation_fn):
        layers = []
        prev_dim = input_dim
        for dim in net_arch:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(activation_fn())
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)
    
    def forward(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        features = self.features_extractor(obs)
        return self.policy_net(features), self.value_net(features)

def get_policy_kwargs(env, **kwargs) -> Dict:
    """
    生成兼容的policy_kwargs配置
    """
    return dict(
        features_extractor_class=DynamicCNN,
        features_extractor_kwargs=dict(
            features_dim=kwargs.get("features_dim", 512)
        ),
        net_arch=[dict(pi=[256, 128], vf=[256, 128])],  # 非共享架构
        activation_fn=nn.ReLU,
        optimizer_class=th.optim.AdamW,
        optimizer_kwargs=dict(weight_decay=1e-5),
        share_features_extractor=False  # 关键参数
    )

def train_with_sb3(continue_training=False, model_path=None):
    log_dir = "./logs/ppo_knight_tour_8x8"
    video_dir = "./videos/ppo_knight_tour_8x8"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)

    # 创建并行环境
    env = make_vec_env(
        lambda: Monitor(KnightsTourEnv(board_size=8), log_dir),
        n_envs=8,
        seed=42,
        vec_env_cls=SubprocVecEnv  # 使用多进程提高训练效率
    )
    
    # 添加视频录制（每10000步录制一次，最多录制100步）
    env = VecVideoRecorder(
        env,
        video_dir,
        record_video_trigger=lambda x: x % 100000 == 0,  # 录制频率
        video_length=65,  # 最大录制长度
        name_prefix="ppo_knight_tour_8x8"
    )

    # 初始化PPO算法
    if continue_training and model_path and os.path.exists(model_path + ".zip"):
        print(f"Loading model from {model_path} to continue training...")
        model = PPO.load(model_path, env=env, tensorboard_log=log_dir)
    else:
        print("Initializing new model...")
        # policy_kwargs = get_policy_kwargs(env, features_dim=256, net_arch=dict(pi=[128,64], vf=[128,64]))
        policy_kwargs = {
            # 增大网络规模
            "net_arch": [
                {"vf": [256, 128], "pi": [256, 128]}  # 分离的policy和value网络
            ],
            
            # 关键参数：不共享特征提取器
            "share_features_extractor": False,
            }
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log="./logs/",
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            device="auto"  # 自动选择GPU/CPU
        )

    # 添加模型保存回调（每50000步保存一次最佳模型）
    eval_callback = EvalCallback(
        env,
        best_model_save_path="./best_models/",
        log_path="./eval_logs/",
        eval_freq=50000,  # 每5万步评估一次
        deterministic=True,
        render=False
    )
    
    # 训练模型
    model.learn(
        total_timesteps=10000000,
        tb_log_name="ppo_knight_tour_8x8",
        callback=eval_callback,  # 包含自动保存功能
        reset_num_timesteps=not continue_training  # 是否重置步数计数器
        )
    
    # 保存模型
    model.save("ppo_knight_tour_8x8")
    
    # 测试训练好的模型
    test_trained_model(model)

def test_trained_model(model):
    env = KnightsTourEnv(board_size=8, render_mode='human')
    obs, _ = env.reset()
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        env.render()
    
    env.close()

if __name__ == "__main__":
    train_with_sb3(continue_training=False, model_path="ppo_knight_tour_8x8")
    print("Training complete and model saved.")
    print("You can now test the trained model by running the script again.")
    test_trained_model(PPO.load("ppo_knight_tour_8x8"))