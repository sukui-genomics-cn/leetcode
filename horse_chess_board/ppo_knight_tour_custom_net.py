import os

import torch as th
import torch.nn as nn
from typing import Dict, Tuple, Type, Union, List
from gym import spaces

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.policies import ActorCriticPolicy, ActorCriticCnnPolicy
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)

from knight_tour_env_gym import KnightsTourEnv


class CustomActorCriticPolicy(ActorCriticCnnPolicy):

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi, obs=obs)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob
    
    def _get_action_dist_from_latent(self, latent_pi: th.Tensor, obs=None) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)
        if obs is not None:
            visited_mask = self._get_visited_mask(obs)
            mean_actions = mean_actions - visited_mask * th.log(th.tensor(2.0))
        # mean_actions = th.softmax(mean_actions, dim=-1)  # 对于分类动作空间，应用softmax

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_pi)
        else:
            raise ValueError("Invalid action distribution")
        
    def _get_visited_mask(self, obs):
        """
        根据观测创建动作掩码：
        - 已访问的位置：权重衰减0.5倍（加上log(2)）
        - 非法的移动（超出边界）：权重衰减到接近0（加上大的负值）
        """
        # 计算每个动作的目标位置
        knight_moves = th.tensor([
            [2, 1], [1, 2], [-1, 2], [-2, 1],
            [-2, -1], [-1, -2], [1, -2], [2, -1]
        ], device=obs.device)
        act_nums = knight_moves.shape[0]
        batch_size = obs.shape[0]
        board_size = obs.shape[-2]
        penalty_mask = th.zeros(batch_size, act_nums, device=obs.device)
        
        # 获取当前骑士位置
        current_pos = th.argmax(obs[..., 1].flatten(start_dim=1), dim=1)
        current_x = current_pos // board_size
        current_y = current_pos % board_size
        current_pos = th.stack([current_x, current_y], dim=1)
        
        
        target_pos = current_pos.unsqueeze(1) + knight_moves.unsqueeze(0)
        
        # 检查目标位置是否合法和已访问
        for b in range(batch_size):
            for a in range(act_nums):
                x, y = target_pos[b, a]
                
                # 检查是否超出边界
                if not (0 <= x < board_size and 0 <= y < board_size):
                    penalty_mask[b, a] = 1e10  # 使非法动作概率接近0
                    continue
                    
                # 检查是否已访问
                if obs[b, int(x), int(y), 0] > 0.5:
                    penalty_mask[b, a] = 1e10  # 使非法动作概率接近0
                    
        return penalty_mask


def train_with_sb3(continue_training=False, model_path=None):
    log_dir = "./logs/ppo_knight_tour_cutompolicy_8x8"
    video_dir = "./videos/ppo_knight_tour_cutompolicy_8x8"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)

    # 创建并行环境
    env = make_vec_env(
        lambda: Monitor(KnightsTourEnv(board_size=64), log_dir),
        n_envs=4,
        seed=42,
        # vec_env_cls=VecTransposeImage  # 自动处理通道顺序
    )
    env = VecTransposeImage(env)  # 再添加转置包装
    
    # 添加视频录制（每10000步录制一次，最多录制100步）
    env = VecVideoRecorder(
        env,
        video_dir,
        record_video_trigger=lambda x: x % 100000 == 0,  # 录制频率
        video_length=65,  # 最大录制长度
        name_prefix="ppo_knight_tour_cutompolicy_8x8"
    )

    # 初始化PPO算法
    if continue_training and model_path and os.path.exists(model_path + ".zip"):
        print(f"Loading model from {model_path} to continue training...")
        model = PPO.load(model_path, env=env, tensorboard_log=log_dir)
    else:
        print("Initializing new model...")
        # policy_kwargs = get_policy_kwargs(env, features_dim=256, net_arch=dict(pi=[128,64], vf=[128,64]))
            # 增大网络规模
            # "net_arch": [
            #     {"vf": [256, 128], "pi": [256, 128]}  # 分离的policy和value网络
            # ],
            
            # # 关键参数：不共享特征提取器
            # "share_features_extractor": False,
            # }
        model = PPO(
            CustomActorCriticPolicy,
            env,
            verbose=1,
            tensorboard_log="./logs/",
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
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
        total_timesteps=1000000,
        tb_log_name="ppo_knight_tour_cutompolicy_8x8",
        callback=eval_callback,  # 包含自动保存功能
        reset_num_timesteps=not continue_training  # 是否重置步数计数器
        )
    
    # 保存模型
    model.save("ppo_knight_tour_cutompolicy_8x8")
    
    # 测试训练好的模型
    test_trained_model(model)

def test_trained_model(model):
    env = KnightsTourEnv(board_size=256, render_mode='human')
    obs, _ = env.reset()
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        env.render()
    
    env.close()

if __name__ == "__main__":
    train_with_sb3(continue_training=False, model_path="ppo_knight_tour_cutompolicy_8x8")
    print("Training complete and model saved.")
    print("You can now test the trained model by running the script again.")
    test_trained_model(PPO.load("ppo_knight_tour_cutompolicy_8x8"))