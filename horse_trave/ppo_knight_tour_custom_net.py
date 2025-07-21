import os

import torch as th

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
)

from knight_tour_env_gym import KnightsTourEnv


class CustomActorCriticPolicy(ActorCriticPolicy):

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
        """Create an action mask from observations"""
        # cal pos for every move
        knight_moves = th.tensor([
            [2, 1], [1, 2], [-1, 2], [-2, 1],
            [-2, -1], [-1, -2], [1, -2], [2, -1]
        ], device=obs.device)
        act_nums = knight_moves.shape[0]
        batch_size = obs.shape[0]
        board_size = obs.shape[-1]
        penalty_mask = th.zeros(batch_size, act_nums, device=obs.device)
        
        # current_pos
        current_pos = th.argmax(obs[:, 1].flatten(start_dim=1), dim=1)
        current_x = current_pos // board_size
        current_y = current_pos % board_size
        current_pos = th.stack([current_x, current_y], dim=1)
        
        target_pos = current_pos.unsqueeze(1) + knight_moves.unsqueeze(0)
        
        # check move
        for b in range(batch_size):
            for a in range(act_nums):
                x, y = target_pos[b, a]
                
                # check borad
                if not (0 <= x < board_size and 0 <= y < board_size):
                    penalty_mask[b, a] = 1e10
                    continue
                    
                # check vistied
                if obs[b, 0, int(x), int(y)] > 0.5:
                    penalty_mask[b, a] = 1e10
                    
        return penalty_mask


def train_with_sb3(
    continue_training=False,
    model_path=None,
    work_name="ppo_knight_tour_cutompolicy_mlp_8x8_0720_v2",
    output_dir="./"
):
    """Train PPO model for Knight's Tour problem.
    
    Args:
        continue_training: Whether to continue training from existing model
        model_path: Path to the model to continue training
        work_name: Identifier for this training session
        output_dir: Directory to save logs and models
    """
    # Setup directories
    log_dir = os.path.join(output_dir, "logs", work_name)
    video_dir = os.path.join(output_dir, "videos", work_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)

    # Create vectorized environment
    env = make_vec_env(
        lambda: Monitor(KnightsTourEnv(board_size=8), log_dir),
        n_envs=4,
        seed=42
    )
    
    # Add video recording
    env = VecVideoRecorder(
        env,
        video_dir,
        record_video_trigger=lambda x: x % 100000 == 0,
        video_length=65,
        name_prefix=work_name
    )

    # Initialize model
    if continue_training and model_path and os.path.exists(model_path + ".zip"):
        model = PPO.load(model_path, env=env, tensorboard_log=log_dir)
    else:
        model = PPO(
            CustomActorCriticPolicy,
            env,
            verbose=1,
            tensorboard_log=log_dir,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
        )

    # Setup evaluation callback
    eval_callback = EvalCallback(
        env,
        best_model_save_path=os.path.join(output_dir, "best_model"),
        log_path=log_dir,
        eval_freq=50000,
        deterministic=True,
        render=False
    )
    
    # Train model
    model.learn(
        total_timesteps=10_000_000,
        tb_log_name=work_name,
        callback=eval_callback, 
        reset_num_timesteps=not continue_training
    )
    
    model.save(os.path.join(output_dir, "models"))
    return model


def test_trained_model(model):
    """Test trained model with human rendering.
    
    Args:
        model: Trained PPO model to test
    """
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
    # Train and test model
    model = train_with_sb3(
        continue_training=True,
        model_path="/home/sukui/03.project/02.RL/leetcode/ckpts/best_model",
        output_dir="/home/sukui/03.project/02.RL/leetcode/outputs",
        work_name="ppo_knight_tour_cutompolicy_mlp_8x8_0720_v3"
    )
    test_trained_model(model)