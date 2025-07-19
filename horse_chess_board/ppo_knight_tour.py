import os

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

from knight_tour_env_gym import KnightsTourEnv


def train_with_sb3(continue_training=False, model_path=None):
    log_dir = "./logs/ppo_knight_tour_8x8"
    video_dir = "./videos/ppo_knight_tour_8x8"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)

    # 创建并行环境
    env = make_vec_env(
        lambda: Monitor(KnightsTourEnv(board_size=8), log_dir),
        n_envs=4,
        seed=42
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
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log="./logs/",
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2
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
        tb_log_name="ppo_knight_tour_8x8",
        callback=eval_callback,  # 包含自动保存功能
        reset_num_timesteps=not continue_training  # 是否重置步数计数器
        )
    
    # 保存模型
    model.save("ppo_knight_tour_8x8")
    
    # 测试训练好的模型
    test_trained_model(model)

def test_trained_model(model):
    env = KnightsTourEnv(board_size=5, render_mode='human')
    obs, _ = env.reset()
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        env.render()
    
    env.close()

if __name__ == "__main__":
    train_with_sb3()
    print("Training complete and model saved.")
    print("You can now test the trained model by running the script again.")
    test_trained_model(PPO.load("ppo_knight_tour_8x8"))