import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
import os
import torch

# --- 설정 ---
MODEL_NAME = "ppo_walker2d_baseline_gpu"
LOG_DIR = f"logs/{MODEL_NAME}/"
TOTAL_TIMESTEPS = 3000000
SEED = 42

set_random_seed(SEED)

# 1. 로그를 저장할 폴더를 생성합니다.
os.makedirs(LOG_DIR, exist_ok=True)

# 2. "Walker2d-v5" 환경을 생성합니다.
env = gym.make("Walker2d-v5")

# 3. Monitor 래퍼로 환경을 감싸줍니다.
env = Monitor(env, LOG_DIR)

# 4. 사용 가능한 장치 확인 (GPU 우선)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# 5. 모델 생성하기
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    seed=SEED,
    device=device
)

# 6. 모델 학습시키기
model.learn(total_timesteps=TOTAL_TIMESTEPS)

# 7. 학습된 모델 저장하기
model.save(f"{MODEL_NAME}.zip")
print("모델 저장이 완료되었습니다.")

env.close()