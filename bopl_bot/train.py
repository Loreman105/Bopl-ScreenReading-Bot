"""Training entrypoints.

Keeps Stable-Baselines3 wiring separate from the environment logic.
"""

from __future__ import annotations

import os
import time

import cv2
import pydirectinput
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from . import config
from .env import BoplPureEnv


def train() -> None:
    if not os.path.exists(config.LOG_DIR):
        os.makedirs(config.LOG_DIR)
    if not os.path.exists(config.MODEL_DIR):
        os.makedirs(config.MODEL_DIR)

    env = BoplPureEnv()
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecFrameStack(vec_env, n_stack=4)

    model = PPO(
        "CnnPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=config.LOG_DIR,
        learning_rate=0.0003,
        ent_coef=0.01,
    )

    print("--- TRAINING STARTED (F8 pause, F9 resume, live cost plot) ---")
    print("Hotkeys: F8 = pause, F9 = resume")
    time.sleep(2)

    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path=config.MODEL_DIR,
        name_prefix="bopl_pure",
    )

    try:
        model.learn(
            total_timesteps=config.TOTAL_TIMESTEPS,
            callback=[checkpoint_callback],
        )
        model.save(f"{config.MODEL_DIR}/bopl_final")
    except KeyboardInterrupt:
        env._clip_cursor_disable()
        for k in ["a", "d", "space"]:
            pydirectinput.keyUp(k)
        for b in ["left", "right", "middle"]:
            pydirectinput.mouseUp(button=b)
        model.save(f"{config.MODEL_DIR}/bopl_interrupted")

    cv2.destroyAllWindows()

# Make train() the main entrypoint for `python -m bopl_bot.train`
def main() -> None:
    train()

if __name__ == "__main__":
    main()