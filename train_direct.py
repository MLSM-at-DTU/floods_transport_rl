#!python
"""
#
# @file: train_direct.py
# @brief: Run RL training for the MAAT environment
# ---
# @website: 
# @repo: https://github.com/MLSM-at-DTU/floods_transport_rl
# @author: Miguel Costa
# MAAT environment
# ---
"""

# General imports
import importlib
import gymnasium as gym

# Importing RL libraries
from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env

# Importing MAAT environment
import maat.world as world

# Filter out the FutureWarning
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import logging
logging.getLogger('climada').setLevel(logging.ERROR)  # to silence all warnings

# CONSTANTS
DATA_LOCATION_LOCAL = "data" # data in this folder does not reflect the original data used, but its rather just used for demonstration purposes
DATA_LOCATION_O_DRIVE = "data" # data in this folder does not reflect the original data used, but its rather just used for demonstration purposes

CRS_25832 = 'epsg:25832'
CRS_4326 = 'epsg:4326'


importlib.reload(world)

env_id = 'maat/BasicEnvironment-v1' # It is best practice to have a space name and version number.

gym.envs.registration.register(
    id=env_id,
    entry_point=world.BasicEnvironment,
    max_episode_steps=100,
    reward_threshold=0 
)
env = make_vec_env(
    env_id, 
    n_envs=1, 
    seed=0, 
    env_kwargs={
        "assets_types": ["network"],
        "render_mode": "ansi",
        "hazard_sampling_scheme": world.HazardSampling.DETERMINISTIC,
        "preload_hazards": True,
        "hazards_to_preload": [160],
        "city_zones": "IndreBy",
        "episode_time_steps": 77,
        })

model = MaskablePPO("MlpPolicy", 
                    env, 
                    learning_rate=0.01,
                    n_steps=48,
                    batch_size=48,
                    n_epochs=4,
                    ent_coef=0.1,
                    tensorboard_log="./tensorboard/",
                    verbose=1)

model.learn(total_timesteps=8000,
            log_interval=3, 
            progress_bar=False, 
            tb_log_name="MaskablePPO_77ts_IndreBy_", 
            # callback=[checkpoint_callback, eval_callback],
            )

model.save("models/MaskablePPO_77ts_IndreBy_withrain_3000")
# del model # remove to demonstrate saving and loading
print("model saved:", "models/MaskablePPO_77ts_IndreBy_withrain_3000")












