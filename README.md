# TAMOLS - RL

Combined hierarchical controller with TAMOLS footstep planner during inference over an RL low level controller. I want a demo y'all.

### Installation

1. Clone the repo
2. Create a new python virtual env with python 3.6, 3.7 or 3.8 (3.8 recommended) - use conda
3. Install pytorch 1.10 with cuda-11.3:

   ```
   pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

   ```
4. Install Isaac Gym

   - Download and install Isaac Gym Preview 4 from [https://developer.nvidia.com/isaac-gym](https://developer.nvidia.com/isaac-gym)
   - `cd isaacgym/python && pip install -e .`
   - Try running an example `cd examples && python 1080_balls_of_solitude.py`
   - For troubleshooting check docs isaacgym/docs/index.html
5. Install rsl_rl (PPO implementation)

   - `cd rsl_rl && pip install -e .`

5. Install go2-hrl (Legged gym with unitree assets)

   - `cd go2-hrl && pip install -e .`

6. Install fetch requirements

   - `cd fetch/tamols`
   - try to run test.py, if missing a library, pip install (like Drake, pip install drake for example)
   - two specific versions necessary:
   - `pip install numpy==1.21.0`
   - `pip install setuptools==59.5.0`

### Usage

Train and play below are the same. If you edit map_size or cell_size for the solver, make sure to do it in both tamols.py and legged_robot.py (its a class variable)

Feel free to create different terrains during training, we should be robust to many terrains now, and if anything it would probably be better for learning the gait.

1. Train:
   `python legged_gym/scripts/train.py --task=go2`

   * To run on CPU add following arguments: `--sim_device=cpu`, `--rl_device=cpu` (sim on CPU and rl on GPU is possible).
   * To run headless (no rendering) add `--headless`.
   * **Important** : To improve performance, once the training starts press `v` to stop the rendering. You can then enable it later to check the progress.
   * The trained policy is saved in `logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`. Where `<experiment_name>` and `<run_name>` are defined in the train config.
   * The following command line arguments override the values set in the config files:
   * --task TASK: Task name.
   * --resume: Resume training from a checkpoint
   * --experiment_name EXPERIMENT_NAME: Name of the experiment to run or load.
   * --run_name RUN_NAME: Name of the run.
   * --load_run LOAD_RUN: Name of the run to load when resume=True. If -1: will load the last run.
   * --checkpoint CHECKPOINT: Saved model checkpoint number. If -1: will load the last checkpoint.
   * --num_envs NUM_ENVS: Number of environments to create.
   * --seed SEED: Random seed.
   * --max_iterations MAX_ITERATIONS: Maximum number of training iterations.
2. Play:`python legged_gym/scripts/play.py --task=go2`

   * By default, the loaded policy is the last model of the last run of the experiment folder.
   * Other runs/model iteration can be selected by setting `load_run` and `checkpoint` in the train config.
