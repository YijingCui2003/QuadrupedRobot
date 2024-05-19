# Install

```bash
conda activate cajun
cd rsl_rl && pip install -e . && cd ..
```

# 检验

```bash
# 简单测试
python -m src.robots.go1_robot_exercise_example --show_gui=True --use_gpu=True --num_envs=40

# clipped least square test
python -m src.controllers.go1_centroidal_body_controller_example --show_gui=True --use_gpu=True --num_envs=40
```

# 训练

```bash
# train
python -m src.agents.ppo.train --config=src/agents/ppo/configs/trot.py --logdir=logs --show_gui=True
# play
python -m src.agents.ppo.eval --logdir=logs/trot_pdhg/2024_04_20_20_04_06 --num_envs=40
```

# 部署

```bash
python -m src.agents.ppo.eval --logdir=logs/trot_pdhg/2024_05_17_17_13_09 --num_envs=1 --use_gpu=False --show_gui=False --use_real_robot=True --save_traj=False

# pronk
python -m src.agents.ppo.eval --logdir=example_checkpoints/pronk_cajun/ --num_envs=1 --use_gpu=False --show_gui=False --use_real_robot=True --save_traj=False
```


```bash
python -m src.agents.ppo.eval --logdir=logs/trot_pdhg/2024_05_17_17_13_09 --num_envs=10 --use_gpu=False --show_gui=True --use_real_robot=False --save_traj=False

```