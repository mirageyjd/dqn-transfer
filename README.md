Playing Atari with Transfer Learning and Deep Reinforcement Learning
====
This is a course project of **CS394R Reinforcement Learning** in UT Austin. 

Code Organization
----
Folder **dqn** implements standard Deep Q-Learning (DQN) algorithm [[1]](#references). Three experiments, including cartpole, Atari Pong, and Atari Tennis are provided. The training logs and models of experiments in our project are also provided in folder **dqn/results**.

Folder **transfer_pretrain** implements DQN used to pretrain an agent through transfer learning. The experiment of transfer learning pretraining from Atari Pong to Atari Tennis is provided. The code in **transfer_pretrain/UNIT** is from [[3]](#references), with a few modifications adapted to image-to-image translation between Atari games.

Folder **plot** provides a few graphs demonstrating our experiment results.

Requirements
----
- pytorch, torchvision
- gym[atari]
- tqdm

How to run it
----
If you want to run an experiment on, e.g, Atari Pong, with standard DQN (remember to replace $REPO_DIRECTORY with the path of repository):
```
export PYTHONPATH=$REPO_DIRECTORY/dqn-transfer:$PYTHONPATH
cd $REPO_DIRECTORY
python dqn/run_pong.py
```

If you want to pretrain a Tennis-playing model using transfer learning from a Pong-playing model:
```
export PYTHONPATH=$REPO_DIRECTORY/dqn-transfer:$PYTHONPATH
cd $REPO_DIRECTORY
python transfer_pretrain/transfer_pretrain_tennis.py
```

Customize experiment parameters
----
For each experiment, we provide a detailed config in the experiment file (e.g., the experiment file for Atari Pong is dqn/run_pong.py). The basic config includes experiment hyper-parameters as well as the path of log and model file. You can easily change most of the experimental settings by modifying the config.

The hyper-parameters in standard DQN are primarily from paper [[1]](#references), and partially fomr paper [[5]](#references). In order to recover some detailed settings, we also look at the code in [[2]](#references) and [[4]](#references).

Special Notice
----
- We implement a very simple but **memory-inefficient** version of replay buffer. If you run an experiment on Atari games with a replay buffer of size 1M, it may use roughly 50G memory. Please reduce the size of replay buffer if you have not enough memory.
- Our project focuses on Atari games. As a result, we just tested the experiment on Cartpole at the very beginning of our project. If you want to run an experiment on Cartpole, please remove "/ 255" (used for pixel value normalization) in **dqn/agent.py**, so that it will return a correct result.

References
----
[1] [Human-level control through deep reinforcement learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
[2] [DQN 3.0](https://github.com/deepmind/dqn)
[3] [UNIT: UNsupervised Image-to-image Translation Networks](https://github.com/mingyuliutw/UNIT)
[4] [Baselines](https://github.com/openai/baselines)
[5] [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298)
