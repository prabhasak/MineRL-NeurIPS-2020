MineRL-NeurIPS-2020
==========================
**Codebase:** Basic implementation of RL algorithms from the [Medipixel](https://github.com/medipixel/rl_algorithms) repo, for the [MineRL](https://minerl.io/docs/) competition. Uses [W&B](https://www.wandb.com/) for logging network parameters (instructions [here](https://github.com/medipixel/rl_algorithms#wb-for-logging))

**Framework, language:** PyTorch 1.3.1, Python 3.7

**Idea**: pick {env, algo} pair -> train RL or train IL (if expert data available)

Prerequisites
-------------

**For discrete action space (run K-means):** ``run_k_means.py --env 'MineRLTreechopVectorObf-v0' --num-actions 32``

Windows: ``SET MINERL_DATA_ROOT=your\local\path`` (example: C:\MineRL\medipixel\data)
Linux: ``MINERL_DATA_ROOT="your/local/path"``

```
# create virtual environment (optional)
conda create -n myenv python==3.7
conda activate myenv

git clone https://github.com/prabhasak/MineRL-NeurIPS-2020.git

# install required libraries and modules (recommended)
pip install -r requirements.txt
pip install minerl
```

Usage
-------------

**Competition envs:**

**RL:** ``python run_MineRL_Treechop_v0.py --cfg-path ./configs/MineRL_Treechop_v0/sac.py --seed 42 --log`` (use ``--off-render`` if running on headless system)

**fD:** **WIP**

**Basic envs:** **WIP**
