MineRL-NeurIPS-2020
==========================
**Codebase:** Basic implementation of RL algorithms from the [Medipixel](https://github.com/medipixel/rl_algorithms) repo, for the [MineRL](https://minerl.io/docs/) competition. Uses [W&B](https://www.wandb.com/) for logging network parameters (instructions [here](https://github.com/medipixel/rl_algorithms#wb-for-logging))

**Framework, language:** PyTorch 1.3.1, Python 3.7

**General Idea**: pick {env, algo} pair -> train to solve MineRL competition envs with RL or demonstration-based (IL, fD) algorithms

Prerequisites - For the competition
-------------
Standard installation (**non-LeNS lab systems**):

```
# create virtual environment (optional)
conda create -n myenv python==3.7
conda activate myenv # Windows
source activate myenv # Linux

git clone https://github.com/prabhasak/MineRL-NeurIPS-2020.git
cd MineRL-NeurIPS-2020

# install required libraries and modules (recommended)
make dep --ignore-errors # ignore certifi error
pip install minerl
```

Usage - For the competition
-------------
``run.py`` is the entrypoint script for Round 1 (**WIP**)

``python run.py --algo DQfD --cfg-path ./configs/MineRL_Treechop_v0/dqfd.py --demo-path "./data/minerltreechopvectorobf_disc_64_flat_20.pkl" --seed 42 -conv``

Prerequisites - For local usage (LeNS lab systems)
-------------
**Step 1:** Enable X11 forwarding
1. [Requirements](http://systems.eecs.tufts.edu/x11-forwarding/) for local machine. Verify with ``xeyes`` command on remote machine
2. Follow Sapana's [lab usage doc](https://docs.google.com/document/d/1oYzmTFAyv6qztkUMDFd0SW0w46ms7DAr9g-VIvIZIcQ/edit?usp=sharing) and ssh into your lab system

**Step 2:** Enable GPU usage
1. Check [``nvidia-smi``](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#cuda-application-compatibility), [``nvcc -v``](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions)
2. Check if [CUDA 10.0](https://pytorch.org/get-started/previous-versions/#linux-and-windows-6) is enabled (``python`` -> ``import torch`` -> ``torch.cuda.is_available()``)

**Step 3:** (Optional) Enable rendering on headless display
1. Install xvfbwrapper with ``conda install -c conda-forge xvfbwrapper``
2. Get display variable with ``echo $DISPLAY``. Add this and XAUTHORITY path to .bashrc (instructions [here](https://unix.stackexchange.com/questions/10121/open-a-window-on-a-remote-x-display-why-cannot-open-display/10126#10126) and [here](https://serverfault.com/questions/51005/how-to-use-xauth-to-run-graphical-application-via-other-user-on-linux/222591#222591))
3. run your .py with the ``xvfb-run`` prefix (note: [xvfb](https://www.x.org/releases/X11R7.6/doc/man/man1/Xvfb.1.xhtml#heading3) emulates a display using virtual memory)

**Step 4:** Set up local repo for running experiments

```
# create virtual environment (optional)
conda create -n lensminerl --clone root
source activate lensminerl
pip install --upgrade pip

git clone https://github.com/prabhasak/MineRL-NeurIPS-2020.git
cd MineRL-NeurIPS-2020

# install medipixel and mineRL dependencies
make dep --ignore-errors # ignore certifi error
conda install -c anaconda openjdk
pip install tensorboard matplotlib==3.0.3 cloudpickle==1.3.0 tabulate sklearn plotly common
pip install --upgrade minerl

# update jupyter notebook dependencies
pip install --upgrade --force jupyter-console --ignore-installed ipython-genutils
pip install wandb tornado==4.5.3 # without this, messes up a lot of things
pip install --upgrade nbconvert
```
**Step 5:** (Optional) Follow steps [here](https://github.com/medipixel/rl_algorithms#wb-for-logging) to create a [W&B](https://www.wandb.com/) account for logging network parameters. Remember to ``wandb login`` and ``wandb on`` to turn on syncing

**Step 6:** Download the [MineRL Dataset](https://minerl.io/docs/tutorials/data_sampling.html). Add ``MINERL_DATA_ROOT`` to your environment variables (Windows) or your bashrc file (Linux). You can also set the variable before running the code as follows:\
Windows: ``SET MINERL_DATA_ROOT=your/local/path`` (example: C:\MineRL\medipixel\data)\
Linux: ``export MINERL_DATA_ROOT="your/local/path"``

Usage
-------------
**Convert to discrete action space:** ``python run_k_means.py --env 'MineRLTreechopVectorObf-v0' --num-actions 64``

**Generate expert data:** Download the MineRL dataset and change the paths (lines 19-21). Some argument options: \
1. ``conv-vec``: Use continuous actions and only vector component of observations (not recommended)
2. ``conv-full``: Use discrete actions (``num-actions``) and both pov, vector of observations
3a. ``conv-full`` and ``flatten``: Flatten pov and append to vector to make state space \
3b. ``conv-full`` and ``aggregate``: Append vector as fourth channel of pov to make state space (pass through CNN) \

``python run_expert_data_format.py -conv-full -flatten --num-actions 64 --traj-use 10 --seed 42``

**View expert data:**
1. ``view-npz`` and ``view-pkl``: Available in both .npz and .pkl formats
2. Use [pdb commands](https://docs.python.org/3/library/pdb.html) to step through the data: ``n`` to view the next step, and ``q`` to quit the program

``python run_expert_data_format.py --view-full --view-pkl -flatten --num-actions 64 --traj-use 10 --seed 42``

**Competition envs:**

1. Prefix ``xvfb-run`` if running on a headless system
2. To sync wandb logging, remember to ``wandb login`` and ``wandb on``. Local logging enabled by default with ``--log``
3. Choose if CNN to be used with ``-conv`` (WIP). Discrete actions enabled by default with ``--is-discrete``

**RL:** ``python run_MineRL_Treechop_v0.py --env MineRLTreechopVectorObf-v0 --algo Rainbow-DQN --cfg-path ./configs/MineRL_Treechop_v0/dqn.py --num-actions 64 --seed 42``

**fD:** ``python run_MineRL_Treechop_v0.py --env MineRLTreechopVectorObf-v0 --algo DQfD --cfg-path ./configs/MineRL_Treechop_v0/dqfd.py --demo-path "./data/minerltreechopvectorobf_disc_32_flat_20.pkl" --seed 42``

3. **Basic envs:** **WIP**
