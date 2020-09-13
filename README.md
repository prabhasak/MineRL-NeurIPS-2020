MineRL-NeurIPS-2020
==========================
**Codebase:** Basic implementation of RL algorithms from the [Medipixel](https://github.com/medipixel/rl_algorithms) repo, for the [MineRL](https://minerl.io/docs/) competition. Uses [W&B](https://www.wandb.com/) for logging network parameters (instructions [here](https://github.com/medipixel/rl_algorithms#wb-for-logging))

**Framework, language:** PyTorch 1.3.1, Python 3.7

**Idea**: pick {env, algo} pair -> train RL or train IL (if expert data available) algo on MineRL envs

Prerequisites
-------------
For standard installation (for non-LeNS lab systems):

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

For the competition (for LeNS lab systems):

**Step 1:** Enable X11 forwarding
1. [Requirements](http://systems.eecs.tufts.edu/x11-forwarding/) for local machine. Verify with ``xeyes`` command on remote machine
2. Follow Sapana's [lab usage doc](https://docs.google.com/document/d/1oYzmTFAyv6qztkUMDFd0SW0w46ms7DAr9g-VIvIZIcQ/edit?usp=sharing) and ssh into your lab system

**Step 2:** Check [``nvidia-smi``](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#cuda-application-compatibility), [``nvcc -v``](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions), and if [CUDA 10.0](https://pytorch.org/get-started/previous-versions/#linux-and-windows-6) is enabled (``python`` -> ``import torch`` -> ``torch.cuda.is_available()``)

**Step 3:** (Optional) If the system does not have a monitor, you will need to enable rendering on headless display
1. Install xvfbwrapper with ``conda install -c conda-forge xvfbwrapper``
2. Get display variable with ``echo $DISPLAY``. Add this and XAUTHORITY path to .bashrc (instructions [here](https://unix.stackexchange.com/questions/10121/open-a-window-on-a-remote-x-display-why-cannot-open-display/10126#10126) and [here](https://serverfault.com/questions/51005/how-to-use-xauth-to-run-graphical-application-via-other-user-on-linux/222591#222591))
3. run your .py with the ``xvfb-run`` prefix (note: [xvfb](https://www.x.org/releases/X11R7.6/doc/man/man1/Xvfb.1.xhtml#heading3) emulates a display using virtual memory)

**Step 4:** Setup local repo for running experiments

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

Usage
-------------
**For discrete action space (run K-means):** ``python run_k_means.py --env 'MineRLTreechopVectorObf-v0' --num-actions 32``

Add ``MINERL_DATA_ROOT`` to your environment variables (Windows) or your bashrc file (Linux). You can also set the variable before running the code as follows:
Windows: ``SET MINERL_DATA_ROOT=your/local/path`` (example: C:\MineRL\medipixel\data)\
Linux: ``export MINERL_DATA_ROOT="your/local/path"``

**Competition envs:**

**RL:** ``python run_MineRL_Treechop_v0.py --cfg-path ./configs/MineRL_Treechop_v0/sac.py --seed 42 --off-render`` (prefix ``xvfb-run`` if running on headless system)

**fD:** **WIP**

**Basic envs:** **WIP**
