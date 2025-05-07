# LeRobot

## Converting the dataset

You can use the included script for converting text-based MCP plans into LeRobot format.

Since installing of `lerobot` dependencies, seriously messes up the IsaacLab installation,
please be careful.

* Activate IsaacLab conda environment:

```bash
conda activate isaaclab-rsl
```

* Install minimum dependencies of lerobot:

```bash
git clone git@github.com:leggedrobotics/lerobot.git -b dev/mm/locomanipulation-dataset
cd lerobot && pip install . && cd ../
```

* Convert the dataset:

```bash
cd /PATH/TO/isaac-locoma-suite
python scripts/data_loader/convert_mcp_plans_to_hf5.py --headless
```

## Training different policies

* Make a clean `lerobot` conda environment. It messes up IsaacLab conda environment so we want to be careful.

```bash
conda create -y -n lerobot python=3.10
```

* Install lerobot with all its dependencies:

```bash
git clone git@github.com:leggedrobotics/lerobot.git -b dev/mm/locomanipulation
cd lerobot && pip install -e .
```

* Run the following from inside the `lerobot` repository:

```bash
python lerobot/scripts/train.py  \
    --config_path=/PATH/TO/isaac-locoma-suite/scripts/lerobot/train_diffusion_cfg.json
```

## Rolling out learned policies


* Diffusion policy:

```bash
python scripts/lerobot/play_lerobot.py \
    --config_path=${HOME}/Projects/isaaclab-locoma/isaac-locoma-suite/scripts/lerobot/play_diffusion_cfg.json \
    --policy.path=${HOME}/Projects/isaaclab-locoma/lerobot/outputs/train/diffusion/horizontal-door-opening/checkpoints/025000/pretrained_model
```


* VQ-BeT policy:

```bash
python scripts/lerobot/play_lerobot.py \
    --config_path=${HOME}/Projects/isaaclab-locoma/isaac-locoma-suite/scripts/lerobot/play_vqbet_cfg.json \
    --policy.path=${HOME}/Projects/isaaclab-locoma/lerobot/outputs/train/vqbet/horizontal-door-opening/checkpoints/025000/pretrained_model
```
