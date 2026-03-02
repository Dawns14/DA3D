# K-Radar_release

This repository is for K-Radar related 3D perception experiments and includes a weather classifier training script.

## 1) Installation

For installation, it is recommended to **follow the K-Radar official repository README first**, especially for Python/PyTorch/CUDA version compatibility. For this repository, you can use the minimal setup below:

1. Create and activate a Python environment (preferably aligned with the official K-Radar setup).
2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Build CUDA extensions under `ops`:

```bash
cd ops
python setup.py develop
cd ..
```

> Note: If the build fails, it is usually caused by CUDA, PyTorch, or GCC version mismatch. Please align versions with the official K-Radar setup and try again.

---

## 2) Train Weather Classifier

Training script:

```bash
python models/img_cls/cls_train.py
```

Current weather classes (7):
- normal
- overcast
- fog
- rain
- sleet
- lightsnow
- heavysnow

### 2.1 Two Available Models

We provide two backbone models:

1. **ResNet18 (for DA3D)**
   - Model file: `models/img_cls/cls_model_resnet.py`
2. **CNN (for RL3DF)**
   - Model file: `models/img_cls/cls_model.py`

By default, `cls_train.py` imports the CNN version:

```python
from cls_model import ImageClsBackbone
```

To train ResNet18, change it to:

```python
from cls_model_resnet import ImageClsBackbone
```

### 2.2 Paths to Update Before Training

In `models/img_cls/cls_train.py`, the config path and checkpoint save path are hard-coded by default. Please update them for your environment:

- `path_cfg` (dataset/config file path)
- `torch.save(...)` (best model save path)

The script evaluates on the test split after each epoch and saves the checkpoint with the best accuracy.

---

## 3) Notes

- This repository depends on local K-Radar data organization. Make sure your dataset directory structure and split files are valid.
- If packages such as `open3d`, `numba`, or `easydict` are missing, install dependencies from `requirements.txt` and verify you are using the correct Python environment.

---

## 4) RTNH Example (Two-Stage Training)

Below is an example workflow using RTNH:

### Step 1. Train a mixed-weather detector

Use the RTNH base config to train a detector on mixed-weather data:

```bash
python main_train_0.py --config configs/cfg_RTNH_base.yml --gpus 0
```

After training, note the experiment directory under your logging path (e.g., `.../results/rtnh/exp_xxx_RTNH/`).

### Step 2. Train DA3D from the pretrained RTNH model

1. Open `configs/cfg_RTNH_da3d.yml`.
2. Set `GENERAL.RESUME.IS_RESUME: True`.
3. Set `GENERAL.RESUME.PATH_EXP` to the experiment directory from Step 1.
4. Optionally set `GENERAL.RESUME.START_EP` to the checkpoint epoch you want to load.

Then run:

```bash
python main_train_0.py --config configs/cfg_RTNH_da3d.yml --gpus 0
```
