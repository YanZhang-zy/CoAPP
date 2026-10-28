# Toward Compact and Adaptive Image Dehazing

## :desktop_computer: Environment

Tested with Python 3.10, PyTorch 2.1.0, and torchvision 0.16.0.

```bash
conda create -n CoAPP python=3.10 -y
conda activate CoAPP
pip install -r requirements.txt
```

## :book: Data Preparation

```text
datasets/
├── THaze/
│   ├── train/
│   │   ├── hazy/
│   │   └── clear/
│   └── val/
│       ├── hazy/
│       └── clear/
└── REAL_FOGGY/
    ├── image_001.png
    └── ...
```

## :floppy_disk: Pretrained Models and Prompts

Download the training resources from [Baidu Netdisk](https://pan.baidu.com/s/1hINu3suhWb3wWVfUWoWlnQ?pwd=0519) (access code: `0519`) and
place them as follows before running the adaptation stage:

```text
clip_model/
├── haze_prompt.pth
└── ViT-B-32.pt
```

## :hotsprings: Model Training

### 1. Train the teacher

```bash
bash training/teacher.sh \
  --train-dir datasets/THaze/train \
  --val-dir datasets/THaze/val \
  --output experiment/Teacher
```

### 2. Distil the student

```bash
bash training/kd.sh \
  --train-dir datasets/THaze/train \
  --val-dir datasets/THaze/val \
  --teacher-checkpoint experiment/Teacher/last.pth \
  --output experiment/Student
```

### 3. Adapt on real hazy images

```bash
bash training/ema.sh \
  --train-dir datasets/REAL_FOGGY \
  --val-dir datasets/THaze/val \
  --student-checkpoint experiment/Student/last.pth \
  --output experiment/CoA++
```

### 4. Convert for deployment

```bash
bash training/reparam.sh \
  --input experiment/CoA++/last.pth \
  --output weight/CoAPP_trained.pth
```

## :taxi: Model Testing

Run inference from the project root using the supplied CoAPP weights:

```bash
bash eval.sh \
  --input datasets/REAL_FOGGY \
  --output outputs/CoAPP \
  --weights weight/CoAPP.pth \
  --device cuda:0
```

### Inference with MUSIQ

To optionally score the generated images, install the MUSIQ dependency:

```bash
pip install pyiqa==0.1.11
```

Add `--musiq` to run inference and MUSIQ on the selected device:

```bash
bash eval.sh \
  --input datasets/REAL_FOGGY \
  --output outputs/CoAPP \
  --weights weight/CoAPP.pth \
  --device cuda:0 \
  --musiq
```

Results are saved under the specified output directory: `visualization/` contains the generated PNG images, and `metrics/` contains `musiq_scores.csv` and `musiq_summary.json` when MUSIQ is enabled.


## :clipboard: Acknowledgments

This code builds on
[DEA-Net](https://github.com/cecret3350/DEA-Net), and
[OpenAI CLIP](https://github.com/openai/CLIP).
The learned prompt methodology follows
[CLIP-LIT](https://github.com/ZhexinLiang/CLIP-LIT).