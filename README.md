
# RS-VLM: Vision–Language Guided Small Object Detection for Remote Sensing

RS-VLM is a vision–language guided detection framework designed for **fine-grained small object recognition in remote sensing imagery**.  
The model integrates **multiscale geometric modeling, oriented detection, and prompt-conditioned semantic alignment** to enable **open-vocabulary, zero-shot, and few-shot detection** under extreme scale variation and limited supervision.

This repository provides the **official PyTorch implementation** of RS-VLM.

---

## 🔍 Key Features

- Vision–language guided object detection for remote sensing
- Oriented and multiscale small-object modeling
- Open-vocabulary and zero-shot detection via textual prompts
- Strong cross-dataset generalization
- Real-time inference efficiency
- Designed for future **edge deployment** and **transfer learning**

---

## 📁 Repository Structure

```text
RS-VLM/
├── models/                 # RS-VLM model architecture
├── datasets/               # Dataset loaders (DOTA, xView, FAIR1M, etc.)
├── losses/                 # Detection and alignment losses
├── utils/                  # Utility functions (logging, metrics, helpers)
├── configs/                # YAML configuration files
├── checkpoints/            # Saved models (ignored by git)
├── train.py                # Training script
├── evaluate.py             # Evaluation on seen categories
├── zero_shot_eval.py       # Zero-shot / prompt-based evaluation
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── .gitignore

```
⚙️ Environment Setup
1. Clone the Repository
git clone https://github.com/<your-username>/RS-VLM.git
cd RS-VLM

2. Create a Virtual Environment
```
conda create -n rsvlm python=3.9 -y
conda activate rsvlm
```


or using venv:
```
python3 -m venv rsvlm
source rsvlm/bin/activate
```
3. Install Dependencies
```
pip install -r requirements.txt
```
🧠 Training

To train RS-VLM using a configuration file:
```
python train.py --config configs/rs_vlm_base.yaml
```

Configuration files control:

Backbone selection

Vision–language encoder

Prompt strategy

Dataset paths

Training hyperparameters

📊 Evaluation
Standard Evaluation
```
python evaluate.py --config configs/rs_vlm_base.yaml --ckpt checkpoints/model.pth
```
Zero-Shot / Open-Vocabulary Evaluation
```

python zero_shot_eval.py \
  --config configs/rs_vlm_base.yaml \
  --ckpt checkpoints/model.pth \
  --prompts configs/prompts.txt
```
🧪 Supported Datasets

DOTA

xView

FAIR1M

Additional datasets can be added via custom loaders in datasets/

Ensure datasets follow the expected directory structure defined in config files.

🚀 Reproducibility

For reproducibility:

Fix random seeds in config files

Use the provided YAML configurations

Refer to paper-reported settings for evaluation

🔮 Future Work

Deployment and benchmarking on edge devices

Lightweight backbone variants

Transfer learning and continual learning

Expansion to more diverse remote sensing datasets

Future updates and extensions will be shared through this repository.
