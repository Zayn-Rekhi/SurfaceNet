# SurfaceNet

**SurfaceNet** is a deep learning framework for estimating spatially-varying BRDFs (SVBRDFs) — including diffuse, normals, roughness, and specular maps — from a single image. Designed and implemented by [Zayn Rekhi](https://github.com/Zayn-Rekhi), the project uses PyTorch and a GAN-style architecture to produce realistic material reconstructions.

---

## 🔧 Features

- Predicts full SVBRDF maps from a single RGB input
- Adversarial loss and patch-based training for high-fidelity results
- Modular, extensible PyTorch codebase
- Includes training and evaluation scripts
- Supports Accelerate for multi-GPU and distributed training

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Zayn-Rekhi/SurfaceNet.git
cd SurfaceNet/src
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Make sure you have `accelerate` configured. If not:

```bash
accelerate config
```

---

## 🏋️‍♂️ Training

```bash
accelerate launch train.py   --tag run_name   --dataset /path/to/data   --logdir logs/
```

Options:
- `--tag`: A name for the run
- `--dataset`: Path to the training dataset
- `--logdir`: Directory to store logs and checkpoints

---

## 🧪 Inference / Evaluation

```bash
python eval.py   --ckpt path/to/checkpoint.ckpt   --input path/to/image.png   --size 256
```

The script will output predicted:
- Diffuse map
- Normal map
- Roughness map
- Specular map

---

## 📁 Directory Structure

```
SurfaceNet/
├── src/
│   ├── train.py
│   ├── eval.py
│   ├── models/
│   ├── utils/
│   └── ...
├── assets/
├── data/
└── README.md
```

---

## 📝 Notes

- Optimized for synthetic BRDF datasets, but can generalize to real-world data with proper preprocessing.
- Outputs are saved in `.png` format by default.
- Evaluation works on single images; batch inference is easy to enable.

---

## 🖼 Example Results

*(Coming soon – include side-by-side input and SVBRDF outputs here)*

---

## 📄 License

This project is licensed under the MIT License.  
Feel free to use, modify, and contribute.

---

## 🙌 Acknowledgments

Thanks to the open-source PyTorch and GAN communities for foundational tools and techniques.
