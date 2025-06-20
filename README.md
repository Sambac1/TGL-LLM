# TGL-LLM: Integrate Temporal Graph Learning into LLM-based Temporal Knowledge Graph Model

This is the official implementation repository of our paper:  
**Integrate Temporal Graph Learning into LLM-based Temporal Knowledge Graph Model**.

---

## 🔧 Setup

### Environment Setup

```bash
conda create -n tgl
conda activate tgl
pip install -r requirements.txt
````

### Download Data and Checkpoints

Please download the following files from [Google Drive](https://drive.google.com/drive/folders/1d5KwMRyTR64_1olvyLJq4i8nMgNUEeoc?usp=sharing):

* `data.zip`
* `checkpoints.zip`

Then unzip them into the root directory:

```bash
unzip data.zip
unzip checkpoints.zip
```

#### Folder Descriptions

* `data/`: Contains all datasets required for training, validation, and testing.
* `checkpoints/`: Contains model checkpoints. These can be directly used to **reproduce the main results** reported in our paper without retraining from scratch.

---

## 🚀 Usage

### 1. Create Graph Data

```bash
python generate_graphs.py --dataset IR
```

Supported datasets: `IR`, `IS`, `EG`.

### 2. Temporal Graph Learning

```bash
python3 train.py -d IR -g YOUR_GPU_NUM --lr 1e-3 --wd 1e-6 --model REGCN
```

### 3. Data Pruning

```bash
python3 prune.py -d IR -g 3 --lr 1e-3 --wd 1e-6 --model REGCN
```

---

### 4.  Model Training 

### Stage 1: 

```bash
CUDA_VISIBLE_DEVICES=0 python3 train_llm.py -d IR -o train -k 9 -rs
```

### Stage 2: 

```bash
CUDA_VISIBLE_DEVICES=0 python3 train_llm.py -d IR -o train -k 9 -a -p PATH_TO_STAGE1_CHECKPOINTS
```

---

## 🧪 Testing

```bash
CUDA_VISIBLE_DEVICES=2 python3 train_llm.py -d IR -o test -k 9 -p PATH_TO_STAGE2_CHECKPOINTS
```

---

## 📄 Citation

If you find our work helpful, please consider citing:

```bibtex
@misc{chang2025TGL, 
  title={Integrate Temporal Graph Learning into LLM-based Temporal Knowledge Graph Model}, 
  author={He Chang and Jie Wu and Zhulin Tao and Yunshan Ma and Xianglin Huang and Tat-Seng Chua},
  year={2025},
  eprint={2501.11911},
  archivePrefix={arXiv},
  primaryClass={cs.IR},
  url={https://arxiv.org/abs/2501.11911},
}
```