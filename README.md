# GPT Attentions Comparison

This repository contains experiments comparing different attention mechanisms in Transformer-based models. We have implemented various attention methods—including the classic Transformer (vanilla), Linformer, Nystromformer, and Performer—and integrated them into a miniature GPT model (nanoGPT, ~12M parameters). The nanoGPT model is used to train on the complete works of Shakespeare, demonstrating that our custom-built attention modules are fully compatible with transformer architectures.

## Overview

- **nanoGPT:**  
  A small GPT model included in this repository, which is trained on a character-level Shakespeare dataset. nanoGPT serves as our testbed for evaluating different attention methods in a real Transformer setting.

- **Attention Methods:**  
  We compare the following attention mechanisms:
  - **Transformer (Vanilla Attention)**
  - **Linformer** (with different projection dimensions)
  - **Nystromformer** (with varying numbers of landmarks)
  - **Performer** (using random feature approximations)

- **Metrics:**  
  The experiments measure:
  - Running time (in seconds) to compute the attention matrices at different sequence lengths.
  - Memory consumption (in MB) during attention computation.
  - Cosine similarity between the approximated and exact (Transformer) attention matrices.

- **Notebook Purpose:**  
  The provided Jupyter Notebook (`attention-is-all-you-need.ipynb`) is used for aggregating, analyzing, and visualizing these metrics. It helps you compare the efficiency and approximation quality of the different attention mechanisms.

## Repository Structure

- **config/**  
  Contains configuration files (e.g., `train_shakespeare_char.py`) where key parameters are defined:
  - `block_size`: the context window (number of tokens).
  - `max_iters`: total number of training iterations.
  - `attention_type`: selects the attention method (e.g., `vanilla`, `linformer`, `nystrom`, `performer`).

- **data/shakespeare_char/**  
  Contains the script to prepare the Shakespeare character-level dataset.

- **data/attention-is-all-you-need.ipynb**  
  A Jupyter Notebook for processing and visualizing metrics such as running time, memory usage, and cosine similarity across different attention methods.
  
- **model.py**  
  Defines the Transformer model (nanoGPT) which uses the selected attention mechanism.

- **train.py**  
  The training script that reads the configuration, trains nanoGPT on the Shakespeare dataset, and logs metrics like loss, running time, memory, and similarity.

- **sample.py**  
  A script for generating text from the trained nanoGPT model.

- **attention/**  
  Contains the implementations of various attention mechanisms (including both fully vectorized and stateful versions of Performer attention).


## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/klaipher/GPTAttentionsComparison.git
   cd GPTAttentionsComparison
   ```

2. **Install Dependencies:**
   Ensure you have Python 3.8+ and CUDA installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Data Preparation
Prepare the Shakespeare dataset:
```bash
python data/shakespeare_char/prepare.py
```

### Training
Train nanoGPT with your desired attention method by modifying the config file (e.g., `config/train_shakespeare_char.py`) or by using command-line overrides. For example, to train with Nystromformer (with 32 landmarks):
```bash
python train.py config/train_shakespeare_char.py --max_iters=1500 --block_size=512 --device='cuda' --attention_type='nystrom'
```

### Metrics and Analysis
Open `attention-is-all-you-need.ipynb` to:
- Aggregate and analyze metrics (running time, memory, similarity).
- Visualize performance comparisons between the different attention methods.
