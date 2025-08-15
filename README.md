# 🚀 GPT-OSS – Pure PyTorch Reimplementation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1sDJUWJNbdxZoMGGoY7YTWwLyE0L8lGon?usp=sharing)


<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.x-red?style=for-the-badge&logo=pytorch" />
  <img src="https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python" />
</p>

<p align="center">
    <img src="./assets/img1.png">
    (img credit ChatGPT)
</p>


**GPT-OSS**, built entirely in **PyTorch** — a beginner-friendly implementation designed to enhance model understanding for future AI researchers. 

Without Inheriting from anything other than `nn.Module` 😁



---

## ✨ Features

* **Pure PyTorch** – No heavy dependencies beyond PyTorch & Hugging Face utilities.
* **Mixture-of-Experts (MoE)** with Top-K routing for scalable inference.
* **RMSNorm** and **Rotary Position Embeddings**.
* **Custom Attention** – supports eager.
* **Easy to Modify** – great for research and teaching.
* In **Pytorch**
---

## 🏗️ Architecture Overview

**Core Components**

* `GptOssRMSNorm` – RMS normalization without bias.
* `GPTOssExperts` – MoE feed-forward experts.
* `GPTOssTopKRouter` – Selects top-k experts per token.
* `GptOssAttention` – Multi-head attention with rotary embeddings.
* `GptOssDecoderLayer` – Standard GPT-style transformer layer.
* `GptOssModel` – Stack of decoder layers + embedding + final norm.
* `GPTOssModelFull` – Full model with language modeling head.

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/gpt-oss.git
cd gpt-oss
```

---
## Model Configurations

We provide three predefined model configurations:

1. **gpt-oss-20b** – 20B parameter variant (`GPTOss_120B`)
2. **gpt-oss-120b** – 120B parameter variant (`GPTOss_20B`)
3. **gpt-oss-small** – A lightweight version with a reduced number of hidden layers (`GPTOss_Small`)

---

## 🚀 Quick Start

```python
import torch
from config import GPTOss_Small
from layers import GPTOssModelFull  # your main file

# 1. Load Config
config = GPTOss_Small()

# 2. Load Model
model = GPTOssModelFull(config)

# 3. Define Inputs
batch_size = 2
seq_len = 5 # for small model testing
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
attention_mask = torch.ones(batch_size, 1, seq_len, seq_len)
position_ids = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)  # shape: [batch_size, seq_len]

# 4. Run forward pass
logits = model(input_ids=input_ids, attention_mask=attention_mask,position_ids=position_ids,use_cache=False)

print("Input IDs shape:", input_ids.shape)
print("Logits shape:", logits.shape)  # (batch_size, seq_len, vocab_size)
```

---

## 🧠 Key Design Choices

* **Modular** – Every component (attention, MLP, norm, router) is a standalone class.
* **Readable** – Code avoids excessive abstraction for clarity.
* **Flexible Routing** – Swap out `GPTOssTopKRouter` for other routing strategies.
* **Custom Attention Backend** – Can integrate with flash attention or other kernels.

---

## 📊 Simple Training Script

```python
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=3e-4)
loss_fn = torch.nn.CrossEntropyLoss()

for step in range(100):
    logits = model(input_ids=input_ids)
    loss = loss_fn(logits.view(-1, config.vocab_size), input_ids.view(-1))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"Step {step}, Loss: {loss.item():.4f}")
```

---

## 🛠️ Roadmap

* [ ] Training Support Purely in Pytorch [SFT,PPO,GRPO]
* [ ] CPU Offloading  
---

## 🙏 Credits
Special thanks to the Hugging Face Transformers team for their exceptional open-source work. This Reimplementation is primarily inspired by Hugging Face Transformers integration but presented in a much simpler, more beginner-friendly way for easier understanding.

## 🙌 Inspiration

This work draws significant inspiration from Andrej Karpathy’s minGPT.


---

## 🤝 Contributing

Pull requests, bug reports, and feature suggestions are welcome!
If you improve the architecture or add new routing strategies, please share them with the community.

---

## 🌟 Star the Repo

If you find this project useful, please ⭐ it on GitHub – it helps more people discover it.
