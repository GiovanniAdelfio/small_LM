# Efficient Small Language Model (<30M Parameters)

> **Status:** Work in Progress

## Project Overview
The goal of this project is to develop a text generation model with fewer than **30 million parameters** that remains syntactically and semantically coherent while minimizing resource usage.

## Methodology
Following the approach outlined in the **"TinyStories"** paper, we utilize a dataset with a limited vocabulary (**DailyDialog**) to maximize performance relative to model size.

We evaluate three distinct training strategies:
* Training from scratch exclusively on the limited vocabulary dataset.
* Training from scratch on a larger, general dataset followed by finetuning on the target dataset.
* Finetuning an existing pre-trained model.

## Optimization Techniques
To further reduce the computational footprint and enable deployment on constrained hardware, the following optimization techniques are applied:
* **Quantization** (Post-training and/or Quantization-Aware Training)
* **Transformer Pruning**
* **Knowledge Distillation**

### References
* Eldan, R., & Li, Y. (2023). *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?* [arXiv:2305.07759](https://arxiv.org/abs/2305.07759)

---
*Authors: Giovanni Adelfio, Elena Mannoni*
