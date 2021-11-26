# Sci-Summarizer
A Notebook which can be used for training **T5, Bart, Pegasus, Prophetnet on Summarization Dataset** on limited compute resources on Kaggle or Colab. This notebook is capable of training large models using either TPU or GPU.

Kaggle Notebook: https://www.kaggle.com/priyammehta/hf-deepspeed

This is achieved via 2 solutions:
1. For TPU, I have implemented a `custom TPU Trainer` class which inherits HuggingFace Trainer module. My custom class implements `xser` module from PyTorch. More information given below.
2. For GPU, I have provided functionality for `DeepSpeed`. DeepSpeed helps in bringing forth the full-potential of a single GPU. It helps in training large models with effective batch-size via different concept like parameter offloading, optimizer offloading etc.

#### Indepth explanation for point 1.
Custom trainer implements the `xser` module from PyTorch. When saving a model with PyTorch-XLA, the model will first be brought back to Host (CPU). Inorder to do this, the model needs to be stored in Local Memory (RAM). When performing this on large models like T5-Large, there is a chance of running in OOM issue due to lack of memory. Inorder to mitigate this issue, PyTorch introduced the `xser` module. Instead of bringing the entire model back to local ram in a single go, `xser` module will split the model into parts and bring the weights in batches. This reduces the memory footprint and as a result helps in saving large models trained on TPU when limited host memory is available. This functionality was not present in HuggingFace Trainer so I implemented a small module of my own.
