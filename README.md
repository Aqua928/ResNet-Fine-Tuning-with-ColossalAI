# ResNet-Fine-Tuning-with-ColossalAI

ResNet, short for Residual Networks, is a type of neural network architecture that facilitates deeper training by using skip connections or shortcuts to jump over some layers. These connections help mitigate the vanishing gradient problem, enabling the network to learn faster and more effectively. Utilizing ColossalAI, researchers can efficiently manage data and model parallelism, optimizing training procedures across multiple GPUs. This method not only enhances the fine-tuning process in terms of speed and scalability but also improves model performance by enabling precise adjustments to the network's layers and learning rates, tailored to the nuances of the targeted task.

## Install requirements

```bash
pip install -r requirements.txt
```

## Dataset

Use CIFAR-10 as the sample dataset, under the `./data`. CIFAR-10 is a popular image classification dataset used in machine learning. It contains 60,000 32x32 color images, divided into 10 classes with 6,000 images per class. These classes include airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. The dataset is split into 50,000 training images and 10,000 testing images, making it ideal for developing and testing machine learning models in computer vision.

## Training

Simply run:

```bash
# train with torch DDP with fp32
colossalai run --nproc_per_node 2 train.py -c ./ckpt-fp32

# train with torch DDP with mixed precision training
colossalai run --nproc_per_node 2 train.py -c ./ckpt-fp16 -p torch_ddp_fp16
```

## Experimental Logs and Checkpoints

The logs of the experiments I ran are in the `log.ipynb` by Google Colab, and the checkpoints are in the `./ckpt-fp16`directory.