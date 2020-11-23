# Deep Reinforcement Quantization
This is the host of paper "deep reinforcement quantization".


**THIS IS NOT THE FINAL VERSION**

## Introduction
Deep Reinforcement Quantization uses a small neural network to perform Multi-Codebook Quantization.

![Framework]()

## Prerequisites
Before run, you should check following requirements.

The code is tested on several operating systems:
* Ubuntu 16.04+
* Windows 10
* CentOS 7+

Hardware requirements:
* Intel MKL-compatible CPU
* CUDA-compatible GPU, ≥8GiB VRAM
* ≥16GiB RAM

### Environment Setup
To run the demo, you can setup environment by two ways:

**A: Install from source**
1. Install Anaconda (Python 3).
2. Clone and install required packages.

```shell
git clone xxx
cd DReinQ/src
conda install -r requirements.yaml
pip install -e .
```

3. Test Installations


```shell
python misc/test.pay
```

**B: Pull from Docker**

TBD

## Train, Encode and Test
### Dataset Prepare
**A: Use public datasets**

* You can directly obtain them by download in the [release page]().
* Or you can manually download them from [SIFT1M](), [DEEP1M](), [LabelMe22K]().
    * After download, you should convert it to `*.npy` by `misc/ConvertDataset.py`.

After download, place them as follows:
```shell
DReinQ
|--data
|  |--SIFT
|  |  |--1M
|  |     |--train.npy
|  |     |--base.npy
|  |     |--query.npy
|  |     |--gt.npy
|  |--DEEP
|  |  |--1M
|  |     | ...
|  |--labelme
|  |  |-train.npy
|  |  | ...
```

**B: Use custom datasets**

You can arrange your file by following:
```shell
DReinQ
|--data
|  |--{YOUR_DATASET}
|  |  | # shape = [nTrain, D]
|  |  |--train.npy
|  |  | # shape = [nBase, D]
|  |  |--base.npy
|  |  | # shape = [nQuery, D]
|  |  |--query.npy
|  |  | # shape = [nQuery, 1]
|  |  |--gt.npy
```

Then, by just modifying config file, your dataset are in use:

```yaml
{
    ...
    "dataset": YOUR_DATASET
    ...
}
```

If this dataset structure can not satisfy you, you should manually create a class that inherits `torch.utils.data.Dataset` and modify source code.

### Train
To reproduce results in paper, you can directly use our provided configs that placed in `./configs/`.

For example:
```shell
python src/main.py --config configs/SIFT/32bits.json
```
This starts a train with 4 sub-codebooks, 256 codewords for each, on SIFT1M dataset.

Also, check the log by:
```shell
tensorboard --logdir saved/SIFT/1M/latest/
```

All options:
```shell
python src/main.py --help
```

### Encode and Evaluate
First locate where the saved model `saved.ckpt` placed. e.g. `saved/SIFT/1M/latest/`. Then, test the model by:
```shell
python src/main.py --path saved/SIFT/1M/latest/ --eval
```
If you want additional results, the encoded `B.npy` and `C.npy` is saved in the same directory.
