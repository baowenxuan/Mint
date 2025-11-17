# Mint

[NeurIPS 2025] Mint: A Simple Test-Time Adaptation of Vision-Language Models against Common Corruptions. Wenxuan Bao*,
Ruxi Deng*, Jingrui He

## Key packages

```
torch          2.4.1
torchvision    0.19.1
clip           1.0
```

## Datasets

1. Downloads data from the [corruption benchmark](https://github.com/hendrycks/robustness) (CIFAR-10-C, CIFAR-100-C, and
   ImageNet-C) and extract them. 
2. Download `classnames.txt` from [this link](https://drive.google.com/file/d/1-61f_ol79pViBFDG_IDlUQSwoLcn2XXF/view). 
3. The directory structure should look like
```
${data_root}/corruption
│ 
├── CIFAR-10-C
│   ├── brightness.npy
│   ├── ...
│   ├── pixelate.npy
│   └── labels.npy
│ 
├── CIFAR-100-C
│   ├── brightness.npy
│   ├── ...
│   ├── pixelate.npy
│   └── labels.npy
│ 
└── CIFAR-100-C
    ├── brightness
    │   ├── 1
    │   ├── ...
    │   └── 5
    │       ├── n01440764
    │       │   ├── ILSVRC2012_val_00000293.JPEG
    │       │   ├── ...
    │       │   └── ILSVRC2012_val_00048969.JPEG
    │       ├── ...
    │       └── n07579787
    ├── ...
    ├── pixelate
    └── classnames.txt
```


## Experiments

To run experiments:

```shell
cd ./shell
./run_mint.sh
```
