# Stochastic Adversarial Network for Multi-Domain Text Classification
Implementation of " Stochastic Adversarial Network for Multi-Domain Text Classification" in Pytorch

## Datasets
We used the same dataset as Conditional Adversarial Networks for Multi-Domain Text Classification (CAN) .So, You can download from https://github.com/YuanWu3/Conditional-Adversarial-Networks-for-Multi-Domain-Text-Classification/tree/main/data
Put the dataset into the corresponding folder: fdu-mtl、prep-amazon and w2v.

## Requirements:
-Python 3.6
-Pytorch 1.10
-Torchnet
-Scipy
-Tqdm

## Initialization:
To obtain the initial model, run and put it into the folder "./save/init_model":
### Experiment 1: MDTC on the multi-domain Amazon dataset
```bash
cd code/
python exp1_init.py --dataset prep-amazon --model mlp 
```
### Experiment 2: Multi-Source Domain Adaptation
```bash
cd code/
# target domain: books
python exp2_init.py --dataset prep-amazon --model mlp --no_wgan_trick --domains dvd electronics kitchen --unlabeled_domains books --dev_domains books
# target domain: dvd
python exp2_init.py --dataset prep-amazon --model mlp --no_wgan_trick --domains books electronics kitchen --unlabeled_domains dvd --dev_domains dvd
# target domain: electronics
python exp2_init.py --dataset prep-amazon --model mlp --no_wgan_trick --domains books dvd kitchen --unlabeled_domains electronics --dev_domains electronics
# target domain: kitchen
python exp2_init.py --dataset prep-amazon --model mlp --no_wgan_trick --domains dvd electronics kitchen --unlabeled_domains kitchen --dev_domains kitchen
```
### Experiment 3: MDTC on the FDU-MTL dataset
```bash
cd code/
python exp3_init.py --dataset fdu-mtl --model cnn --max_epoch 30
```

## Training
All the parameters are set as the same as parameters mentioned in the article. You can use the following commands to the tasks:
### Experiment 1: MDTC on the multi-domain Amazon dataset
```bash
cd code/
python exp1_with_pseu_label.py --dataset prep-amazon --model mlp
```
### Experiment 2: Multi-Source Domain Adaptation
```bash
cd code/
# target domain: books
python exp2_with_pseu_label.py --dataset prep-amazon --model mlp --no_wgan_trick --domains dvd electronics kitchen --unlabeled_domains books --dev_domains books
# target domain: dvd
python exp2_with_pseu_label.py --dataset prep-amazon --model mlp --no_wgan_trick --domains books electronics kitchen --unlabeled_domains dvd --dev_domains dvd
# target domain: electronics
python exp2_with_pseu_label.py --dataset prep-amazon --model mlp --no_wgan_trick --domains books dvd kitchen --unlabeled_domains electronics --dev_domains electronics
# target domain: kitchen
python exp2_with_pseu_label.py --dataset prep-amazon --model mlp --no_wgan_trick --domains dvd electronics kitchen --unlabeled_domains kitchen --dev_domains kitchen
```
### Experiment 3: MDTC on the FDU-MTL dataset
```bash
cd code/
python exp3_with_pseu_label.py --dataset fdu-mtl --model cnn --max_epoch 30
```
