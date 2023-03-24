

# Deep transfer learning for predicting frontier orbital energies of organic materials using small data and its application to porphyrin photocatalysts

Description
A deep transfer learning approach is used to predict HOMO/LUMO energies of organic materials with a small amount of training data.
DOI: [10.1039/D3CP00917C](https://pubs.rsc.org/en/Content/ArticleLanding/2023/CP/D3CP00917C)




### 上手指南

The code here is the code for model training validation prediction; the data here is the PBDD database, and the detailed data and specific partition of the MpPD dataset.
###### **安装步骤**

1. The PorphyBERT model in this work references *Prediction of chemical reaction yields using deep learning* (https://github.com/rxn4chemistry/rxn_yields)
2. The PorphyBERT model in this work references *Data augmentation strategies to improve reaction yield predictions and estimate uncertainty * (https://github.com/rxn4chemistry/rxnfp)
3. The PorphyDMPNN model in this work is derived from the *Analyzing learned molecular representations for property prediction* (https://chemprop.readthedocs.io/en/latest/index.html)



###### 配置

The training was completed on a desktop computer with a single NVIDIA RTX 3060 12 GB GPU, DDR4 32 GB RAM, and Intel Core i7-11700 CPU.

###### **架构**

1. Python 3.7
2. PyTorch 1.2.1









### 鸣谢
Thanks for the code reference for the following projects

1. [Prediction of chemical reaction yields using deep learning](https://github.com/rxn4chemistry/rxn_yields)
2. [Data augmentation strategies to improve reaction yield predictions and estimate uncertainty](https://github.com/rxn4chemistry/rxnfp)
3. [Analyzing learned molecular representations for property prediction](https://github.com/chemprop/chemprop)



