# DeepMHCI
DeepMHCI: A Novel Anchor Position-Aware Deep Interaction Model for Accurate MHC I-peptide Binding Affinity Prediction

## Requirements
* python == 3.8.8
* pytorch == 1.7.1
* numpy == 1.19.2
* scipy == 1.6.1
* scikit-learn == 0.24.1
* click == 7.1.2
* ruamel.yaml == 0.16.12
* tqdm == 4.56.0
* logzero == 1.6.3

## Experiments
The commands corresponding to the different experiments are shown below.
1. Train 10 models to ensemble for five-fold cross-validation.
2. Test on testsets with 10 models (after 5cv training).
3. Test on the epitope dataset with 10 models (after 5cv training).
4. Output the top 1% predicted binders to draw sequence logos.

```
python main.py -d config/data.yaml -m config/model.yaml --mode 5cv -s 0 -n 10 --eval_len
python main.py -d config/data.yaml -m config/model.yaml --mode test-5cv -s 0 -n 10 --eval_len
python main.py -d config/data.yaml -m config/model.yaml --mode epitope -s 0 -n 10 --eval_len
python main.py -d config/data.yaml -m config/model.yaml --mode seq2logo -s 0 -n 10 --allele HLA-A1101
```
## Server
The DeepMHCI server is publicly available at https://dmiip.sjtu.edu.cn/DeepMHCI.

## Declaration
It is free for non-commercial use. For commercial use, please contact Mr.Wei Qu and Prof.Shanfeng Zhu (zhusf@fudan.edu.cn).
