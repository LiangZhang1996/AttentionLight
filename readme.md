## 1. Introduction

Offical code for article [AttentionLight: Rethinking queue length and attention mechanism for traffic signal control](http://arxiv.org/abs/2201.00006).

If you use our method, please cite our article.
```latex
@misc{https://doi.org/10.48550/arxiv.2201.00006,

title = {AttentionLight: Rethinking queue length and attention mechanism for traffic signal control},
author = {Zhang, Liang and Wu, Qiang and Deng, Jianming},
publisher = {arXiv},
  
year = {2022},
  
copyright = {arXiv.org perpetual, non-exclusive license}
}
```

## 2. Rquirements
`python3.6`,`tensorflow=2.4`, `cityflow`, `pandas`, `numpy`

[`cityflow`](https://github.com/cityflow-project/CityFlow.git) needs a linux environment, and we run the code on Manjaro Linux.


## 3. Quick start

For the method in our article, run:
```shell
python run_max_ql.py
```
```shell
python run_ql_dqn.py
```
```shell
python run_ql_frap.py
```
```shell
python run_ql_gat.py
```

For the baseline methods,
- Fixed-Time
```shell
python run_fixedtime.py
```
- Max-Pressure
```shell
python run_maxpressure.py
```
- PressLight
```shell
python run_presslight.py
```
- MPLight
```shell
python run_mplight.py
```
- FRAP
```shell
python run_frap.py
```
- Colight
```shell
python run_colight.py
```
### 3.1 Evaluate the results
Change the folder name in `summary.py` as yours, and run: 
```shell
python summary.py
```
## 4. Code details
### 4.1、structure
- `models`: contains all the models used in our article.
- `utils`: contains all the methods to simulate and train the models.

### 4.2、Reference

The code is modified from [Efficient_XLight](https://github.com/LiangZhang1996/Efficient_XLight.git).
The `Max-Pressure` is created by ourselves, based on [MaxPressure](https://www.sciencedirect.com/science/article/pii/S0968090X13001782) .
- `PressLight`: Bsed on `LIT` model, which comes from [Colight](https://github.com/wingsweihua/colight.git).
- `Colight` : Based on [Colight](https://github.com/wingsweihua/colight.git).
- `Fixed-Time`: From [MPLight](https://github.com/Chacha-Chen/MPLight.git).
- `MPLight`: From [MPLight](https://github.com/Chacha-Chen/MPLight.git).


