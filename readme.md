## 1. Introduction


Official code for article [Leveraging Queue Length and Attention Mechanisms for Enhanced Traffic Signal Control Optimization](https://doi.org/10.1007/978-3-031-43430-3_9).

This article has been received by ECML PKDD 2023.

If you use our method, please cite our article.
```latex
@inproceedings{attentionlight,
  title={Leveraging Queue Length and Attention Mechanisms for Enhanced Traffic Signal Control Optimization},
  author={Zhang, Liang and Xie, Shubin and Deng, Jianming},
  booktitle={Joint European Conference on Machine Learning and Knowledge Discovery in Databases},
  pages={141--156},
  year={2023},
  organization={Springer}
}

```

## 2. Requirements
`python3.6`,`tensorflow=2.4`, `cityflow`, `pandas`, `numpy`

[`cityflow`](https://github.com/cityflow-project/CityFlow.git) needs a Linux environment, and we run the code on Manjaro Linux.


## 3. Quick start

For the method in our article, run the following:
```shell
python run_attention_light.py
```
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
Change the folder name in `summary.py` to yours, and run: 
```shell
python summary.py
```
## 4. Code details
### 4.1、structure
- `models`: contains all the models used in our article.
- `utils`: contains all the methods to simulate and train the models.

### 4.2、Reference

The code is modified from [Efficient_XLight](https://github.com/LiangZhang1996/Efficient_XLight.git).
The `Max-Pressure` is created by ourselves, based on [MaxPressure](https://www.sciencedirect.com/science/article/pii/S0968090X13001782).
- `PressLight`: Based on `LIT` model, which comes from [Colight](https://github.com/wingsweihua/colight.git).
- `Colight` : Based on [Colight](https://github.com/wingsweihua/colight.git).
- `Fixed-Time`: From [MPLight](https://github.com/Chacha-Chen/MPLight.git).
- `MPLight`: From [MPLight](https://github.com/Chacha-Chen/MPLight.git).

## License
This project is licensed under the GNU General Public License version 3 (GPLv3) - see the LICENSE file for details.
