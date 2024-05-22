# Spatial-Temporal Large Language Model for Traffic Prediction

## Dependencies

* Python 3.11
* PyTorch 2.1.2
* cuda 12.1
* torchvision 0.8.0

```bash
> conda env create -f env_ubuntu.yaml
```

## Dataset


## Training

```bash
CUDA_VISIBLE_DEVICES=0
nohup python train.py --data taxi_pick --device cuda:0  > your_log_name.log &
```

## Citation
If you find our work useful in your research, please cite:

```bash
@inproceedings{liu2024spatial,
  title={Spatial-temporal large language model for traffic prediction},
  author={Liu, Chenxi and Yang, Sun and Xu, Qianxiong and Li, Zhishuai and Long, Cheng and Li, Ziyue and Zhao, Rui},
  booktitle={MDM},
  year={2024}
}
```
