# Spatial-Temporal Large Language Model for Traffic Prediction
This repository contains the code of ST-LLM *"Spatial-Temporal Large Language Model for Traffic Prediction"* [paper](https://github.com/ChenxiLiu-HNU/ST-LLM/blob/main/ST-LLM.pdf)

## Abstract
> *Traffic prediction, an essential component for intelligent transportation systems, endeavours to use historical data to foresee future traffic features at specific locations. Although existing traffic prediction models often emphasize developing complex neural network structures, their accuracy has not improved. Recently, large language models have shown outstanding capabilities in time series analysis. Differing from existing models, LLMs progress mainly through parameter expansion and extensive pretraining while maintaining their fundamental structures. Motivated by these developments, we propose a Spatial-Temporal Large Language Model (ST-LLM) for traffic prediction. In the ST-LLM, we define timesteps at each location as tokens and design a spatial-temporal embedding to learn the spatial location and global temporal patterns of these tokens. Additionally, we integrate these embeddings by a fusion convolution to each token for a unified spatial-temporal representation. Furthermore, we innovate a partially frozen attention strategy to adapt the LLM to capture global spatial-temporal dependencies for traffic prediction. Comprehensive experiments on real traffic datasets offer evidence that ST-LLM is a powerful spatial-temporal learner that outperforms state-of-the-art models. Notably, the ST-LLM also exhibits robust performance in both few-shot and zero-shot prediction scenarios.*

<img width="1098" alt="image" src="https://github.com/ChenxiLiu-HNU/ST-LLM/assets/46647878/15bf40a4-333f-42ed-a241-32432a5484ce">

## Dependencies

* Python 3.8.19
* PyTorch 2.4.1
* cuda 11.7
* torchvision 0.19.1

```bash
> conda env create -f env_ubuntu.yaml
```

## Datasets
We provide preprocessed datasets, which you can access [here](https://drive.google.com/drive/folders/1iif59LObrPu-QrpL8Y6lWeajbn_gRf7v?usp=drive_link).   
If you need the original datasets, please refer to the [ESG](https://github.com/LiuZH-19/ESG).

## Training

```bash
CUDA_VISIBLE_DEVICES=0
nohup python train.py --data taxi_pick > your_log_name.log &
```

## BibTex
> If you find our work useful in your research. Please consider giving a star ⭐ and citation 📚:
```bibtex
@inproceedings{liu2024spatial,
  title={Spatial-Temporal Large Language Model for Traffic Prediction},
  author={Liu, Chenxi and Yang, Sun and Xu, Qianxiong and Li, Zhishuai and Long, Cheng and Li, Ziyue and Zhao, Rui},
  booktitle={MDM},
  year={2024}
}
```

## Further Reading
[**ST-LLM+: Graph Enhanced Spatio-Temporal Large Language Models for Traffic Prediction**](https://www.computer.org/csdl/journal/tk/5555/01/11005661/26K27tC6ki4), in *TKDE* 2025.
[\[GitHub Repo\]](https://github.com/kethmih/ST-LLM-Plus)

**Authors**: Chenxi Liu, Kethmi Hirushini Hettige, Qianxiong Xu, Cheng Long, Shili Xiang, Gao Cong, Ziyue Li, Rui Zhao

```bibtex
@article{liu2025stllm_plus,
  title={{ST-LLM+}: Graph Enhanced Spatial-Temporal Large Language Model for Traffic Prediction},
  author={Chenxi Liu and  Hettige Kethmi Hirushini and Qianxiong Xu and Cheng Long and Ziyue Li and Shili Xiang and Rui Zhao and Gao Cong},
  journal    = {{IEEE} Transactions Knowledge Data Engineering},
  pages      = {1-14},
  year={2025}
}
```

[**TimeCMA: Towards LLM-Empowered Multivariate Time Series Forecasting via Cross-Modality Alignment**](https://arxiv.org/abs/2406.01638), in *AAAI* 2025.
[\[GitHub Repo\]](https://github.com/ChenxiLiu-HNU/TimeCMA)

**Authors**: Chenxi Liu, Qianxiong Xu, Hao Miao, Sun Yang, Lingzheng Zhang, Cheng Long, Ziyue Li, Rui Zhao

```bibtex
@inproceedings{liu2024timecma,
  title={{TimeCMA}: Towards LLM-Empowered Multivariate Time Series Forecasting via Cross-Modality Alignment},
  author={Liu, Chenxi and Xu, Qianxiong and Miao, Hao and Yang, Sun and Zhang, Lingzheng and Long, Cheng and Li, Ziyue and Zhao, Rui},
  booktitle={AAAI},
  year={2025}
}
```

[**Efficient Multivariate Time Series Forecasting via Calibrated Language Models with Privileged Knowledge Distillation**](https://arxiv.org/abs/2505.02138), in *ICDE* 2025.
[\[GitHub Repo\]](https://github.com/ChenxiLiu-HNU/TimeKD)

**Authors**: Chenxi Liu, Hao Miao, Qianxiong Xu, Shaowen Zhou, Cheng Long, Yan Zhao, Ziyue Li, Rui Zhao

```bibtex
@inproceedings{liu2025timekd,
  title={Efficient Multivariate Time Series Forecasting via Calibrated Language Models with Privileged Knowledge Distillation},
  author={Chenxi Liu and Hao Miao and Qianxiong Xu and Shaowen Zhou and Cheng Long and Yan Zhao and Ziyue Li and Rui Zhao},
  booktitle    = {ICDE},
  year={2025}
}
```

[**Towards Cross-Modality Modeling for Time Series Analytics: A Survey in the LLM Era**](https://arxiv.org/abs/2505.02583), in *IJCAI* 2025.
[\[GitHub Repo\]](https://github.com/ChenxiLiu-HNU/CM2TS)

**Authors**: Chenxi Liu, Shaowen Zhou, Qianxiong Xu, Hao Miao, Cheng Long, Ziyue Li, Rui Zhao

```bibtex
@inproceedings{liu2025cm2ts,
  title={Towards Cross-Modality Modeling for Time Series Analytics: A Survey in the LLM Era},
  author={Chenxi Liu and Shaowen Zhou and Qianxiong Xu and Hao Miao and Cheng Long and Ziyue Li and Rui Zhao},
  booktitle    = {IJCAI},
  year={2025}
}
```

[**LLMs Meet Cross-Modal Time Series Analytics: Overview and Directions**](https://dl.acm.org/doi/10.1145/3748777.3748810), in *SSTD* 2025.
[\[Website\]](https://llm-cmts.github.io/)

**Authors**: Chenxi Liu, Hao Miao, Cheng Long, Yan Zhao, Ziyue Li, Panos Kalnis

```bibtex
@inproceedings{sstd2025cmts,
  title={LLMs Meet Cross-Modal Time Series Analytics: Overview and Directions},
  author={Chenxi Liu and Hao Miao and Cheng Long and Yan Zhao and Ziyue Li and Panos Kalnis},
  booktitle    = {SSTD},
  year={2025}
}
```

## Acknowledgement
Our implementation adapts [OFA](https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All) as the code base and has extensively modified it for our purposes. We are grateful to the authors for providing their implementations and related resources.

## Contact Us
For inquiries or further assistance, contact us at [chenxi.liu@ntu.edu.sg](mailto:chenxi.liu@ntu.edu.sg).
