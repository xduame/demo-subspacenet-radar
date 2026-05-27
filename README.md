# SubspaceNet for Radar DoA Estimation

> 基于 SubspaceNet 的雷达辐射源 DoA 估计 - 硕士研究项目

## 项目简介

SubspaceNet 是一种深度学习辅助的子空间 DoA 估计方法，核心思想是用神经网络学习阵列接收信号的代理协方差矩阵，再与 Root-MUSIC、ESPRIT 等可解释的子空间方法结合。原论文为 *SubspaceNet: Deep Learning-Aided Subspace Methods for DoA Estimation*，发表于 IEEE Transactions on Vehicular Technology, Vol. 74, No. 3, 2025。

本项目以原版 SubspaceNet 代码为基础，面向雷达电子侦察场景做工程化整理。当前仓库保留原版高斯/合成信号训练与评估管线，并提供一个最小可运行的 DoA 演示脚本：加载已训练权重，读取一条阵列接收信号，输出 DoA 估计、真值和误差。

后续研究目标是将信号源从理想高斯源扩展到雷达脉冲辐射源，使模型能够服务于 ESM 系统中的辐射源 DoA 估计任务。扩展会保持与原版 SubspaceNet 训练接口兼容，避免破坏已有模型、损失函数和评估代码。

## 核心扩展

当前仓库已经完成：

- 原版 SubspaceNet 代码基线整理。
- 最小 DoA demo：`demo_doa.py`。
- 使用现有测试样本和权重进行单样本 DoA 估计，输出真值、估计值、逐源误差、RMSE 和 MAE。

规划中的雷达扩展：

- 雷达辐射源库：5 型号 x 6 工作模式，共 30 个 PDW 生成器。
- PDW 到 I/Q 波形合成：支持 LFM 调制。
- 向后兼容原版 SubspaceNet 训练管线：雷达信号源作为数据生成分支接入，不改模型主体结构。

## 环境要求

- Python 3.10+
- PyTorch >= 2.0
- 主要依赖：
  - numpy, scipy
  - torch
  - matplotlib
  - scikit-learn
  - tqdm

雷达辐射源手册接入后还会使用：

- pandas
- openpyxl

### 安装

```bash
conda create -n subspacenet python=3.10
conda activate subspacenet
pip install -r pyEnv/requirements.txt
```

## 快速开始

### 1. 运行当前 DoA Demo

当前可运行入口是 `demo_doa.py`。它依赖本地已有的测试集和模型权重：

- `data/datasets/diff_esprit/test/Generic_DataSet_NarrowBand_coherent_100_M=5_N=8_T=100_SNR=10_eta=0_sv_noise_var0_.h5`
- `data/weights/final_models/SubspaceNet_M=5_T=100_SNR_10_tau=8_NarrowBand_diff_method=esprit_coherent_eta=0_sv_noise=0`

这些数据和权重属于大文件，按 `.gitignore` 不提交到 GitHub。

```bash
python -B demo_doa.py
```

指定样本编号：

```bash
python -B demo_doa.py --sample-index 99
```

默认示例输出包括：

```text
Ground truth: [ -53.970,  -21.490,   22.840,   44.430,   85.840]
Prediction : [ -52.710,  -21.924,   23.021,   44.911,   87.254]
RMSE       : 0.899 deg
MAE        : 0.754 deg
```

### 2. 原版训练入口

`main.py` 是原版 SubspaceNet 的训练/评估入口。运行前需要确认脚本中的数据路径、系统参数和 `commands` 配置与本地数据一致。

```bash
python main.py
```

## 项目结构

```text
.
├── data/                         # 本地数据与权重目录，不提交到 GitHub
│   ├── datasets/                 # 生成的训练/测试集
│   ├── simulations/              # 仿真结果
│   └── weights/                  # 训练后的模型权重
├── pyEnv/
│   ├── requirements.txt          # pip 依赖
│   └── SubspaceNetEnv.yaml       # conda 环境记录
├── src/
│   ├── signal_creation.py        # 原版高斯/合成信号源
│   ├── data_handler.py           # 数据集生成、加载、自相关张量构造
│   ├── system_model.py           # 阵列和系统参数
│   ├── models.py                 # SubspaceNet 网络结构
│   ├── methods.py                # MUSIC、Root-MUSIC、ESPRIT、MVDR
│   ├── training.py               # 训练循环
│   ├── evaluation.py             # 模型与经典方法评估
│   ├── criterions.py             # RMSPE/MSPE 损失函数
│   ├── plotting.py               # 谱图与结果绘图
│   └── utils.py                  # 工具函数
├── main.py                       # 原版训练/评估入口
├── demo_doa.py                   # 当前组会演示脚本
└── README.md                     # 项目入口文档
```

## 实验结果

### 当前仓库 Smoke Test

| 配置 | N | M | T | SNR | 样本 | Test RMSPE / RMSE |
|---|---:|---:|---:|---:|---:|---:|
| SubspaceNet + ESPRIT | 8 | 5 | 100 | 10 dB | 0 | 0.899 deg |
| SubspaceNet + ESPRIT | 8 | 5 | 100 | 10 dB | 99 | 1.145 deg |

### 雷达版本实验记录

| 配置 | N | M | T | SNR | Test RMSPE |
|---|---:|---:|---:|---:|---:|
| 雷达基线 | 16 | 2 | 200 | 0 dB | 1.65 deg |
| 雷达扩展实验 A | 16 | 2 | 200 | 0 dB | 1.87 deg |
| 雷达扩展实验 B | 16 | 2 | 200 | 0 dB | 2.15 deg |

## 引用

如果使用本项目，请引用原 SubspaceNet 论文：

```bibtex
@article{shmuel2025subspacenet,
  title={SubspaceNet: Deep Learning-Aided Subspace Methods for DoA Estimation},
  author={Shmuel, Dor Haim and Merkofer, Julian P. and Revach, Guy and van Sloun, Ruud J. G. and Shlezinger, Nir},
  journal={IEEE Transactions on Vehicular Technology},
  volume={74},
  number={3},
  pages={4962--4976},
  year={2025},
  doi={10.1109/TVT.2024.3496119}
}
```

## 后续工作

- [ ] 新增 `src/radar_samples.py`，实现雷达 PDW 生成与 I/Q 波形合成。
- [ ] 新增 `main_radar.py`，提供雷达版本训练入口。
- [ ] 新增 `demo.py`，整合预设场景、批量评估和交互式输入。
- [ ] SNR sweep 实验。
- [ ] 相干源场景验证。
- [ ] 多任务联合估计：DoA + PDW 参数。
- [ ] 脉冲检测器集成：仅作为可部署版本扩展，不进入当前最小 demo。

## License

本项目基于 SubspaceNet 扩展，原项目地址：https://github.com/ShlezingerLab/SubspaceNet

本仓库遵循原项目的许可协议；新增研究代码的许可证将在项目稳定后补充说明。
