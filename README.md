# SubspaceNet for Radar DoA Estimation

> 基于 SubspaceNet 的雷达辐射源 DoA 估计 - 硕士研究项目

## 项目简介

SubspaceNet 是一种深度学习辅助的子空间 DoA 估计方法，核心思想是用神经网络学习阵列接收信号的代理协方差矩阵，再与 Root-MUSIC、ESPRIT 等可解释的子空间方法结合。原论文为 *SubspaceNet: Deep Learning-Aided Subspace Methods for DoA Estimation*，发表于 IEEE Transactions on Vehicular Technology, Vol. 74, No. 3, 2025。

本项目以原版 SubspaceNet 代码为基础，面向雷达电子侦察场景做工程化整理。当前版本已经接入雷达辐射源手册，可从 PDW 参数生成基带 I/Q 脉冲信号，并复用原版 SubspaceNet 的数据生成、训练、评估流程。

应用场景是电子侦察 ESM 系统中的辐射源 DoA 估计：输入阵列接收信号，输出多个辐射源的方向角估计，并与真值计算误差。

## 核心扩展

- 雷达辐射源库：`data/data_manual.xlsx`，当前使用第一个工作表 `空中平台追踪`，包含 5 个型号、6 种工作模式，共 30 个模式。
- PDW 到 I/Q 波形合成：`src/radar_samples.py`，支持 PRI/RF 变化和 LFM 基带合成。
- 向后兼容原版 SubspaceNet：默认 `data_source="gaussian"`，只有设置 `data_source="radar"` 时才启用雷达信号源。
- 组会 demo：`demo.py` 支持预设场景、500 样本批量评估和交互式 DoA 输入。

## 环境要求

- Python 3.10+
- PyTorch >= 2.0
- 主要依赖：
  - numpy, scipy
  - torch
  - tqdm
  - matplotlib, scikit-learn
  - pandas, openpyxl

当前代码对 `matplotlib`、`scikit-learn`、`pandas/openpyxl` 做了运行兼容：缺少这些包时，训练和 Excel 表格 1 解析仍可在最小环境下运行。

### 安装

```bash
conda create -n subspacenet python=3.10
conda activate subspacenet
pip install -r pyEnv/requirements.txt
```

## 快速开始

### 1. 运行原版 SubspaceNet 基线

```bash
python -B main.py
```

当前 `main.py` 使用仓库本地已有的原版 `diff_esprit` 数据和权重做评估，不重新训练模型。

### 2. 训练雷达版本模型

```bash
python -B main_radar.py
```

默认配置：

- N=16 阵元
- M=2 源
- T=200 快拍
- SNR=0 dB
- 5000 条训练样本 + 500 条测试样本
- 15 epochs
- batch size 128
- learning rate 1e-4

训练完成后，权重保存在：

```text
data/weights/final_models/Radar_SubspaceNet_M=2_T=200_SNR_0_tau=8_NarrowBand_diff_method=esprit_non-coherent_eta=0_bias=0_sv_noise=0.pt
```

### 3. 运行组会 Demo

预设 3 组 DoA 场景：

```bash
python -B demo.py --mode preset
```

500 条测试样本批量评估：

```bash
python -B demo.py --mode batch
```

交互式输入，角度范围限制为 -60 到 +60 度：

```bash
python -B demo.py --mode interactive
```

非交互方式指定角度：

```bash
python -B demo.py --mode interactive --angles "-20 35"
```

一次性运行三档：

```bash
python -B demo.py --mode all --angles "-20 35"
```

## 项目结构

```text
.
├── data/
│   ├── data_manual.xlsx          # 雷达辐射源手册，入库
│   ├── datasets/                 # 生成的训练/测试集，不入库
│   └── weights/                  # 训练后的模型权重，不入库
├── pyEnv/
│   ├── requirements.txt
│   └── SubspaceNetEnv.yaml
├── src/
│   ├── radar_samples.py          # 雷达 PDW + I/Q 信号源
│   ├── signal_creation.py        # 原版高斯/合成信号源
│   ├── data_handler.py           # 数据生成、加载、雷达/高斯分发
│   ├── system_model.py           # 系统参数与雷达扩展参数
│   ├── models.py                 # SubspaceNet 网络结构
│   ├── methods.py                # MUSIC、Root-MUSIC、ESPRIT、MVDR
│   ├── training.py               # 训练参数与训练入口
│   ├── evaluation.py             # 评估函数
│   ├── criterions.py             # RMSPE/MSPE 损失函数
│   ├── plotting.py               # 原版绘图工具
│   └── utils.py
├── main.py                       # 原版 SubspaceNet 基线评估入口
├── main_radar.py                 # 雷达版本训练入口
├── demo.py                       # 组会演示脚本
├── demo_doa.py                   # 早期最小 DoA demo
└── README.md
```

## 实验结果

| 配置 | N | M | T | SNR | Test RMSPE |
|---|---:|---:|---:|---:|---:|
| 原版 SubspaceNet + ESPRIT | 8 | 5 | 100 | 10 dB | 1.624 deg |
| 雷达 SubspaceNet + ESPRIT | 16 | 2 | 200 | 0 dB | 1.403 deg |
| 雷达 SubspaceNet + Root-MUSIC 增强 | 16 | 2 | 200 | 0 dB | 1.458 deg |

雷达基线训练输出：

```text
Minimal Validation loss: 0.024531 rad
SubspaceNet Test loss = 0.02448356477335886 rad = 1.403 deg
```

Demo 单样本推理时间在 CPU 上约 1.5-6 ms，满足组会演示流畅度需求。

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

- [ ] SNR sweep 实验。
- [ ] 相干源雷达场景验证。
- [ ] 多任务联合估计：DoA + PDW 参数。
- [ ] 脉冲检测器集成：仅作为可部署版本扩展，不进入当前最小 demo。

## License

本项目基于 SubspaceNet 扩展，原项目地址：https://github.com/ShlezingerLab/SubspaceNet

本仓库遵循原项目的许可协议；新增研究代码的许可证将在项目稳定后补充说明。
