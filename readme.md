# 方向三：大模型知识编辑实验

本目录是《大模型安全与知识增强》结课作业方向三“知识编辑”的完整提交物。项目包含 Task 1~4 的可执行脚本、10 条自定义事实更新数据、500 条批量编辑数据生成逻辑、评估结果和 PDF 实验报告。

## 文件结构

```text
baseline.py              # Task 1：编辑前基线测试
edit_rome.py             # Task 2：ROME 单条事实编辑
edit_memit.py            # Task 3：MEMIT 批量事实编辑
evaluate.py              # Task 4：ES / PS / NS 综合评估
data/custom_facts.json   # 10 条自定义事实更新数据
data/counterfact_500_sample.json  # 运行 edit_memit.py 后生成
artifacts/results/       # JSON 实验结果
artifacts/logs/          # 终端输出文本
report/knowledge_editing_report.pdf
```

## 环境安装

```bash
pip install -r requirements.txt
```


## 1. 实验背景与目的

随着大语言模型（LLM）的广泛应用，模型知识过期或包含错误事实（幻觉）的问题日益凸显。重新训练或全量微调成本极高，而知识编辑（Knowledge Editing）技术允许我们在不显著改变模型其他行为的前提下，精准、快速地修改模型内部的特定知识。

**实验目的：**

1. 深入理解大语言模型中事实知识的存储机制。
2. 掌握并实践主流的知识编辑算法（如 ROME, MEMIT）。
3. 理解知识编辑的三大核心评估指标：编辑成功率（Efficacy）、泛化性（Generalization）与局部性/特异性（Locality）。

------

## 2. 实验环境与工具准备

- **基础模型：**`Qwen2.5-0.5B`等模型即可。
- **核心框架：** 推荐使用开源框架 [EasyEdit](https://github.com/zjunlp/EasyEdit)（由浙江大学开源，集成了主流编辑算法）。



本实验使用 conda 环境、EasyEdit、Qwen2.5-0.5B-Instruct 和 RTX 3060 12GB GPU 完成。

为适配新版 PyTorch，已对 EasyEdit 的 `easyeditor/util/nethook.py` 做 forward hook 参数顺序补丁。
## 一键运行

## Task1、2
```bash
python baseline.py
python edit_rome.py
```

## Task3
```bash 
python prepare_memit_covariance.py      #先运行这个，预计算 mom2 协方差统计
python edit_memit.py                    #正式执行
```


## Task4
```bash
python evaluate.py
```

运行完成后，核心输出位于：

- `artifacts/results/evaluation_summary.json`
- `artifacts/results/baseline_results.json`
- `artifacts/results/rome_results.json`
- `artifacts/results/memit_results.json`
- `report/knowledge_editing_report.pdf`

## 评估指标
- ES（Efficacy）：直接编辑 prompt 是否输出目标新答案。
- PS（Generalization）：同义改写 prompt 是否输出目标新答案。
- NS（Locality）：无关事实查询是否保持原答案。


## 当前实验结果
| 算法 | 样本数 |    ES |    PS |      NS |
|---|---:|------:|------:|--------:|
| Baseline | 10 |  0.0% |  0.0% |   20.0% |
| ROME | 10 | 98.3% | 74.8% |   79.2% |
| MEMIT | 500 | 84.0% | 62.4% |  48.7% |


