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

本实验使用 conda 环境、EasyEdit、Qwen2.5-0.5B-Instruct 和 RTX 3060 12GB GPU 完成。

为适配新版 PyTorch，已对 EasyEdit 的 `easyeditor/util/nethook.py` 做 forward hook 参数顺序补丁。
## 一键运行

```bash
E:\app\anaconda3\envs\GCG\python.exe baseline.py
E:\app\anaconda3\envs\GCG\python.exe edit_rome.py
E:\app\anaconda3\envs\GCG\python.exe edit_memit.py
E:\app\anaconda3\envs\GCG\python.exe evaluate.py
E:\app\anaconda3\envs\GCG\python.exe make_report.py
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
| MEMIT | 500 | 73.6% | 75.0% |  100.0% |


