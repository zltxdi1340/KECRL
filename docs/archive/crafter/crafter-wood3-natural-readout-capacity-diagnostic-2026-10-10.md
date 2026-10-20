# Crafter wood>=3 自然状态读出容量诊断（2026-10-10）

## 目的

本轮继续诊断冻结空间 CNN 后的 actor 读出能力。前一轮已经在自然 RGB 状态上完成原始线性 head 和标准化线性 head 的诊断；本轮加入一个固定宽度 64 的两层 MLP head，用来检查线性读出是否成为局部动作选择的瓶颈。

这仍是 SPI 可行性诊断，不是正式 Policy 训练，也不创建 Module、更新 Knowledge 或注册 SPT。

## 固定边界

- CNN checkpoint、critic、PPO optimizer 和 RGB 输入保持不变。
- 训练集和验证集复用上一轮已验证的 376/153 个自然状态；旧 heldout 166 个状态只用于历史审计，不能进入本轮拟合或选择。
- 新增 fresh 环境 seed 为 `93000000..93000023`，动作 seed 为 `94000000..94000023`；新 heldout 状态与旧自然状态做 exact RGB 排除。
- 监督目标是几何 teacher 给出的合法局部动作集合，损失为按类别平衡的 valid-action-set negative log likelihood。
- 标准化均值和方差只由训练集计算；MLP 的 epoch 和 learning rate 只由验证集宏准确率、再由集合 NLL 选择。
- 所有 head 在 fresh heldout 采集前锁定；采集和 8-step greedy/sample 窗口都不更新策略、encoder、critic 或 PPO optimizer。
- 每个 seed 使用 baseline、复用的标准化线性 head、标准化 MLP64 三个 arm；主运行包含跨进程 seed 0 重复。

## MLP 解释边界

MLP 使用固定初始化，线性 head 复用上一轮已经拟合并锁定的 artifact。因此该实验是容量诊断，不能宣称完全由架构单变量造成的因果差异。即使 MLP 在局部验证集上更好，也不能推出它能解决生存、探索、导航、挖矿或完整 wood>=3 任务；如果两者执行成功率接近，优先继续查执行链、状态覆盖和训练策略。

## 预验证 smoke

seed 0 smoke 已完成：fresh heldout 224 个状态，三种 head 共执行 2016 个 8-step 窗口；源策略和 optimizer 冻结检查通过。验证集宏准确率为：标准化线性 `0.436`、标准化 MLP64 `0.365`；fresh heldout 宏准确率为：标准化线性 `0.363`、标准化 MLP64 `0.366`。MLP 的 heldout 集合 NLL 为 `1.865`，线性为 `2.162`。局部 8-step greedy 成功没有出现同步提升，所以 smoke 不能支持“读出容量已经是主瓶颈”的判断。

smoke 使用 `--repeat-seed0`，没有生成独立进程重复目录；它只验证执行链，不是正式结果。

## 三 seed 诊断结果

正式诊断于 2026-10-10 完成。三个来源策略均为固定 100k-step CNN checkpoint；每个 head 只用原有 376 个训练状态和 153 个验证状态拟合/选型。24 个新环境 episode 产生 224 个去重 heldout 状态（ready 56、turn 66、approach 102），三个 seed 共评估 6,048 个相同的 8-step 窗口。策略、encoder、critic 和 PPO optimizer 均保持冻结。

Heldout 的类别宏准确率（合法动作集合读出，不是环境成功率）：

| policy seed | 原始 actor | 标准化线性 head | 标准化 MLP64 |
| ---: | ---: | ---: | ---: |
| 0 | 0.285 | 0.363 | 0.366 |
| 1 | 0.236 | 0.436 | 0.379 |
| 2 | 0.232 | 0.399 | 0.389 |
| 均值 | 0.251 | 0.399 | 0.378 |

标准化线性 head 的 heldout 宏准确率三个 seed 都高于原始 actor。MLP64 只在 seed 0 略高于线性 head，seed 1 和 2 都更低；平均低 0.021。MLP 的 heldout 集合 NLL 均值为 2.627，线性 head 为 2.173，且 seed 2 的 MLP NLL 达 3.611（线性为 1.801）。因此本协议没有显示稳定的非线性容量收益；在这份数据和拟合设置下，线性适配比扩大到 MLP 更可靠。

固定自然状态上 8-step 内成功采到指定目标树木（同一场景的短窗重复，按三个 seed 合并）：

| 动作模式 | 原始 actor | 标准化线性 head | 标准化 MLP64 |
| --- | ---: | ---: | ---: |
| greedy | 90/672 (13.4%) | 156/672 (23.2%) | 119/672 (17.7%) |
| sample | 306/1344 (22.8%) | 434/1344 (32.3%) | 319/1344 (23.7%) |

greedy 的线性 head 在 ready、turn、approach 三类状态分别为 56.0%、13.1%、11.8%；原始 actor 分别为 48.8%、0.5%、2.3%。三类都有局部提升，turn/approach 的提升最大。按每个 seed 单独看，线性 head 在 seed 0 的 greedy 窗口略低于原始 actor（21.0% 对 21.4%），但 seed 1、2 更高；sample 窗口三个 seed 全部更高。合并窗口是描述性统计：同一批目标状态和多个 seed/重复并非独立样本，不能据此报告独立 episode 置信区间。

## 独立复核

运行器的跨进程 seed 0 重复中，数据数组、记录、采集轨迹、CNN 特征、预测、窗口轨迹和 MLP 权重摘要 7 项全部一致。独立分析器再次从原始 RGB 和 checkpoint 重算三个 seed 的 CNN 特征、head logits、MLP 训练/验证指标及验证集选型，并复算全部 6,048 个窗口；全部通过。主运行和重复运行的 48 项输入哈希均未变化。

独立分析文件位于 `results/crafter_wood3_natural_readout_capacity_cuda_20261010_v1/analysis_verified/verification.json`，逐窗 CSV 位于同目录 `window_metrics.csv`。

## 判断与下一步

这个实验说明冻结 CNN 表征上存在可由线性 actor head 利用的局部木材采集信号；因此当前证据不支持把“actor head 容量不足”作为主要阻塞点。MLP64 没有稳定收益。线性 head 的局部收益仍不足以证明它能跨场景生存、导航或达成 `wood>=3`，本实验也没有把 teacher head 注册成 Module。

下一步是对三个冻结策略 seed 做完整 `wood>=3` episode 配对：原始 actor 与标准化线性 head 使用同一组全新环境 seed 和 action seed，记录 success、death/截断、最终 wood、首次 wood 1/2/3 步数及伤害来源。该测试可区分局部树木交互改进是否传递到完整任务链；它仍是冻结策略评估，不启动 PPO 正式训练，也不构成 Module 资格。

## 正式运行

```bash
PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=0 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/zl202621/.conda/envs/kecrl/bin/python -u -m experiments.run_crafter_wood3_natural_readout_capacity \
  --config configs/crafter_wood3_natural_readout_capacity_cuda_v1.yaml \
  --output results/crafter_wood3_natural_readout_capacity_cuda_20261010_v1
```

独立复核：

```bash
PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=0 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/zl202621/.conda/envs/kecrl/bin/python -u -m experiments.analyze_crafter_wood3_natural_readout_capacity \
  --input results/crafter_wood3_natural_readout_capacity_cuda_20261010_v1 \
  --output results/crafter_wood3_natural_readout_capacity_cuda_20261010_v1/analysis_verified
```

正式结果仍需按 seed、category、greedy/sample 和窗口依赖关系解释。无论 MLP 结果如何，本轮都不能把 wood 局部 teacher readout 当作完整 Module 资格；后续必须继续诊断 survival、导航执行以及 stone/coal/pickaxe 的任务链覆盖。
