# Crafter wood>=3 自然状态覆盖诊断（2026-10-11）

本轮诊断了一个具体假设：上一轮交叉读出显示，natural standardized linear head 能在
baseline 实际访问的状态上改善 turn/approach 动作，但它改变动作后会进入自己难以读出的
后续状态。这里固定 100k CNN-only checkpoint、Crafter wrapper、`wood>=3` 任务和
监督损失，只改变自然状态覆盖来源，检查“覆盖不足”是否是主要瓶颈。

本轮仍是可行性诊断，不是正式训练：没有 PPO 更新、encoder 更新、Knowledge/SPT 更新、
Module 注册，也没有 stone、coal 或 pickaxe 资格结论。

## 第一轮覆盖 pilot

第一轮把已完成自然任务链中的 `baseline` 和 `standardized_linear` 两种轨迹重放，按完整
episode 隔离 train/validation/heldout，并每个类别最多选 6 个、间隔至少 8 步的自然状态。
三 seed 的覆盖集分别为 571、590、600 个状态。fresh 任务链使用独立的
`103000000/105000000` 环境/action seed。

| 冻结策略 | wood>=3 成功 | 死亡 | external truncation |
| --- | ---: | ---: | ---: |
| 原始 baseline actor | 40/90 = 44.4% | 44 | 6 |
| 旧 natural standardized head | 27/90 = 30.0% | 60 | 3 |
| 非匹配扩大覆盖 head | 37/90 = 41.1% | 50 | 3 |

这轮结果看起来比旧 head 好，但训练样本总量和各 source path 的类别比例没有与对照严格
匹配，因此只能作为 pilot，不能归因于 coverage 本身。产物在
`results/crafter_wood3_natural_coverage_cuda_20261011_v1/`，独立复核在其
`analysis_verified_v2/`。

## 等预算 matched 对照

为补齐可比性，第二轮对每个 seed、每个 `ready/turn/approach` 类别固定相同训练状态数：

- `matched_baseline_only`：训练状态全部来自 baseline 轨迹；
- `matched_mixed_coverage`：一半来自 baseline，一半来自旧 natural head 轨迹；
- 两臂使用相同 validation episode、相同 coverage-heldout episode，heldout 不参与拟合或选择；
- 两臂使用相同 fresh 环境/action seed，每个 seed 每臂 30 集，共 90 集；
- seed 0 独立进程重跑，训练行 digest、head digest 和 fresh trace digest 全部一致。

完整结果：

| seed | baseline-only | mixed coverage | baseline-only 死亡 | mixed coverage 死亡 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 5/30 = 16.7% | 4/30 = 13.3% | 24 | 24 |
| 1 | 14/30 = 46.7% | 9/30 = 30.0% | 15 | 21 |
| 2 | 10/30 = 33.3% | 13/30 = 43.3% | 19 | 16 |
| **合计** | **29/90 = 32.2%** | **26/90 = 28.9%** | **58** | **61** |

配对结果为 baseline-only 独有成功 20 集、mixed 独有成功 17 集、两者都成功 9 集、
两者都失败 44 集。seed 方向不一致，mixed coverage 没有稳定提升，反而在 pooled 结果
下降 3 个成功 episode。独立分析器还验证了动作概率归一化、episode seed manifest、
初始 RGB 配对、observer 不改变轨迹以及 stage funnel。

阶段漏斗的 pooled 计数如下：

| 策略 | wood=0 进入/完成 | wood=1 进入/完成 | wood=2 进入/完成 |
| --- | ---: | ---: | ---: |
| matched baseline-only | 90/59 | 59/37 | 37/29 |
| matched mixed coverage | 90/61 | 61/42 | 42/26 |

mixed coverage 在前两阶段并没有形成稳定优势，进入 wood=2 后完成率从 `29/37`
降为 `26/42`。主要失败类别仍是死亡、没有观察到可用相邻树、没有进入可采集朝向，
以及已经 ready 但没有完成 collection；这说明本次等预算的自然 RGB 覆盖混合方式
没有稳定改善完整任务，尚不能判定各失败原因的因果贡献。

## 结论与 SPI 关系

本次覆盖混合方式没有稳定收益，不能将该线性 head 注册为 wood Module。这不能排除
更充分覆盖、不同采样或更大数据预算的作用。当前还没有满足候选 `0.8` 资格门槛的真实
executor，也不能把结果推广到 stone、coal 或 pickaxe。

下一项先保持 encoder、wrapper 和 wood 任务不变，复用既有 trace 对动作后的存活、
局部采集机会、wood gain 和短期阶段进展做只读审计，再决定最小可区分的干预。
2026-10-12 已完成的[动作后果报告](crafter-wood3-action-consequence-diagnostic-2026-10-12.md)
给出具体结果；consequence/feasibility 目标尚未训练或验证。

## 产物

- [第一轮配置](../configs/crafter_wood3_natural_coverage_cuda_v1.yaml)
- [第一轮 runner](../experiments/run_crafter_wood3_natural_coverage.py)
- [第一轮独立复核](../experiments/analyze_crafter_wood3_natural_coverage.py)
- [matched 配置](../configs/crafter_wood3_natural_coverage_matched_cuda_v1.yaml)
- [matched runner](../experiments/run_crafter_wood3_natural_coverage_matched.py)
- [matched 独立复核](../experiments/analyze_crafter_wood3_natural_coverage_matched.py)
- [matched 主摘要](../results/crafter_wood3_natural_coverage_matched_cuda_20261011_v1/summary.json)
- [matched 独立结果](../results/crafter_wood3_natural_coverage_matched_cuda_20261011_v1/analysis_verified/verification.json)
- [matched 阶段漏斗](../results/crafter_wood3_natural_coverage_matched_cuda_20261011_v1/analysis_verified/stage_funnel.csv)
