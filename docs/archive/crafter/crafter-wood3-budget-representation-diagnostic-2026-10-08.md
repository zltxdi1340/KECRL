# Crafter wood>=3 预算、表示与生存归因记录（2026-10-08）

本轮仍是正式实验前的非正式诊断，所有运行均为 `formal_result=false`。没有注册
Module，没有更新 Knowledge Bank 或 SPT，也没有启动正式训练。本轮结论只适用于
`wood >= 3`。

## 固定边界

- 环境采用 `balance-object-insertion-order-v1` 稳定排序 wrapper，安装的 Crafter 包
  未修改；`PYTHONHASHSEED=0`。
- PPO 采用 episode 更新、256 步外部截断、外部截断 value bootstrap，并按实际交互
  步在 25k 和 100k 精确保存 checkpoint。
- 表示对照共用两层空间 CNN、128 维 embedding、128 维 actor/critic 输入、动作
  allowlist、奖励、训练和评估 seed；唯一结构差异是 CNN 后是否有 128 维 GRU。
- 最终 v3 开启 PyTorch deterministic algorithms，固定 cuDNN，关闭 TF32，并使用
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`。固定 64x64 输入下，将无确定性 CUDA backward
  的 adaptive average pooling 换为等价的 `AvgPool2d(2, 2)`。

训练确定性审计使用两个独立 Python 进程分别重复 25k CUDA 训练。审计比较完整训练、
开发和资格轨迹的 canonical digest，以及 checkpoint 中 policy、optimizer、Python、
NumPy、Torch 和 CUDA RNG 状态。CNN-only 与 CNN+GRU 两种表示均纳入审计。
两种表示各自的两次完整轨迹 digest 相同，所有 checkpoint 比较字段也全部相同，
`cross_process_training_reproducible=true`。

## 交互预算曲线

结果：`results/crafter_wood3_budget_curve_cuda_pilot_20261008_v1/aggregate.json`。

稳定 wrapper 下的 avgpool 基线使用 3 个 seed。训练从 25k 到 100k 后，区间训练成功率
从 `0.314` 上升到 `0.399`，但 100k 最终独立资格率只有
`[0.50, 0.70, 0.55]`，均值 `0.583`，仍为 `0/3` 达到候选门槛 0.8。增加预算有学习
信号，但没有形成合格 Module。

## 表示对照

最终结果：
`results/crafter_wood3_spatial_representation_deterministic_v3_cuda_pilot_20261008_v1/aggregate.json`。
训练确定性审计：
`results/crafter_spatial_training_determinism_audit_cuda_20261008_v2/audit.json`。

最初的表示运行存在两个归因缺口：GRU hidden 为 64 而 CNN-only head 输入为 128；相同
CNN-only 配置的重复 CUDA 运行也在约第 97--197 个 episode 后发生轨迹分歧。这些运行
仅作为探索记录，不用于最终表示结论。

最终 deterministic v3 的 3 seed 曲线如下：

| 表示 | 检查点 | 区间训练成功率 | 累计训练成功率 | 开发成功率 | explained variance | value loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CNN-only | 25k | 0.300 | 0.300 | 0.400 | 0.354 | 0.146 |
| CNN-only | 100k | 0.408 | 0.383 | 0.400 | 0.581 | 0.144 |
| CNN+GRU | 25k | 0.229 | 0.229 | 0.350 | 0.360 | 0.165 |
| CNN+GRU | 100k | 0.406 | 0.366 | 0.517 | 0.575 | 0.140 |

独立资格结果：

| 表示 | seed 0 | seed 1 | seed 2 | 均值 | 达到 0.8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CNN-only | 0.50 | 0.45 | 0.65 | 0.533 | 0/3 |
| CNN+GRU | 0.55 | 0.30 | 0.65 | 0.500 | 0/3 |
| GRU - CNN | +0.05 | -0.15 | 0.00 | -0.033 | - |

GRU 在 100k 后的区间训练成功率与 CNN-only 基本相同，资格差值方向随 seed 改变，
没有形成稳定记忆收益。当前应保留更简单的 CNN-only 作为后续基础策略基线。

## 失败归因

checkpoint 重放结果：
`results/crafter_wood3_survival_checkpoint_diagnostic_20261008_v2/result.json`。

adapter 曾错误读取不存在的 `player.food/drink/energy` 属性，导致这三项历史诊断均为
`None`。Crafter 实际把它们存放在 `player.inventory`。该问题只影响诊断字段，策略输入、
奖励、训练和资格结果不受影响。修复后使用六个 100k checkpoint 重放原独立资格集；
120 个 episode 的成功、步数和终止原因均与原结果一致。

| 表示 | 成功 | 死亡 | 外部截断 | wood1 | wood2 | wood3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CNN-only | 32/60 | 26/60 | 2/60 | 49/60 | 39/60 | 32/60 |
| CNN+GRU | 30/60 | 30/60 | 0/60 | 47/60 | 35/60 | 30/60 |

CNN-only 的死亡阶段为首次 wood 前 10 次、wood1 到 wood2 间 10 次、wood2 到 wood3 间
6 次；CNN+GRU 分别为 13、12、5 次。失败分布贯穿定位和重复采集过程。

56 次资格死亡中，53 次死亡时 food、drink、energy 均大于 0。终止步伤害规则提示为
36 次敌人或投射物、1 次熔岩、1 次资源耗尽，另有 18 次混合或未知。这些是依据 health
差值、资源和地形的推断，不能当作直接伤害来源或确证的因果归因。死亡 episode 在
终止前平均经历约 4.9 次掉血。当前资格轨迹支持优先诊断危险导航与敌害下的持续执行，
简单资源耗尽不足以解释大部分死亡。

## 判断与下一步

100k 预算提高了后 75k 区间的训练成功率，但资格仍低于门槛；PPO 的 KL、clip fraction、
value loss 和 explained variance 没有显示训练器爆炸。GRU 没有解决问题。当前基础 Policy
的主要损失来自搜索和重复采集期间的敌害与危险移动。

下一项诊断应固定 deterministic v3 的 CNN-only 基线，只增加一个训练期 terminal death
penalty，并使用同一组 3 seed、25k/100k 交互检查点和独立资格集。主要指标是 100k 独立
资格成功率，次要指标是死亡比例、wood1/2/3 转化和死亡阶段。若明确死亡惩罚仍无收益，
下一步再测试面向导航的辅助目标或结构化空间记忆；不应继续单纯增加 GRU 容量。

在任一策略达到候选 0.8 后，仍需执行完整 Module 契约资格检查。stone、coal 和 pickaxe
需要分别重新验证，不能沿用本轮结论。

2026-10-09 已完成固定 -1.0 死亡惩罚对照；结果与后续判断见
[死亡惩罚诊断记录](crafter-wood3-death-penalty-diagnostic-2026-10-09.md)。
