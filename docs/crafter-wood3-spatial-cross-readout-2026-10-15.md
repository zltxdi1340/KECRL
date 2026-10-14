# Crafter wood>=3 同状态空间动作头 cross-readout（2026-10-15）

本轮承接 [RGB 空间动作头 fresh 诊断](crafter-wood3-spatial-actor-fresh-2026-10-15.md)。上一轮的三个冻结策略在相同 40 个 seed 上分别实际访问了不同的状态：source 成功 24/40，final embedding head 13/40，conv2 spatial head 22/40。上一轮 fresh 结果只能混合两类因素：策略访问的状态分布不同，以及给定同一状态时动作 head 的输出不同。本轮把这两类因素拆开。

## 固定边界

- 读取上一轮已通过独立审计的 120 个 fresh episode；重新按原环境 seed 和已记录 action 逐步恢复环境，只保存原生 RGB、snapshot、wood 阶段和局部 action-set 标签。状态恢复使用的是已记录动作，不是新的动作采样；恢复完成后不再执行新策略动作。
- 三个冻结评分策略为 source CNN-only、final embedding head 和 conv2 spatial head。每个策略都在三种 path arm 的同一状态池上输出 7-action 概率，共 15,550 个真实访问状态、46,650 行预测。没有 PPO/encoder 更新、没有 oracle、新动作、Module 注册或 Knowledge/SPT 更新。
- 每个 path arm 的 40 集仍按原 fresh seed 配对，初始 RGB 一致。`p(do)` 约束沿用上一轮，两个拟合头的最大浮点误差均在 `5e-7` 容差内。
- 结果按 episode 内状态先求均值，再按 episode 求宏平均，避免长轨迹在汇总中占据全部权重；同时保留逐状态预测，便于后续按阶段和死亡前后重分层。

## 关键结果

表中 `turn p_valid` 和 `approach p_valid` 是评分策略在该 path arm 实际访问状态上，把概率质量分配给当前有效局部动作集合的 episode 宏平均。`trajectory valid` 是该 path arm 原实际动作落在集合内的宏平均，用于描述访问轨迹本身，不是评分策略的新动作命中率。

| 状态来源 path | 评分策略 | turn p_valid | approach p_valid | ready p_do |
| --- | --- | ---: | ---: | ---: |
| source | source | 8.5% | 19.4% | 66.2% |
| source | final embedding head | 8.7% | 17.4% | 66.2% |
| source | conv2 spatial head | **14.8%** | 18.0% | 66.2% |
| final embedding | source | **13.9%** | **21.6%** | 63.5% |
| final embedding | final embedding head | 6.0% | 17.1% | 63.5% |
| final embedding | conv2 spatial head | 12.7% | 20.9% | 63.5% |
| conv2 spatial | source | 10.3% | 18.7% | 62.1% |
| conv2 spatial | final embedding head | 10.5% | 19.3% | 62.1% |
| conv2 spatial | conv2 spatial head | 9.4% | **21.1%** | 62.1% |

每个 path 的实际 trajectory-valid 比例不同，因为三种策略访问到的状态和死亡位置不同；例如 source path 的 turn trajectory-valid 为 14.5%，final path 为 8.2%，conv2 path 为 13.1%。它们不是同一策略的 counterfactual 命中率，不能与上表的 `p_valid` 混用。

## 诊断解释

第一，状态分布确实是重要因素。final head 自己访问的 turn 状态上，source 和 conv2 的有效动作概率质量约为 13.9% 和 12.7%，而 final head 只有 6.0%；这与 final fresh 成功率降到 32.5%、第一根 wood 前死亡 18/40 相一致。单看 validation 状态的 22.3% turn 有效质量会高估它在自己的自然访问状态上的表现。

第二，空间 head 的作用有条件性。conv2 head 在 source path 的 turn 状态上达到 14.8%，高于 source 的 8.5%，但在 conv2 自己的 turn 状态上为 9.4%；它在 conv2 path 的 approach 状态上达到 21.1%，高于 source 的 18.7% 和 final 的 19.3%。因此“保留空间特征一定提升整条动作链”仍未成立；它改善了部分局部 readout，却没有稳定转化为 fresh 成功率提升。

第三，原始 source 在三个状态池上都保持相对平滑的动作分布，而两个 head 的优势随状态来源改变。这支持把当前瓶颈定位为“监督状态到自然自访状态的条件泛化 + 访问分布反馈”，并保留表示聚合损失作为上游因素。当前证据不能把失败单独归因于 RGB 不可读、head 容量、奖励信用或 survival 任一项。

阶段分层还显示，final head 的 27 次死亡中有 18 次发生在第一根 wood 之前；source 是 4 次，conv2 是 8 次。使用冻结 source critic 对相同 RGB 评分后，在 source path 的 episode 宏平均上，value 从 wood 0 的约 0.98（approach）/0.59（turn），下降到 wood 1 后的 0.10/0.18、wood 2 后的 -0.01/0.07；source-path 死亡前最后 10 步的 approach value 约 -0.25。对应折扣 Monte Carlo return（`gamma=0.99`）分别约 1.00/0.93、-0.03/0.07、-0.03/-0.03、-0.16。critic 对任务阶段和临近死亡有粗粒度区分，但各状态组的 episode 宏平均绝对误差约 0.23–0.61，不能据此说 reward/credit 已解决；一个 external-truncation episode 的 return 没有 bootstrap，死亡前 turn 样本也很少。

## 验证状态

- 主运行、跨进程重复均通过：state pool、replay manifest、46,650 条预测 digest、aggregate、episode 宏平均和 `p(do)` 审计完全一致。
- 独立 analyzer 重新加载 source/head/checkpoint，逐步恢复 120 集，重新生成 state pool 与全部预测；验证通过。source Policy 和 PPO optimizer 未改变，输入哈希未改变。
- 全量测试回归：`413 passed`。
- 状态池保存在本机结果目录，压缩后约 18 MiB；GitHub 只归档配置、汇总 CSV、summary/verification JSON，不提交原始 RGB、checkpoint、head artifact 或逐步预测。

## 下一项

暂不进入 survival intervention，也不开始正式 PPO。下一步做一个按 episode 隔离的新状态覆盖对照：从独立新环境 seed 的原策略/冻结 head 轨迹收集 RGB，按 episode 划分 train、validation、heldout；比较现有 216 行训练集与加入自然 on-policy turn/approach 状态后的同容量 head。heldout 状态只在 head 锁定后读出，且另用全新 fresh seeds 检查链路是否改善。这样能检验训练覆盖能否弥合 validation 到自然自访分布的落差。

若覆盖扩展后局部 readout 和 fresh 动作链改善，继续完成 wood 动作链复现；若动作读出已稳定而死亡仍占主要失败，再进入用户提出的共享生存诊断。wood 诊断稳定后，仍需逐项验证 stone、coal、pickaxe、工具前置条件和持续任务链；本轮没有这些任务结论。

## 产物

- [配置](../configs/crafter_wood3_spatial_actor_cross_readout_cuda_pilot_v1.yaml)
- [运行器](../experiments/run_crafter_wood3_spatial_actor_cross_readout.py)
- [独立分析器](../experiments/analyze_crafter_wood3_spatial_actor_cross_readout.py)
- [同状态指标](../results/crafter_wood3_spatial_actor_cross_readout_cuda_pilot_20261015_v1/cross_readout_metrics.csv)
- [公开汇总](../results/crafter_wood3_spatial_actor_cross_readout_cuda_pilot_20261015_v1/analysis_verified/public_summary.json)
- [按阶段/死亡窗口的动作与 value 指标](../results/crafter_wood3_spatial_actor_cross_readout_cuda_pilot_20261015_v1/analysis_verified/stage_death_value_return_readouts.csv)
- [阶段/死亡窗口汇总](../results/crafter_wood3_spatial_actor_cross_readout_cuda_pilot_20261015_v1/analysis_verified/stage_death_value_return_summary.json)
- [独立验证](../results/crafter_wood3_spatial_actor_cross_readout_cuda_pilot_20261015_v1/analysis_verified/verification.json)
