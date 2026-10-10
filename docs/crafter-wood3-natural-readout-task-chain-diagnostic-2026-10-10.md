# Crafter wood>=3 自然读出完整任务链诊断（2026-10-10）

本轮评估自然状态数据拟合得到的 `standardized_linear` actor head，能否从原生
Crafter 的自然 reset 完成完整 `wood>=3` 子任务。对照是同一个 CNN-only source
checkpoint 的原始 actor。两个策略均被冻结，没有 PPO 更新、重新拟合、模型选择、
Knowledge/SPT 更新或 Module 注册，因此本轮是 SPI 候选执行器的任务链诊断，不是
正式训练和 Module 资格结论。

## 协议

- source seed 为 `0/1/2`，使用三份固定的 100k interaction-step CNN checkpoint。
- 每个 seed、每个 variant 30 个自然 reset episode，最多 256 个交互步；总计 180
  个主 episode。
- 两个 variant 使用同一组配对的环境 seed/action seed。环境种子起点为
  `99000000`，动作种子起点为 `101000000`，seed 间步长为 `1000000`；这些区间
  与先前自然读出数据和旧迁移诊断不重叠。
- 策略输入只有 `64x64x3` RGB，动作 allowlist 为 `0..6`，环境对象排序固定为
  `balance-object-insertion-order-v1`。
- 逐步记录 wood 1/2/3 首次达到步数、health、伤害来源、ready/turn/approach
  状态、动作分布和终止原因。诊断 oracle 只在动作之后读取，不进入 observation。
- 每个策略首个 episode 做 observer 对照；seed 0 做独立进程完整重复。

配置为 `configs/crafter_wood3_natural_readout_task_chain_cuda_v1.yaml`，主结果为
`results/crafter_wood3_natural_readout_task_chain_cuda_20261010_v1`。

## 结果

| source seed | variant | wood>=3 | wood=1 | wood=2 | death | external truncation |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 原始 actor | 14/30 (46.7%) | 25 | 18 | 16 | 0 |
| 0 | standardized linear | 9/30 (30.0%) | 18 | 14 | 18 | 3 |
| 1 | 原始 actor | 15/30 (50.0%) | 25 | 19 | 14 | 1 |
| 1 | standardized linear | 9/30 (30.0%) | 19 | 12 | 18 | 3 |
| 2 | 原始 actor | 14/30 (46.7%) | 19 | 16 | 16 | 0 |
| 2 | standardized linear | 17/30 (56.7%) | 24 | 18 | 11 | 2 |

合并三个 seed 后，原始 actor 为 `43/90 = 47.8%`，linear 为 `35/90 = 38.9%`。
配对四格为：两者都成功 18，对照独有成功 25，linear 独有成功 17，两者都失败
30。linear 在 seed 2 有收益，但 seed 0/1 都下降；这不是稳定的跨 seed 改善。

wood 阶段漏斗（合并三个 seed）：

| variant | wood0 -> wood1 | wood1 -> wood2 | wood2 -> wood3 |
| --- | ---: | ---: | ---: |
| 原始 actor | 69/90 (76.7%) | 53/69 (76.8%) | 43/53 (81.1%) |
| standardized linear | 61/90 (67.8%) | 44/61 (72.1%) | 35/44 (79.5%) |

主要差距首先出现在 wood0 到 wood1 的自然采集入口，之后各阶段仍略低。linear
的 pooled death 为 47，原始 actor 为 46；linear 的 external truncation 为 8，
原始 actor 为 1。

## 动作与失败归因

三个 seed 合并的 ready 状态统计：原始 actor 观察到 423 个 ready 步，执行 `do`
165 次（39.0%）；linear 观察到 1,178 个 ready 步，但只执行 `do` 140 次
（11.9%）。linear 轨迹更长并且到达更多 ready 状态，但在 ready 状态的 `p(do)`
和实际 `do` 率显著更低。它的移动动作位置阻挡次数为 2,890，原始 actor 为 927，
说明当前差异同时包含动作读出和导航状态访问问题。

按 source seed 的 ready `do` 率如下：

| seed | 原始 actor | standardized linear |
| ---: | ---: | ---: |
| 0 | 60.6% | 6.1% |
| 1 | 33.1% | 16.9% |
| 2 | 32.5% | 21.9% |

因此上一轮局部短窗中 linear 的树木采集提升，没有自然地转化为完整任务链的
`wood>=3` 提升。短窗状态固定了树木几何位置；自然 episode 中策略必须先访问
正确位置、维持生存、保持朝向，再重复三次采集。linear head 目前没有同时满足
这些条件。

## 独立复核

- 独立分析器重新计算 6 条主轨迹 JSONL 的 digest、episode 数、交互步、成功数、
  终止计数和 wood 里程碑；全部与 runner 汇总一致。
- 固定三 seed、每 seed 30 episode 的 environment/action seed manifest 全部通过。
- 90 对初始 RGB digest 全部相同，确认配对起始画面一致。
- seed 0 的 baseline 和 standardized linear 跨进程轨迹均一致；独立读取的重复
  结果与主结果分别完全匹配。
- observer audit 全部通过；策略参数、optimizer、输入工件和已安装 Crafter 文件
  的哈希在评估前后未改变。

独立结果位于
`results/crafter_wood3_natural_readout_task_chain_cuda_20261010_v1/analysis_verified_v2/verification.json`，
配对明细位于同目录 `paired_outcomes.csv`。

## 对 SPI 的判断

自然 readout head 已经能在部分 source seed 上完成完整任务链，但没有达到稳定的
候选 Module 资格边界。当前不能把它注册为 wood SPI executor，也不能据此进入
stone、coal 或 pickaxe 的连续任务链。现有证据把主要问题从“局部 actor head 是否
有容量”推进到两个更具体的边界：

1. 同一个自然 RGB 状态上，linear 是否真的给出更好的 turn/approach/do 动作
   概率；
2. linear 是否因为动作改变而进入了更差的后续状态分布，并因此在完整链路中失败。

下一项固定同一批自然状态和同一 CNN encoder，逐状态比较原始 actor 与 linear 的
logits、`p(do)`、合法转向/接近动作概率和 8-step 执行结果；不重新拟合、不使用
新 teacher 标签做训练。该分解完成后，再决定是扩大自然状态覆盖重新拟合、修正
导航/生存奖励，还是回到 PPO 表示与优化条件对照。

## 产物

- [配置](../configs/crafter_wood3_natural_readout_task_chain_cuda_v1.yaml)
- [冻结评估 runner](../experiments/run_crafter_wood3_actor_natural_migration.py)
- [独立分析器](../experiments/analyze_crafter_wood3_actor_natural_migration.py)
- [主摘要](../results/crafter_wood3_natural_readout_task_chain_cuda_20261010_v1/summary.json)
- [独立复核](../results/crafter_wood3_natural_readout_task_chain_cuda_20261010_v1/analysis_verified_v2/verification.json)
