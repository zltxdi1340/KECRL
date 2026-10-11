# Crafter wood>=3 自然状态交叉读出诊断（2026-10-11）

本轮把上一项完整自然 episode 的状态固定下来，在相同 RGB 状态上同时评估原始
CNN-only actor 和自然状态拟合得到的 `standardized_linear` actor head。目的在于
区分两种原因：linear head 本身是否给出了更好的局部动作读出，以及它是否因为
前序动作改变而访问到了不同、更加困难的后续状态。没有 PPO 更新、监督重新拟合、
head 选择、Knowledge/SPT 更新或 Module 注册。

## 协议与复核

- 输入是已完成审计的自然 `wood>=3` 配对评估：三个 source seed、两个路径
  variant、每个 30 episode，共 180 个 episode。
- 每条 episode 按固定 environment seed 和已记录 action 逐步重放，逐步核对
  RGB before/after、reward、wood inventory、终止标志和 collection snapshot。
- 每个重放状态同时送入 baseline 与 standardized linear 两个冻结 actor，记录
  `p(do)`、合法转向/接近动作概率、轨迹动作概率、greedy 合法动作率和熵。
- 主运行重放 `24,267` 个自然状态；seed 0 做独立进程重复。
- 独立分析器重新核对 6 条 trace 的 SHA256、预测 digest、概率归一化和每个状态
  两个 policy variant 的共享字段。主/重复 baseline 与 linear prediction digest
  均一致，所有输入哈希未变化。

配置为 `configs/crafter_wood3_natural_state_cross_readout_cuda_v1.yaml`，结果为
`results/crafter_wood3_natural_state_cross_readout_cuda_20261011_v2`，独立复核为
其 `analysis_verified/verification.json`。

## 核心矩阵

下表中的“路径”表示状态来自哪一种策略实际运行的自然轨迹；“被测 head”表示在
这些相同 RGB 状态上计算哪个 actor 的动作分布。ready/turn/approach 是逐状态
collection oracle 的诊断类别，不是 policy 输入。

| 状态路径 | 被测 head | 类别 | 状态数 | 平均合法动作概率 | 平均 `p(do)` | greedy 合法率 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| baseline | baseline | ready | 423 | 0.386 | 0.386 | 46.3% |
| baseline | linear | ready | 423 | 0.389 | 0.389 | 44.0% |
| baseline | baseline | turn | 787 | 0.091 | 0.455 | 2.9% |
| baseline | linear | turn | 787 | 0.276 | 0.402 | 29.6% |
| baseline | baseline | approach | 6,183 | 0.220 | 0.474 | 12.3% |
| baseline | linear | approach | 6,183 | 0.368 | 0.352 | 37.2% |
| linear | baseline | ready | 1,178 | 0.499 | 0.499 | 64.1% |
| linear | linear | ready | 1,178 | 0.128 | 0.128 | 10.0% |
| linear | baseline | turn | 1,307 | 0.207 | 0.419 | 27.5% |
| linear | linear | turn | 1,307 | 0.072 | 0.567 | 6.7% |
| linear | baseline | approach | 6,544 | 0.295 | 0.373 | 35.7% |
| linear | linear | approach | 6,544 | 0.202 | 0.498 | 18.4% |

## 结果解释

在 baseline 实际访问的自然状态上，linear head 的 ready `p(do)` 与 baseline 基本
相同（`0.389` 对 `0.386`），但 turn 合法动作概率从 `0.091` 提升到 `0.276`，
approach 的 greedy 合法率从 `12.3%` 提升到 `37.2%`。这确认线性 head 能利用
现有 CNN 表征中的局部转向和接近信号。

然而，在 linear 自己实际访问的状态上，linear 的 ready `p(do)` 只有 `0.128`，
baseline 在同一批状态上为 `0.499`；linear 的 turn 合法动作概率为 `0.072`，
baseline 为 `0.207`。这说明 linear 的动作改变了状态访问分布，使其进入了对
linear 自身读出不利的自然 RGB 状态区域。它并非单纯“在同一状态上动作更差”，而
是“局部动作改进与后续状态分布退化同时存在”。

这与完整任务链结果一致：linear 在 seed 2 成功率达到 `56.7%`，但 seed 0/1 为
`30.0%`，三 seed pooled `35/90 = 38.9%`，低于原始 actor 的 `43/90 = 47.8%`。
因此不能把自然 linear head 直接注册为 wood SPI executor；当前仍没有跨 seed、
跨阶段稳定的 Module 资格证据。

## 对后续 SPI 设计的作用

本轮把阻塞点从“actor head 容量是否足够”进一步拆成两个可操作问题：

1. 现有 CNN 是否覆盖自然任务中需要的状态，特别是移动后形成的 approach/turn
   状态；
2. head 优化是否只拟合了局部几何动作，而没有约束动作后的生存、导航和下一阶段
   可达性。

下一项应在保持 encoder 和环境 wrapper 不变的前提下，扩大自然状态覆盖，加入按
   episode 隔离的 validation/heldout，并评估一个只改变训练目标或 rollout coverage
   的单因素对照。该对照仍先看自然 `wood>=3` 资格与阶段漏斗，不进入 stone、coal
   或 pickaxe；只有 wood executor 满足完整 Module 契约后，才把相同诊断方法扩展
   到挖矿和连续任务链。

## 产物

- [配置](../configs/crafter_wood3_natural_state_cross_readout_cuda_v1.yaml)
- [交叉读出 runner](../experiments/run_crafter_wood3_natural_state_cross_readout.py)
- [独立分析器](../experiments/analyze_crafter_wood3_natural_state_cross_readout.py)
- [主摘要](../results/crafter_wood3_natural_state_cross_readout_cuda_20261011_v2/summary.json)
- [独立复核](../results/crafter_wood3_natural_state_cross_readout_cuda_20261011_v2/analysis_verified/verification.json)
- [动作矩阵](../results/crafter_wood3_natural_state_cross_readout_cuda_20261011_v2/analysis_verified/matrix.csv)
