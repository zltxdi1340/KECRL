# Crafter wood>=3 动作后果审计（2026-10-12）

本轮复用上一轮 matched 对照的 180 个完整自然 episode，对动作后的局部状态、采集、
存活和 8 步延续做独立只读分析。新增环境交互为 0，没有重新拟合 head、PPO 更新或
encoder 更新，没有 Module 注册或 Knowledge/SPT 更新。

当前最具体的断点是：策略经常在已经 ready 时重复无法移动的方向动作，没有衔接 do。
这比“局部动作标签错误”更符合本批 trace，但描述性审计不能证明修复这一点就能解决
完整任务中的导航、生存和后续阶段问题。

## 输入与统计口径

- 输入为 `crafter_wood3_natural_coverage_matched_cuda_20261011_v1`，三个 policy seed，
  每 seed 每臂 30 集。两臂都为冻结 100k CNN 上重新监督拟合的标准化线性 head；
  `matched_baseline_only` 不是原始 PPO actor。
- 主 trace 共 26,253 步：baseline-only 12,737 步，mixed coverage 13,516 步。
  两臂初始环境、action RNG seed、RGB 与 collection snapshot 配对；动作后路径不同。
- ready/turn/approach 为互斥诊断类别，优先级依次下降；approach 使用已有局部可通行、
  减少到可见树 Manhattan 距离的动作集合，不表示全局导航安全或完整路径可达。
- 单步采集与存活分别计数；采到 wood 的同一步仍可能死亡。
- 8 步延续只看当前动作之后的最多 8 个动作。当前动作已死亡或已达到 wood=3 的起点
  不进入延续分母；后续成功/死亡结束窗口，外部截断导致的不足 8 步窗口单列为删失。
  complete-case 率和包含删失不确定性的上下界均输出，不把未观察未来记成失败。
- 多步窗口相互重叠，且每集会反复访问局部状态。所有步级比例都是相关观察的描述统计，
  不作为独立 episode 成功率、置信区间或动作的因果效应。

## 单步后果与动作衔接

| 指标 | matched baseline-only | matched mixed coverage |
| --- | ---: | ---: |
| 原有完整 wood>=3 成功率，本轮未重测 | 29/90 = 32.2% | 26/90 = 28.9% |
| ready 时执行 do | 125/1055 = 11.8% | 129/2248 = 5.7% |
| ready+do 后 wood 增加 1 | 125/125 | 129/129 |
| ready 非 do 为清醒方向动作但位置未变 | 827/930 = 88.9% | 1975/2119 = 93.2% |
| ready 非 do 后下一帧仍 ready | 827/930 = 88.9% | 1950/2119 = 92.0% |
| turn 时选择局部有效转向动作 | 69/1118 = 6.2% | 122/1472 = 8.3% |
| 有效转向后下一帧 ready | 69/69 | 122/122 |
| 有效转向后紧接着执行 do | 20/69 = 29.0% | 27/122 = 22.1% |
| 有效转向后 8 步内采到 wood | 39/69 = 56.5% | 49/122 = 40.2% |
| approach 时选择局部有效接近动作 | 1323/7169 = 18.5% | 1519/6982 = 21.8% |
| 有效接近后可观察到的最近树距离减少 | 1323/1323 | 1518/1518 |

mixed 有一次有效 approach 当步死亡，无法读取 after snapshot，因此不进入最后一行
分母。mixed 的 129 次 ready+do 中有一次采到 wood 的同时死亡，不能把采集正确率
解释成生存保证。

所有观察到的 ready+do 都符合原生一次交互采一份 wood 的规则，有效转向也都正确改变了
下一帧朝向。真正的执行缺口同时存在于选择和衔接：有效转向/接近本身很少被选择，已经
对准树之后又经常继续按方向键。mixed 的 ready 访问步数增加了一倍多，实际采集仅从
125 增至 129，完整成功没有提高；更多 ready 步还包含原地停滞造成的重复访问，不能
当作独立采集机会数量增加的证据。

ready 时 do 的分 seed 比例：

| seed | baseline-only | mixed coverage |
| ---: | ---: | ---: |
| 0 | 28/290 = 9.7% | 29/744 = 3.9% |
| 1 | 58/202 = 28.7% | 41/685 = 6.0% |
| 2 | 39/563 = 6.9% | 59/819 = 7.2% |

因此不能宣称 mixed 在每个 seed 都退化。没有半径 3 内 hostile 的 ready 步中，do 比例
仍仅为 baseline-only `108/966 = 11.2%`、mixed `109/1670 = 6.5%`，现象并不限于
记录到附近敌人的时刻；该 proximity 标签也不证明伤害来源或全局安全。

## 后续 8 步与解释边界

ready 时错过 do、当步仍存活的观察中，后续 8 步采到 wood 的比例为 baseline-only
`210/929 = 22.6%`、mixed `288/2106 = 13.7%`。这些分层没有外部截断删失。
它们描述错过采集之后的自然策略延续，不是“选择非 do 导致这些结果”的因果估计。

全体可延续起点为 12,650/13,429 个，各有 24 个窗口因外部截断而删失。相关计数和
各 wood=0/1/2 阶段、seed、局部动作与 hostile 条件的分层均保存为 CSV。

本轮没有训练 consequence/feasibility 目标，也没有验证这种目标会带来收益。观察到
原地阻塞和低 do 率，仍不能独立区分 RGB 泛化不足、head 拟合/校准不足、长期信用分配
以及生存压力的贡献；更不能据此排除更大或更合适的覆盖数据。

## 验证与复现

独立分析器重新校验主运行 6 条 trace 的文件 SHA256 和 canonical digest，逐 event
核对相邻 RGB/snapshot/wood 连续性、原生采集规则、health delta、终止边界、动作
概率与 allowlist，以及重新计算的 episode success、milestone 和 summary。
还重新加载 seed 0 的两条独立进程 trace，确认其 digest 与主运行完全一致。
输入文件、原始 checkpoint、数据、环境安装包和源码的分析前后哈希均未变化。

```bash
/home/zl202621/.conda/envs/kecrl/bin/python -m experiments.analyze_crafter_wood3_action_consequence \
  --config configs/crafter_wood3_action_consequence_analysis_v1.yaml \
  --input results/crafter_wood3_natural_coverage_matched_cuda_20261011_v1 \
  --output results/crafter_wood3_action_consequence_analysis_20261012_v3/analysis
```

分析器只用 Python 标准库，不导入模型或环境。输出目录必须是新目录，禁止覆盖已有
记录。早期 `20261011_v1` 和 `20261012_v1` 输出属于汇总口径调试，不用于本报告；
`20261012_v2` 与最终 v3 统计相同，v3 将 CSV 换行统一为 LF 并重新记录哈希。

新增 14 项针对性测试覆盖跨 seed episode 计数、turn->ready->do 衔接、当前动作和未来
增量分离、成功/死亡的延续边界、外部截断删失、ready 阻塞，以及 trace 篡改拒绝。
全量回归为 `280 passed`。第二个独立分析进程生成的 7 个文件与主分析逐字节一致；
另从原始 trace 独立重算的采集、阻塞、after-ready 和 next-do 关键计数全部相符。

## 下一项与 SPI 关系

下一项先做最小动作干预诊断：冻结现有 head、encoder 和稳定 wrapper，使用全新自然
episode，在每个 head 内配对比较原策略和“仅在 ready 时执行 do”的诊断副本。每步
仍先采样并记录原策略动作，再记录是否替换和实际执行动作，保持动作 RNG 抽样规则
一致；其他局部状态和生存行为不干预，环境轨迹允许随动作分歧。

该副本明确使用 collection oracle，不能称为 RGB-only executor，也不能进入正式
Module 资格。如果仅补齐 ready->do 就明显改善完整阶段漏斗，则优先设计只用 RGB
学习这一衔接的目标；若完整任务仍失败，则继续定位 turn/approach 和生存。原有全量
局部 head 拟合、MLP 容量和 wood-gain auxiliary 已做过，本轮不重复将它们视为新假设。

这些诊断用于收紧 wood SPI 的程序与学习规格；只有 RGB executor 通过独立成功率评估
和完整 Module 契约资格检查，才可注册 Module。候选 0.8 门槛仍未验证，本轮没有
stone、coal、pickaxe 或持续任务链的新资格结论。

上述最小干预已完成，见[ready->do 干预报告](crafter-wood3-ready-do-intervention-diagnostic-2026-10-12.md)。
新批次配对结果支持衔接缺口影响完整任务，但 oracle gate 后 pooled 成功率仍仅为
33.3%/51.1%；下一项定位到达 ready 之前的局部转向与接近问题。

## 产物

- [分析配置](../../../configs/crafter_wood3_action_consequence_analysis_v1.yaml)
- [独立分析器](../../../experiments/analyze_crafter_wood3_action_consequence.py)
- [针对性测试](../../../tests/test_crafter_action_consequence.py)
- [验证清单](../../../results/crafter_wood3_action_consequence_analysis_20261012_v3/analysis/verification.json)
- [局部动作后果](../../../results/crafter_wood3_action_consequence_analysis_20261012_v3/analysis/transition_outcomes_by_local_action.csv)
- [分 seed 后果](../../../results/crafter_wood3_action_consequence_analysis_20261012_v3/analysis/local_action_outcomes_by_seed.csv)
