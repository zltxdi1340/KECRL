# Crafter wood>=3 RGB 动作链与覆盖审计（2026-10-13）

本轮按“先检查智能体能否自己完成接近、转向、do，再研究跨任务生存”的顺序推进。
先复用上一轮完整轨迹做只读审计，随后进行
[独立 RGB 学习 pilot](crafter-wood3-rgb-chain-learning-diagnostic-2026-10-13.md)。
本报告的数字仅来自既有轨迹；没有增加环境交互、CUDA 推理或监督更新。

## 审计范围

输入为上轮局部接近增量诊断的 360 集、21833 个交互事件，policy seed 为 0/1/2。
两种既有 head 为 `matched_baseline_only`、`matched_mixed_coverage`，条件为
`turn_ready_do` 和 `turn_ready_do_approach`。全部 70 集失败死亡和 8 集外部截断保留，
没有改变完整任务的成功率分母。原始轨迹此前已经通过全部独立 CUDA RGB/环境回放。

本轮恢复每个 head 实际参与训练的行索引、监督标签、来源 snapshot 和 RGB digest，
核对拟合清单的行数与 digest。旧拟合/选择 episode 的环境及 action seed 与本轮
轨迹分离；审计结果不回流到训练或重新选 head。

比较的是同一记录状态上原 RGB head 的概率、原始采样与实际执行动作。oracle
修改了执行动作，所以这些状态分布不能被当成关闭 oracle 后的自然轨迹。

## 原 head 仍不能自主衔接动作

以下仅列 `turn_ready_do_approach` 条件，汇总三个 policy seed。
“原采样正确”依据该状态的局部有效动作集合；实际执行在这三类状态均被门控保证有效。

| head | 类别 | 状态数 | 原采样正确 | 原概率质量均值 | 贪心动作正确 |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline-only | approach | 703 | 228/703 = 32.4% | 31.8% | 38.7% |
| baseline-only | turn | 92 | 5/92 = 5.4% | 7.1% | 5.4% |
| baseline-only | ready/do | 259 | 78/259 = 30.1% | 29.3% | 23.2% |
| mixed coverage | approach | 795 | 286/795 = 36.0% | 34.7% | 38.0% |
| mixed coverage | turn | 90 | 16/90 = 17.8% | 15.1% | 15.6% |
| mixed coverage | ready/do | 261 | 67/261 = 25.7% | 26.5% | 25.7% |

按连续局部机会片段划分，在 oracle 记录路径上，baseline 完成了 259 次单份 wood
采集，其中 252 次包含 approach；这 252 段中，仅 3 段的全部原采样均有效，全部
贪心动作有效仅 1 段。mixed 对应为 261 次采集、252 个含 approach 片段，全部原
采样有效 6 段，全部贪心动作有效 3 段。原采样首次失误常在 approach，随后仍有
turn 和 ready 错误，不能把接近修好等同于动作链修好。

每段有效概率的连乘只描述固定记录状态上的联合概率。发生动作分歧后自然轨迹会改变，
因此它不是自主成功率，也没有把高度相关的状态或片段作为独立资格 episode。

## 覆盖不足有证据，但不是唯一解释

粗粒度覆盖签名使用 `category / wood_stage / distance_bucket / valid_actions / facing`，
只与各 head 的实际训练行比较。增强 oracle 条件下，baseline approach 中
`463/703 = 65.9%`、mixed 中 `533/795 = 67.0%` 的签名未见于各自训练集。
turn 未覆盖分别为 `24/92`、`20/90`；ready 未覆盖为 `10/259`、`24/261`。

这支持把连续动作链的状态覆盖作为下一项可干预因素，但签名相同不意味着 RGB
背景相同、特征相同或类别可分。ready 和 turn 已覆盖比例较高，动作仍很差，所以
“增加覆盖即可解决”尚未成立，也不能从这份审计判断 CNN 信息是否足够、线性
head 是否能泛化，或 PPO 奖励/信用分配的独立贡献。

不同 episode seed 仍可能产生相同 RGB：oracle 条件中的 approach 精确训练 RGB
重合分别有 2/3 个状态，拟合加验证集合重合均有 3 个状态。因此本轮只声称 episode
seed 独立，明确报告精确 RGB 重合，不声称每张图都未见过。

## 死亡处理与后续路线

核查当前安装的 Crafter `objects.py`、`data.yaml`：没有摔落或溺水机制，Player
不能进入 water；存在敌人/投射物伤害、资源耗尽以及 lava。当前数据没有可剔除的
“摔落/溺水”死亡。所有失败死亡继续进入总结果，局部动作指标单独观察，不删去
失败轨迹来抬高成功率。详细生存干预暂不加入动作链学习 pilot。

下一项使用新的监督、验证、保留及自然评估 episode seed；冻结同一个 CNN 和
PPO checkpoint，仅在副本上拟合相同线性 head。两臂各类别训练状态数、初始化和
优化更新数相同，比较原策略状态与 oracle 连续采集链状态。oracle 仅负责生成
训练路径和监督标签，fresh 执行必须只接收 RGB、关闭动作替换。保留集不参与拟合
或选模型。这个诊断先判断可学性及状态覆盖作用，不能作为无 teacher 正式训练结论。

若自主局部动作明显改善、完整任务仍主要被死亡阻断，就转入共享生存能力诊断；若
局部泛化仍差，先定位 RGB 表示与读出困难，不等到总成功率 0.8 才调整路线。
wood 动作链是 SPI 实例执行器的一部分；未通过完整 Module 契约检查，不注册 Module、
不更新 Knowledge/SPT，也不能推广到 stone、coal 或 pickaxe。

## 产物与验证

- [固定配置](../configs/crafter_wood3_rgb_chain_audit_v1.yaml)
- [只读分析器](../experiments/analyze_crafter_wood3_rgb_chain.py)
- [验证清单](../results/crafter_wood3_rgb_chain_audit_20261013_v1/verification.json)
- [局部读出汇总](../results/crafter_wood3_rgb_chain_audit_20261013_v1/readout_summary.csv)
- [连续片段汇总](../results/crafter_wood3_rgb_chain_audit_20261013_v1/chain_summary.csv)

针对审计的测试为 `16 passed`；与学习 pilot 代码一并完成的全量回归为 `380 passed`。
配置、输入文件和源代码在运行前后哈希不变。逐状态 CSV、逐片段 CSV 及完整 trace
保留在本地 results，GitHub 归档摘要和验证清单。
