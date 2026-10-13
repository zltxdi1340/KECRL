# Crafter wood>=3 同状态 RGB head 交叉读出（2026-10-13）

本轮只读复核前一轮 seed 0 RGB 学习 pilot 已评估的三组自然轨迹。目的是在完全
相同的 RGB 状态和局部有效动作标签上交叉评分三个冻结 head，以分开 head 遗忘与
策略轨迹状态分布改变。该 seed 和 fresh 结果已被前一份报告打开；本轮不是新的
独立测试，也没有训练、策略采样、重新选择 head 或生存干预。

## 协议

- 输入是三个策略各 30 集的 fresh trace：原 actor、native-state head、chain-state
  head；初始 wood=0、目标 wood>=3、每集最多 256 步。包含所有失败死亡及外部截断。
- 稳定 wrapper 重放 90 集、13750 个环境步，检查完整 RGB、snapshot、动作和转移。
  对局部 `approach / turn / ready` 状态，使用共同 frozen CNN 提取特征，再由三个
  actor head 同时读出同一帧和同一有效动作集合。每个局部状态恰好有三行交叉评分。
- 报告的有效动作概率质量是每帧所有有效动作的 softmax 概率之和；贪心正确率是
  最高概率动作是否在有效集合。轨迹动作有效率记录行为策略在该帧真实采样的动作，
  它对同一个 RGB 状态的三份 head 评分相同，不冒充其他 head 的采样结果。
- 总计 11160 个局部状态、33480 条 head-state 读出记录。按策略来源/head/category
  报告状态均值与 episode 平衡均值；episode、类别状态和跨 head 行均相关，不做把
  它们视为独立样本的置信区间或资格推断。
- fresh 状态与训练集有 14 个事件级精确 RGB 重合，与 heldout 有 3 个；与 validation
  没有精确重合。因此这是对已有诊断的交叉复核，不是全新独立图像集。
- 三个原 head actor、optimizer 和 checkpoint 保持不变，没有 categorical policy sample。
  独立验证确认 33480 条记录的标签、轨迹动作完全匹配，参数和源文件哈希未变。

## 同一状态的 head 差异

表内单元格式为“有效动作概率质量 / 贪心正确率”。行是实际产生 RGB 轨迹的行为
策略，列是对该行同一批 RGB 状态评分的 head。

| 轨迹来源 | 评分 head | ready/do | turn | approach |
| --- | --- | ---: | ---: | ---: |
| 原 actor | 原 actor | 61.1% / 81.7% | 7.4% / 0.0% | 17.6% / 1.9% |
| 原 actor | native-state | 60.2% / 58.5% | 12.8% / 11.2% | 30.3% / 30.9% |
| 原 actor | chain-state | 57.8% / 64.6% | 8.9% / 9.3% | 25.5% / 26.9% |
| native-state | 原 actor | 66.2% / 92.4% | 22.1% / 8.1% | 24.0% / 5.2% |
| native-state | native-state | 2.4% / 1.9% | 1.0% / 1.1% | 13.6% / 12.8% |
| native-state | chain-state | 55.2% / 57.3% | 15.0% / 14.3% | 15.5% / 14.7% |
| chain-state | 原 actor | 66.7% / 94.4% | 9.3% / 0.1% | 15.6% / 0.6% |
| chain-state | native-state | 22.8% / 22.9% | 8.9% / 8.7% | 26.8% / 26.9% |
| chain-state | chain-state | 18.7% / 16.7% | 1.3% / 0.6% | 7.2% / 7.1% |

## 诊断结论

**ready/do 能力遗忘是直接证据。** 在 native-state executor 自己访问的同一批 ready
帧上，原 head 的 do 有效概率质量为 66.2%，native-state 拟合 head 降到 2.4%。在
chain-state 轨迹的同一批 ready 帧上，原 head 为 66.7%，chain-state head 为 18.7%。
轨迹状态完全相同而评分差异很大，所以这部分退化不能解释为 head 只遇到更难状态。

**转向是更普遍的读出难点。** chain-state 轨迹的 turn 帧上，三个 head 的贪心正确率
分别为 0.1%、8.7%、0.6%；native-state turn 帧上为 8.1%、1.1%、14.3%。同一状态换
head 会有差异，但所有 head 的绝对结果仍差，表明只保护 do 还不够。

**接近存在明显 head 与状态分布交互。** chain-state 轨迹的 approach 帧上，
native-state head 有效概率质量为 26.8%、贪心正确率 26.9%，高于原 head 的
15.6%/0.6% 及 chain head 的 7.2%/7.1%。然而 chain head 在自己访问的 approach
状态上只给有效动作 7.2% 概率质量，说明其拟合数据没有覆盖后续自行产生的全部
状态。按 episode 平衡后的值另见 CSV；这里的 pooled 状态均值不能作为独立样本统计。

每个行为策略 30 集都至少出现一个局部无效动作。原 actor 和 chain-state 的首次
局部错误均为 approach（各 30 集）；native-state 为 approach 22、ready/do 4、
turn 4。首次错误不一定终止整集，12/3/4 集仍分别达成目标；它标记最早破坏动作
链的位置，不等于失败的充分原因。

所以当前可区分出两种训练问题：拟合局部类别时破坏已学会的 ready/do 行为，以及
新策略进入的状态上 turn/approach 泛化不足。死亡仍是大量失败的终点，但本轮没有
改变生存机制，不能把它当作上述 readout 退化的唯一原因。RGB encoder 是否包含
足够信息、reward/credit assignment 的独立贡献仍未被本诊断回答。

## 下一项

使用新 episode seed 做同数据、同初始化的两个监督 head 对照：普通有效动作集合
拟合 vs. 在 ready/do 状态上额外蒸馏原 actor 分布。heldout/fresh 使用新的 seed，
执行器仍只看 RGB，关闭 oracle 动作替换，所有死亡保留。该实验只问能否学习
approach/turn 同时保护 do；若 turn 在新保留状态仍普遍失败，再转做空间表示诊断。
局部闭环稳定后，若完整任务仍主要因死亡失败，再开展共享生存能力研究。

这仍是 wood 的基础 SPI executor 可行性工作，不是完整 Module 资格；不注册 Module，
不更新 Knowledge/SPT，也不对 stone、coal、pickaxe 作结论。

## 产物

- [跨 head 分析器](../experiments/analyze_crafter_wood3_rgb_chain_cross_readout.py)
- [按终止原因分组的读出](../results/crafter_wood3_rgb_chain_learning_cuda_pilot_20261013_v1/analysis_cross_readout_v2/cross_readouts.csv)
- [合并终止结果的对照表](../results/crafter_wood3_rgb_chain_learning_cuda_pilot_20261013_v1/analysis_cross_readout_v2/cross_readout_pooled.csv)
- [首次局部失误](../results/crafter_wood3_rgb_chain_learning_cuda_pilot_20261013_v1/analysis_cross_readout_v2/first_failures.csv)
- [回放与冻结边界验证](../results/crafter_wood3_rgb_chain_learning_cuda_pilot_20261013_v1/analysis_cross_readout_v2/verification.json)

交叉分析器测试 `4 passed`；连同此前的动作链审计与学习 pilot 针对性测试共 `36 passed`。
