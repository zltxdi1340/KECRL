# Crafter wood>=3 ready-only 蒸馏诊断（2026-10-14）

本轮比较普通动作集合拟合与 ready 状态上蒸馏原 actor 分布，检验能否在学习
approach/turn 时保住已有 do 行为。原 actor、新的普通 head、蒸馏 head 在共同的
30 个新自然 episode 上分别成功 **13/30、5/30、10/30**。蒸馏臂高于普通臂，
仍低于原 actor；单一来源策略 seed 的 pilot 不构成稳定训练收益或 Module 资格。

## 固定协议

- 冻结原 seed 0 的 100k-step CNN-only checkpoint。训练只更新 actor 副本；
  encoder、critic、原 actor、PPO optimizer 和 Crafter 安装包保持不变。
- 两臂精确复用前一轮 native-state head 的同一批 216 行 RGB 数据，
  ready/turn/approach 配额为 90/30/96。分别来自 25/23/31 个训练 episode；
  状态数不能当作独立 episode 数。训练行 digest 与旧 artifact 完全一致。
- 新 validation/heldout 各 16 集，只由原 actor 自然采样。validation 环境/动作
  seed 从 185000000/195000000 开始，heldout 从 186000000/196000000 开始；
  fresh 从 187000000/197000000 开始，各策略 30 集。监督集与此前整个 pilot
  排除精确 RGB 重合，validation 和 heldout 之间也排除重合。
- 新 validation 共 220 行（58/66/96），heldout 共 183 行（42/48/93）。
  三类标签依次为 ready、turn、approach，来自真实环境几何诊断；标签不进入
  fresh 的动作选择。采集没有使用 oracle 动作链，fresh 只输入 RGB。
- 两臂用相同原 actor 初始化、训练集标准化、Adam、候选 lr `[0.003, 0.03]`，
  每个候选 1500 次更新；每臂总计 3000 次更新。每 50 次更新检查 validation，
  先最大化类别宏准确率，再最小化类别宏 set-NLL。两臂使用同一个选择规则。
  head 在 heldout 评分和 fresh 执行前锁定；heldout 数据从未传入拟合或选择。
- 普通臂选择 lr=0.003、epoch=650；蒸馏臂选择 lr=0.003、epoch=800。
  这是相同验证规则的选择结果，不能称作相同选中 epoch 的对照。
- 初始 wood=0、目标 wood>=3、每集最多 256 步、allowlist 0..6。
  共同 seed 保证初始状态一致，后续轨迹及实际交互步数允许不同。
  原 actor/普通/蒸馏的 fresh 实际步数为 3412/4867/3626。
- 所有死亡和外部截断计入分母；没有生存干预，没有注册 Module 或更新
  Knowledge/SPT，也没有启动正式 PPO 训练。

设类别等权重为 `w_i = N/(3*N_category(i))`。普通臂目标为
`mean(w_i * set_NLL_i)`；蒸馏臂只额外加入
`mean(w_i * 1_ready(i) * KL(p_original || p_head))`，固定系数为 1.0。
teacher 是冻结原 actor 在同一训练 RGB 上、allowlist 0..6 内的完整分布，
不是把 do 概率直接指定成 1。KL 也按类别权重计算，因此等于 ready 行平均 KL 的
三分之一。没有在打开 heldout/fresh 后选择蒸馏系数。

## 新 heldout 读出

每格为“有效动作概率质量 / 贪心有效率”。同一列的三种 head 使用相同 RGB
和有效动作集合；这是一步离线读出，不是完整任务成功率。

| head | ready/do（42 行） | turn（48 行） | approach（93 行） |
| --- | ---: | ---: | ---: |
| 原 actor | 64.5% / 100.0% | 6.1% / 0.0% | 16.9% / 0.0% |
| 普通集合拟合 | 52.1% / 57.1% | 30.6% / 29.2% | 40.1% / 44.1% |
| ready 蒸馏 | 42.1% / 42.9% | 31.5% / 31.3% | 43.0% / 53.8% |

普通/蒸馏臂的训练类别宏准确率为 89.2%/89.6%，heldout 为 43.5%/42.6%。
被选中 head 的训练 ready 加权 KL 为 0.440/0.090，说明蒸馏目标确实生效；
但在新 ready 状态上，蒸馏没有比普通臂更好地保护 do。

## 自主自然 episode

| 策略 | wood>=3 | wood>=1 | wood>=2 | 死亡 | 外部截断 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原 actor | 13/30（43.3%） | 27/30 | 20/30 | 17 | 0 |
| 普通集合拟合 | 5/30（16.7%） | 19/30 | 11/30 | 23 | 2 |
| ready 蒸馏 | 10/30（33.3%） | 18/30 | 14/30 | 20 | 0 |

各自轨迹上 ready 有效概率质量为 63.1%/9.2%/11.4%；turn 贪心有效率为
0.0%/5.9%/6.7%；approach 有效概率质量为 19.2%/12.3%/30.5%。
这些数值受各策略访问状态分布影响，必须结合独立分析器的同状态交叉评分解释。

独立分析器在 9177 个局部状态上交叉评分三种 head，共 27531 条记录。以下是
同一组 ready 帧的 do 概率质量；行是轨迹来源，列是评分 head：

| ready 轨迹来源 | 原 actor | 普通 head | 蒸馏 head |
| --- | ---: | ---: | ---: |
| 原 actor（90 帧） | 63.1% | 51.0% | 40.7% |
| 普通 head（415 帧） | 68.6% | 9.2% | 10.1% |
| 蒸馏 head（386 帧） | 66.2% | 36.2% | 11.4% |

蒸馏策略自己的 ready 状态上，原 actor 给 do 66.2%，蒸馏 head 只有 11.4%；
按 episode 平衡后仍为 65.5% 对 23.1%。因此退化不只是三个策略遇到不同状态。
在蒸馏轨迹的同一批 turn 帧上，原/普通/蒸馏 head 贪心有效率为
0.4%/10.2%/6.7%；转向仍没有稳定。两种新 head 在原 actor 轨迹上的 turn
读出更好，进一步说明泛化随状态分布改变。

表格使用状态均值；长时间停在同一局部类别的 episode 会贡献更多帧，因此另存
episode 平衡均值。共同 seed 只配对初始环境与动作 RNG，不强制后续状态相同。
新 fresh 轨迹与训练集仍有 2 个事件的精确 RGB 重合，与 validation/heldout 有
3/5 个；新 episode seed 不能等同于全新像素集合。三臂、同 episode 的帧和
跨 head 评分相关，不把这些行当作独立样本做资格或统计显著性结论。

原 actor/普通/蒸馏的首次局部无效动作分别为：approach 28/27/29 集，ready
1/3/0 集；另有原 actor 和蒸馏各 1 集未观察到局部无效动作。首次错误不等于
失败的充分原因：后续仍可能恢复并达成任务。

## 判断与后续

本协议下，ready-only 蒸馏在训练样本上降低 teacher 分布偏移，却没有可靠保留
新状态的 do 行为，也没有解决自主动作链。10/30 对 5/30 是本组配对 pilot 的
描述性差异，不是跨来源策略 seed 的稳定优势；13/30 原 actor 仍更高。

目前可确认的是局部 head 泛化与能力干扰仍存在。不能由这个负结果推出所有
蒸馏系数、数据覆盖或保留能力方法都无效；也不能由离线改善推断奖励/信用分配
已解决、RGB 信息不足，或死亡是唯一原因。之前在构造场景上的高可读性不代表
夜间、实体干扰等自然失败状态同样可读。

下一项聚焦真实自然失败状态的分项空间读出：在相同 RGB/标签上比较冻结 CNN
最终表示与固定玩家中心的局部 RGB，分别检查树方向、玩家朝向和可执行接近方向。
训练/验证/保留仍按 episode 隔离，使用新保留 seed；按局部几何与昼夜/资源条件
分组，容量差异单独报告。这一 probe 只解释信息可读性，不接管策略。
根据结果再决定修表示、补状态覆盖或修改 head 学习目标。局部自主链改善后，
继续开展跨任务共享生存研究；总成功率 0.8 不是开始生存研究的硬条件。

这仍是 wood 的基础 SPI executor 可行性工作。wood 修通后还需验证挖矿、工具
前置条件、组合调用与持续任务链；本轮没有 stone/coal/pickaxe 结论。

## 产物与复现

- [固定配置](../../../configs/crafter_wood3_rgb_chain_ready_distillation_cuda_pilot_v1.yaml)
- [运行器](../../../experiments/run_crafter_wood3_rgb_chain_ready_distillation.py)
- [独立分析器](../../../experiments/analyze_crafter_wood3_rgb_chain_ready_distillation.py)
- [完整任务结果](../../../results/crafter_wood3_rgb_chain_ready_distillation_cuda_pilot_20261014_v1/analysis_verified/outcomes.csv)
- [同状态交叉读出](../../../results/crafter_wood3_rgb_chain_ready_distillation_cuda_pilot_20261014_v1/analysis_verified/cross_readout_pooled.csv)
- [独立验证](../../../results/crafter_wood3_rgb_chain_ready_distillation_cuda_pilot_20261014_v1/analysis_verified/verification.json)

运行器的跨进程完整重复已通过，数据、特征、head 和轨迹 digest 一致。
独立分析器重建 216 行旧训练数据及全部新 split，核对 32 集新监督轨迹的
4439 个源轨迹步、90 集 fresh 的 11905 个源轨迹步，并重算特征、head 指标、
验证选择和同状态交叉读出；实际动作与记录概率的最大绝对误差为 0。
三臂共同初始状态、相同拟合输入/预算和验证选择检查均通过。Policy、PPO
optimizer 和封存源文件保持不变；完成后两张 GPU 空闲。
全量回归 `391 passed`，新增 7 项测试验证 ready-only 梯度范围、共同数据/预算、
新 seed 边界、验证选择规则、监督状态重建及普通臂与旧拟合器的等价性。

GitHub 归档协议、源码、测试、中文报告和紧凑分析摘要。完整 checkpoint、RGB、
特征、head artifact 和逐步轨迹保留在实验机器；新检出不能直接恢复本机原始产物。
已有输入可在实验机器用全新的 output 路径重复本协议（不会成为新的独立测试）：

```bash
PYTHONPATH=. PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
/home/zl202621/.conda/envs/kecrl/bin/python -u \
  -m experiments.run_crafter_wood3_rgb_chain_ready_distillation \
  --config configs/crafter_wood3_rgb_chain_ready_distillation_cuda_pilot_v1.yaml \
  --output results/crafter_wood3_rgb_chain_ready_distillation_cuda_REPLAY

PYTHONPATH=. PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
/home/zl202621/.conda/envs/kecrl/bin/python -u \
  -m experiments.analyze_crafter_wood3_rgb_chain_ready_distillation \
  --input results/crafter_wood3_rgb_chain_ready_distillation_cuda_REPLAY \
  --output results/crafter_wood3_rgb_chain_ready_distillation_cuda_REPLAY/analysis_verified
```
