# Crafter wood>=3 CNN 分层空间读出诊断（2026-10-14）

本轮回答一个限定问题：上一轮自然状态 probe 在最终 128 维 CNN 表示上只有约
62% 的三任务读出，信息究竟在卷积层已经不存在，还是在空间池化和最终压缩中
变得难以读取。它是冻结 Policy 的离线诊断，不是正式训练、Module 资格或
wood 成功率结果。

## 固定边界

- 复用已通过独立审计的自然数据：上一轮的 180 集、2026 个 RGB 状态、同一
  train/validation/heldout split、同一 seed 0 的 100k-step CNN-only checkpoint。
  本轮没有重新运行环境，新增原生 episode 为 0，因此不是新的独立样本。
- 只读取 source encoder，Policy、PPO optimizer、checkpoint 和 RNG 不更新；没有
  action oracle、动作替换、生存干预或死亡 episode 排除。所有 probe 只用 train
  拟合、validation 选 epoch/weight decay；heldout 只在锁定后评分。
- 比较四个 encoder 阶段：第二卷积块的 `32x16x16` 激活、原生平均池化后的
  `32x8x8` 激活、最终线性层的 128 维输出、最终 Tanh 后的 128 维 embedding。
- 为控制读出容量，前两个空间张量用固定 `adaptive max pool 2x2` 压成 128 维，
  后两层本来就是 128 维。四层使用完全相同的 linear 或 MLP64 probe，输入维度
  均为 128，读出参数量分别为 1548 和 9036。

固定 max-pool 是为了避免错误的层间对照：若在两层都继续使用 average pool，
它会和 encoder 中已有的 `AvgPool2d(2)` 交换，两个输出在当前结构上完全相同。
第一次试运行因此废弃；正式 v3 在同一数据和 checkpoint 上使用 max-pool，并把
该规则写入配置与 provenance。

## 留出结果

下表是三个 readout 初始化 seed 的均值。指标是树方向、朝向、距离 2 接近方向
三个离线任务的 balanced accuracy 均值，不是成功率。

| encoder 特征 | probe | 树方向 | 朝向 | 接近方向 | 三任务均值 |
| --- | --- | ---: | ---: | ---: | ---: |
| conv2 空间激活 + max2x2 | linear | 83.7% | 86.9% | 64.9% | 78.5% |
| conv2 空间激活 + max2x2 | MLP64 | 83.3% | 88.1% | 65.2% | 78.8% |
| 原生池化后空间激活 + max2x2 | linear | 71.0% | 70.7% | 63.2% | 68.3% |
| 原生池化后空间激活 + max2x2 | MLP64 | 72.8% | 74.1% | 64.9% | 70.6% |
| projection 前 128 维 | linear | 68.7% | 57.1% | 58.4% | 61.4% |
| projection 前 128 维 | MLP64 | 62.6% | 52.4% | 60.7% | 58.6% |
| 最终 128 维 embedding | linear | 63.9% | 60.3% | 61.4% | 61.9% |
| 最终 128 维 embedding | MLP64 | 62.5% | 57.2% | 57.2% | 58.9% |

四层均使用 535 个 heldout 状态；所有 probe 的输入维度都为 128。linear 和
MLP64 的参数量跨层完全相同，独立分析器也验证了 24 个 probe 的状态、标准化、
预测、数据 digest 和跨进程重复。

## 诊断解释

第一，现有 CNN 并非没有局部空间信息。conv2 的树方向和朝向读出已经达到
83.7%/86.9%，明显高于最终 embedding 的 63.9%/60.3%。因此“RGB 本身读不出
树和朝向”不能解释当前动作链失败。

第二，主要缺口出现在空间信息经过 encoder 的原生 `AvgPool2d` 后。相同 128 维
读出从 conv2 到池化后下降约 10.2 个百分点，说明位置和局部几何在聚合时被
混合，后续线性 projection 不是唯一问题。

第三，projection 前与最终 Tanh 后几乎同一水平，MLP64 也没有稳定补回差距。
这不支持“只增加 head 容量”作为当前首要修复方向；最终 Tanh 也不是主要嫌疑。

这些结果仍不能证明某一种 encoder 改法一定提高自主 wood 成功率。probe 标签是
诊断用几何标签，heldout 被上一轮自然数据使用过，且多个状态来自同一 episode；
它们只定位表示瓶颈，不能替代 RGB-only 动作链 fresh episode 评价。

## 验证状态

- v3 主运行：24 个 probe，19200 次离线 optimizer update，source Policy 更新步数 0。
- 跨进程重复通过：dataset、records、四层 features、predictions、probe states
  全部一致。
- 独立分析器通过：重新加载 checkpoint、重建四层 features、重算全部 24 个 probe，
  输入哈希未变；heldout 未参与 fit/selection。
- 针对性测试 `3 passed`；完整回归需在合并本轮测试后再次执行。

## 下一项

下一步不是注册 Module，而是在同一 source CNN 上实现一个只输入原生 RGB 的空间
动作链对照：actor 读取保留空间布局的 conv2 特征，继续使用原有 `do` 行为 head，
与最终 128 维 CNN-only arm 做 fresh episode 对比。该对照必须重新验证动作概率、
死亡和截断分母，不把离线 probe 的几何标签喂给执行策略。若空间保留确实改善
turn/approach 而不破坏 do，再独立研究奖励/信用分配和共享生存；wood 通过后仍需
按同样边界验证 stone、coal、pickaxe 以及持续任务链。

## 产物

- [配置](../configs/crafter_wood3_cnn_layer_readout_cuda_pilot_v1.yaml)
- [分层特征与协议](../experiments/crafter_cnn_layer_readout.py)
- [运行器](../experiments/run_crafter_wood3_cnn_layer_readout.py)
- [独立分析器](../experiments/analyze_crafter_wood3_cnn_layer_readout.py)
- [留出汇总](../results/crafter_wood3_cnn_layer_readout_cuda_pilot_20261014_v3/analysis_verified/layer_summary.csv)
- [独立验证](../results/crafter_wood3_cnn_layer_readout_cuda_pilot_20261014_v3/analysis_verified/verification.json)

复现时使用全新 output 路径；它会复用已封存的上一轮自然数据，不新增环境样本：

```bash
PYTHONPATH=. PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
/home/zl202621/.conda/envs/kecrl/bin/python -u \
  -m experiments.run_crafter_wood3_cnn_layer_readout \
  --config configs/crafter_wood3_cnn_layer_readout_cuda_pilot_v1.yaml \
  --output results/crafter_wood3_cnn_layer_readout_cuda_REPLAY

PYTHONPATH=. PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
/home/zl202621/.conda/envs/kecrl/bin/python -u \
  -m experiments.analyze_crafter_wood3_cnn_layer_readout \
  --input results/crafter_wood3_cnn_layer_readout_cuda_REPLAY \
  --output results/crafter_wood3_cnn_layer_readout_cuda_REPLAY/analysis_verified
```
