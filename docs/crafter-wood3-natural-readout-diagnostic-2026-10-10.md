# Crafter 冻结 CNN 自然状态动作可读性诊断（2026-10-10）

## 结论

自然状态上的 head 拟合能改善局部执行，但改善尚不足以形成稳定的 wood 执行器。
同一批未见自然状态上，三个冻结 CNN 的 standardized natural head 在 8 步 greedy
窗口中的采木次数为：ready `92/138`、turn `22/138`、approach `34/222`；原 actor
分别为 `67/138、4/138、7/222`。

动作读出仍明显缺少泛化：新标准化 head 的三类平均动作准确率，训练约 `0.828`、
验证约 `0.430`、未见测试约 `0.387`。训练集选出的固定动作是 `do`，三类平均准确率
为 `0.333`；均匀随机合法动作的测试期望为 `0.178`。因此当前结果能确认局部改善，
但不能宣称自然覆盖问题已经解决，也不能将线性读出失败解释为 CNN 不含任务信息。

本轮是 **teacher-assisted head 诊断**：仅在副本上监督拟合线性 actor，冻结原 CNN、
value 网络、PPO 策略与优化器。没有 PPO 更新、Knowledge/SPT 更新、Module 注册或
正式训练；结果不是 `wood>=3` 的自然 reset 资格成功率。

## 固定协议

本轮在提交 `712f14a` 后开始。使用三个已有 CNN-only 100k checkpoint（source seeds
`0/1/2`），四个 actor arm：

| arm | 来源 |
| --- | --- |
| baseline | 原始 PPO actor，冻结 |
| fixture_standardized | 已保存的旧 fixture standardized linear head，冻结 |
| natural_raw | 同一冻结 CNN 上，用自然状态监督拟合 raw linear head 副本 |
| natural_standardized | 同一数据/损失/预算，用训练集 mean/std 坐标拟合，再折回 raw linear head |

三个 checkpoint 共用同一批自然 RGB 和 labels。采集策略按 split 内 episode index
循环使用原 actor `0/1/2`，不使用新 head，不按成功、logits 或预测错误挑样本。
环境为原生 Crafter，自然 reset，使用进程内稳定对象排序
`balance-object-insertion-order-v1`；没有改变世界地形、资源、库存、实体或光照。

| split | 完整 episode 数 | environment seed | action seed |
| --- | ---: | --- | --- |
| train | 48 | `86000000..86000047` | `87000000..87000047` |
| validation | 16 | `86100000..86100015` | `87100000..87100015` |
| heldout | 24 | `86200000..86200023` | `87200000..87200023` |

每个 episode 最多 256 步，到 native done 或 `wood>=3` 停止。每类最多选 6 个
状态，同类状态至少相隔 8 步；只选择清醒、wood<3、具有局部采集机会的状态。
同一 episode 内重复 RGB 不重复采样；跨 split 的完全相同 RGB 按固定 split 顺序
排除后出现的帧，不追加替代 episode。本轮排除 1 个 episode 内重复帧，跨 split
重复为 0。完整 environment seed 和 episode 未跨集合。

状态标签为局部几何规则：

- ready：已朝向无遮挡邻接树，合法目标集合为 `{do}`。
- turn：存在无遮挡邻接树但尚未 ready，保留所有可用转向方向。
- approach：不存在可用邻接树，但有距离 2 的无遮挡可见树与可通行首步；保留
  所有能把任意候选树距离降到 1 的方向。

多个方向都可采集时不强行指定一个类别。另以固定几何顺序保留一棵 designated
tree，用于区分指定树与其他树的采集。主执行指标是 **8 步内产生任何 wood 增量**；
指定树成功为辅助指标。标签是局部动作集合，不是全局最优策略或生存 teacher。

policy 始终只输入 `64×64×3` RGB，合法动作 `0..6`。几何 snapshot 在采集行为动作
选择后读取；它可用于离线标签、采样和结果核验，不进入执行 policy。

## 拟合和测试边界

新 head 从原 actor 函数初始化，训练损失为目标动作集合总概率的负对数，按训练集
三类状态数量做平衡。raw 与 standardized arm 均比较学习率 `0.003/0.03`，每个
1500 次 full-batch 更新，每 50 次检查验证集；先最大化验证三类平均准确率，平分
时最小化验证三类平均 set NLL。均值和标准差仅来自训练集，std 下限 `1e-4`。

三个 source seed 的六个新 head 全部锁定后，才提取 heldout features 并执行测试。
保存 `selection_lock.json`、artifact SHA256、训练/验证 row digest 和 fit feature digest。
heldout 没有参与学习率、epoch 或 head 选择。原 actor 的 `7..16` 非法动作参数行
逐项保持不变；encoder、value 与原 PPO optimizer 未更新。

每个未见状态、source seed、actor arm 运行一次 greedy、两次 sampled 窗口，最多
8 步；配对 arm 使用同一 action seed。产生 wood 增量后继续运行，只有 native done
提前停止，不把一个正确首步等同于成功执行。

## 数据覆盖

共 695 个自然状态，来自固定的 88 个 episode：

| split | ready | turn | approach | 合计 |
| --- | ---: | ---: | ---: | ---: |
| train | 105 | 100 | 171 | 376 |
| validation | 43 | 47 | 63 | 153 |
| heldout | 46 | 46 | 74 | 166 |

训练集 ready/turn/approach 分别来自 40/38/44 个环境 seed；heldout 分别来自
17/16/17 个环境 seed。未出现相应机会的 episode 不补选、不替换。

覆盖仍不均衡：训练集中夜间状态只有 ready 3、turn 1、approach 2；低 health≤3
状态只有 2/1/3。heldout 没有 health≤3 的初始状态。因此本轮不能代表夜间、生存
危机或全部自然访问分布。非零 sapling 的训练状态为 80/78/116，heldout 为
37/38/55；相比旧 fixture，已包含自然库存变化。

## 动作读出

下表为三个 source seed 的三类平均准确率均值。每类按本类状态计算，三个类别
等权；它不是整体任务成功率。

| arm | train | validation | heldout |
| --- | ---: | ---: | ---: |
| baseline | 0.275 | 0.295 | 0.297 |
| fixture_standardized | 0.240 | 0.245 | 0.217 |
| natural_raw | 0.688 | 0.400 | 0.351 |
| natural_standardized | 0.828 | 0.430 | 0.387 |
| 训练集选出的固定 `do` 参考 | 0.333 | 0.333 | 0.333 |
| 均匀合法动作的解析期望 | 0.180 | 0.184 | 0.178 |

固定动作参考只通过 train 选取，随机参考按 labels 解析计算；没有为它们执行
环境窗口，也没有将其作为新的 policy 候选。

natural_standardized 的 heldout 三类平均准确率按 seed 为 `0.319、0.420、0.422`。
natural_raw 为 `0.277、0.403、0.372`。训练与未见测试有明显差距，说明当前 376 个
状态上的拟合效果不足以支持泛化结论。也不能从本轮判定继续增加预算或数据一定
无效；目前仍有样本量、覆盖和线性 head 容量的未分离因素。

## 真实短窗口执行

同一 166 个 heldout 状态在三个 checkpoint 上重复评估，因此 greedy 分母为
ready/turn `46×3=138`，approach `74×3=222`。sampled 每个状态再有两次重复。

| arm / mode | ready 采木 | turn 采木 | approach 采木 |
| --- | ---: | ---: | ---: |
| baseline / greedy | 67/138（48.6%） | 4/138（2.9%） | 7/222（3.2%） |
| fixture_standardized / greedy | 1/138（0.7%） | 2/138（1.4%） | 0/222 |
| natural_raw / greedy | 88/138（63.8%） | 16/138（11.6%） | 20/222（9.0%） |
| natural_standardized / greedy | 92/138（66.7%） | 22/138（15.9%） | 34/222（15.3%） |
| baseline / sampled | 177/276（64.1%） | 82/276（29.7%） | 92/444（20.7%） |
| natural_standardized / sampled | 190/276（68.8%） | 107/276（38.8%） | 127/444（28.6%） |

新 head 有局部改善，但 turn/approach 的连续执行仍明显不足。另有 source seed
差异和阶段权衡：seed 0 原 actor 的 greedy ready 采木为 `41/46`，换成新标准化
head 后为 `25/46`；其余两个 seed 的 ready 分别从 `11/46、15/46` 提升到
`31/46、36/46`。不能用汇总改善表示所有 seed、所有阶段都改善。

natural_standardized greedy 指定树采集为 ready `87/138`、turn `14/138`、approach
`20/222`。这些值低于任何树采木数，说明指定目标选择与“采到 wood”应分别报告。
旧 head 与新 head 的跨诊断比较还改变了数据、标签集合和拟合预算，不能把差异
唯一归因于数据覆盖；本轮 raw/standardized 两个新 arm 的边界保持一致。

## 核验、成本和产物

主评估共 5976 个短窗口，47,748 个原生交互步；source PPO 训练步数为 0。
自然采集 11,110 步，9 个关闭 diagnostics 的采集重放控制 1021 步，12 个窗口
observer 控制 96 步。独立进程重复完整数据采集与 seed 0 两个拟合及所有测试
窗口，包含采集 11,110、采集控制 1021、窗口 15,936、窗口控制 32 步。
总原生步数为 88,074。主进程仅 diagnostic head 的监督更新为 18,000 次，重复
进程为 6000 次；这些更新没有作用于 source policy、CNN 或 PPO optimizer。

- 主/重复进程分别重新核对 32 个输入/source/安装包文件哈希，全部一致。
- 原生 Crafter 安装包代码未修改。三份 source checkpoint 与旧 head 未改变。
- 采集、RGB、records、features、predictions、两个新 head 和窗口轨迹均跨进程一致。
- 独立分析重算 split 边界、采样间隔/数量/重复剔除、几何目标集合、RGB bytes、
  dataset digest、动作概率、验证选择、train-only 标准化、参数 provenance、木材
  增量、指定树标签、连续事件与终止及汇总统计，全部通过。
- 保存 logits 与 source/head 权重的 CPU 重算符合浮点容差；最大 CPU/CUDA logit
  差为 `0.00354`。批量预测与窗口单帧推理首步概率最大差为 `6.55e-5`，greedy
  首步动作分歧为 0。
- 新增 8 项边界/多目标损失/参数冻结/原生 observer 测试；全量回归为
  `251 passed in 41.00s`。

配置：[crafter_wood3_natural_readout_cuda_v1.yaml](../configs/crafter_wood3_natural_readout_cuda_v1.yaml)
；拟合工具：[crafter_natural_readout.py](../experiments/crafter_natural_readout.py)
；runner：[run_crafter_wood3_natural_readout.py](../experiments/run_crafter_wood3_natural_readout.py)
；独立分析：[analyze_crafter_wood3_natural_readout.py](../experiments/analyze_crafter_wood3_natural_readout.py)。
结果为 `results/crafter_wood3_natural_readout_cuda_20261010_v1`，最终分析为
`analysis_verified/`，包含 readout/coverage/window CSV、verification JSON 与
`natural_readout_comparison.png/.svg`。初版 `analysis/` 未包含固定动作参考，保留
原输出；补充解析参考后重新分析，没有修改或重跑主 CUDA 实验。

## 下一步与 SPI

下一项增加**冻结 CNN 的非线性读出容量对照**：在本轮同一 train/validation 数据
和局部目标集合上比较线性与小型 MLP，明确固定其预算和验证选择；再用另一批
新环境 seed 检查动作读出及真实窗口。已经查看过的本轮 heldout 只作为开发证据，
不继续用于新的确认性模型选择或“未见”结果。

这能进一步检查现有 encoder 的信息是否难以由线性 actor 读出。如果非线性读出
仍不能恢复，再考虑自然数据规模/覆盖与受控 encoder 表示更新的对照，保留原
source，并明确 teacher-assisted 范围。当前结果不足以决定必须更换 CNN，也不足
以启动正式 PPO/SPT/SPI 持续训练。

SPI 需要可执行且满足契约的候选 Module。本轮确认部分自然局部动作可以改善，
但尚未解决导航、无树探索、生存和完整 `wood>=3`。局部 teacher 不覆盖这些状态，
不能把新 head 直接当作完整基础 Policy 或 wood Module。候选 0.8 资格门槛尚待
验证，后续仍需自然 reset 资格、完整 Module 契约及管线连续任务检查，之后再
扩展 stone/coal/pickaxe。

本轮为探索性诊断。同一 episode 内状态、采样重复和三个 checkpoint 共用的
测试场景相互依赖，分母是状态暴露次数，不能当作独立资格 episode 或确认性
显著性结论。数据也只代表原 actor 能访问且满足几何条件的状态。

## 复现

```bash
PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/zl202621/.conda/envs/kecrl/bin/python -u -m \
experiments.run_crafter_wood3_natural_readout \
--config configs/crafter_wood3_natural_readout_cuda_v1.yaml \
--output results/crafter_wood3_natural_readout_cuda_NEW

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/zl202621/.conda/envs/kecrl/bin/python -m \
experiments.analyze_crafter_wood3_natural_readout \
--input results/crafter_wood3_natural_readout_cuda_NEW \
--output results/crafter_wood3_natural_readout_cuda_NEW/analysis_verified
```
