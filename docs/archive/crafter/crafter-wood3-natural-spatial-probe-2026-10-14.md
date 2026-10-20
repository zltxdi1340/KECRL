# Crafter wood>=3 自然状态空间分项读出（2026-10-14）

本轮承接 ready-only 蒸馏诊断，检查自然失败状态中的树方向、玩家朝向和局部
接近方向是否能从现有 CNN 表示读出。它是 wood SPI executor 的可行性诊断；
probe 不参与动作选择，不更新 Policy，不构成 Module 资格或正式训练结果。

## 固定协议

- 来源固定为 seed 0 的 100k-step CNN-only checkpoint，以及上一轮已锁定的
  普通集合 head 和 ready-only 蒸馏 head。三个冻结行为策略只输入原生 64x64 RGB，
  按各自分布采样，没有动作替换、oracle 动作链或生存干预。
- 每种行为策略采集 train/validation/heldout 各 32/12/16 集，共 180 集，
  对应 60 个环境 seed。同一 seed 的三个策略属于同一 split，不能当作三个
  独立环境样本。初始状态配对不要求后续轨迹相同。
- 三个 split 的环境 seed 分别从 201000000/202000000/203000000 开始，
  动作 seed 从 211000000/212000000/213000000 开始。每集 wood=0 开始，
  目标 wood>=3，最多 256 步，allowlist 为 0..6。
- 每个 episode 的 ready、turn、approach_near、other 最多各取 4 个状态，
  同类别采样间隔至少 8 步；按几何、RGB 身份和步数选择，不按动作是否正确或
  最终成功/死亡选择。保留所有原始轨迹及死亡、截断分母。
- 同 episode 内去除精确重复 RGB；排除前两轮已选监督 RGB，并排除跨 split
  精确重复 RGB。同一 split 内允许不同策略在共同初始状态重复，按环境 seed
  分组。无重复像素不代表没有相似几何，另报告几何覆盖。
- 先采集 train/validation 并拟合全部 probe，写入 selection lock 后才采集
  heldout。所有标准化和类别权重只用 train；验证集选择 epoch 和 weight decay，
  heldout 从不参与拟合或选模。

树方向是四个方向上是否有未被实体占据的相邻树，使用四位集合，不强制在多棵
树之间指定唯一方向。玩家朝向是左、右、上、下四分类。接近方向只在最近可见
未阻挡树的 Manhattan 距离恰为 2、且没有睡眠动作覆盖时定义，标出相邻可通行
且移动后能与树相邻的方向；允许多方向或空集合。这里的“可执行”只指几何
通行条件，不代表对敌对实体安全，不评估远距离导航。

## 相同数据上的对照

四个输入条件为已训练 CNN 的最终 128 维表示、同结构随机 CNN 的 128 维表示、
固定玩家中心 5x5 格 RGB，以及已训练 CNN 加打乱训练标签的负对照。固定裁剪为
RGB 行 `[7:42]`、列 `[14:49]`，共 35x35x3=3675 维，不根据树位置调整。
距离为 2 的限制确保局部 RGB 包含接近标签所需的树和相邻通行信息。

每种条件分别拟合 linear 和 MLP64，初始化 seed 为 0/1/2，共 24 个 probe。
这些 seed 只衡量离线读出初始化变化，不是三个独立来源 Policy。随机 CNN 使用
固定 seed=37，从未训练；打乱标签使用 seed=73，对训练目标和 eligibility
联合行置换，验证与 heldout 标签保持原样。

所有 probe 输出 12 个 logits：树集合 4、朝向 4、接近集合 4。目标为三个任务
等权平均；集合任务每位正负类平衡，朝向任务按类别平衡，权重只来自 train。
Adam 学习率 0.01，候选 weight decay 为 `[0.0001, 0.01]`，每个候选 400 次
全批次更新，每 25 次检查 validation。选择三个任务平衡准确率的均值，平局
优先先出现的候选和 epoch；每个 probe 共 800 次更新。

树与接近指标是四位二分类 balanced accuracy 的均值，朝向指标是四类 recall
均值。全阴性集合的基线为 50%，固定朝向基线为 25%，三任务均值基线为 41.7%。
另报告精确集合准确率、非空接近集合的 greedy 有效率和分项组合结果。
分层缺少正负类或某些朝向时显式报告支持缺口，不能横向比较支持不同的宏均值。

局部 RGB 的输入维度和参数量更大，并提供固定玩家中心的空间归纳偏置。
因此它优于最终 CNN 不能单独证明 PPO 丢失信息或某一种架构有因果优势；
同维随机 CNN、两种 probe 容量和打乱标签对照用于限制解释范围。

## 留出结果

主运行完成 180 集、26561 个源轨迹步。选中 train/validation/heldout 各
1080/411/535 个 RGB 状态，其中精确不同 RGB 为 978/382/480 个。
类别配额与步数间隔不按终局筛选，跨 split 去掉 28 次精确 RGB 重合，
去掉 6 次旧监督 RGB 重合和 6 次 episode 内重合。三个 split 的朝向四类、
树集合和近距离接近集合各方向的正负类均有支持。

下表为三个读出初始化 seed 的均值，指标是分项 balanced accuracy，不是
wood 成功率。所有条件使用相同 RGB 与标签，所有 probe 在留出采集前锁定。

| 输入 | probe | 相邻树方向 | 玩家朝向 | 距离 2 接近方向 |
| --- | --- | ---: | ---: | ---: |
| 已训练 CNN | linear | 63.9% | 60.3% | 61.4% |
| 已训练 CNN | MLP64 | 62.5% | 57.2% | 57.2% |
| 同结构随机 CNN | linear | 88.4% | 77.2% | 71.3% |
| 同结构随机 CNN | MLP64 | 82.6% | 77.8% | 72.3% |
| 固定局部 RGB | linear | 99.8% | 99.1% | 96.3% |
| 固定局部 RGB | MLP64 | 97.1% | 93.6% | 87.9% |
| CNN + 打乱训练标签 | linear | 48.5% | 22.6% | 50.2% |
| CNN + 打乱训练标签 | MLP64 | 51.0% | 24.7% | 47.7% |

已训练和随机 CNN 的 linear/MLP64 参数量相同，分别为 1548/9036；局部 RGB
分别为 44112/236044，容量差异必须保留。MLP64 在此预算和验证选择下没有
解决已训练 CNN 的分项读出缺口。不能从这个比较断言更大网络永远无效。

留出集有 150 个近距离接近状态。535 个状态中，261 个来自最终死亡 episode，
272 个是在原轨迹中局部动作无效的状态。158 个状态的局部几何签名没有在
选中训练 RGB 中出现；签名只包含距离<=2 的未阻挡树、朝向、相邻材质/实体
与睡眠覆盖，几何相同不意味着 RGB 或完整环境相同。

下表使用 linear、三个初始化的均值；这些组的分项正负类与四种朝向均有支持。

| 留出分组 | 状态数 | 输入 | 树方向 | 朝向 | 接近方向 |
| --- | ---: | --- | ---: | ---: | ---: |
| 最终死亡 episode | 261 | 已训练 CNN | 61.8% | 62.4% | 55.9% |
| 最终死亡 episode | 261 | 局部 RGB | 99.8% | 99.1% | 95.2% |
| 训练未见几何 | 158 | 已训练 CNN | 59.9% | 65.8% | 56.1% |
| 训练未见几何 | 158 | 局部 RGB | 99.3% | 96.6% | 91.5% |
| 轨迹局部动作无效 | 272 | 已训练 CNN | 60.2% | 64.6% | 59.5% |
| 轨迹局部动作无效 | 272 | 局部 RGB | 99.9% | 99.7% | 96.8% |

自然失败状态的差距仍存在；“RGB 没有足够信息”不能解释这些近距离分项。
同维随机 CNN 比已训练 CNN 更易读出，支持优先检查已训练最终表示的局部
空间可读性。现有最终表示也明显高于打乱标签对照，并非完全没有信息。
probe 有限、预算固定，不能据此证明信息不可逆丢失，也不能确定是哪一次 PPO
更新、哪层压缩或哪个奖励目标导致差距。

留出暗光组仅 31 个状态、7 个环境 seed。linear 的已训练 CNN 朝向/接近
读出为 27.3%/44.4%，局部 RGB 为 95.0%/87.5%。小样本分层只用于提示问题；
主要证据仍来自整个留出集和死亡/未见几何分组。

## 原行为策略与边界

留出 16 集中，原 actor、普通 head、蒸馏 head 分别成功 8/16、5/16、6/16，
死亡 7/9/10 集，外部截断 1/2/0 集，实际交互步数为 2325/2705/2334。
这些是冻结行为策略的采集结果，probe 从未参与执行；本轮没有提高策略成功率。
60 个环境 seed 上所有 180 集的成功、死亡、外部截断均保留在原始统计中。

## 验证状态与下一项

新增 13 项针对性测试已通过，全量回归为 `404 passed`。跨进程完整重复通过：
RGB/标签数据、状态记录、三个输入特征、所有预测、24 个 probe 的参数与
标准化、180 集自然轨迹的规范化 digest 全部一致。重复同一数据不增加独立样本。

独立分析器逐步回放全部 180 集、26561 个源轨迹步，重建选中 RGB、记录和
数据 split；从原生材质/实体、可见树偏移与通行条件独立计算几何标签。
实际采样动作完全一致，记录动作概率最大绝对误差为 0。所有特征、预测、
验证选模、训练标准化、标签打乱及留出分层指标重算通过。9 组轨迹各自首集的
有/无观察器对照也保持一致。源 Policy/PPO optimizer、随机 encoder、
checkpoint、封存源码和 Crafter 安装包未变。结束时两张 GPU 空闲。

26561 是被复核源轨迹的步数；采集固定动作回放、观察器对照、跨进程重复与
独立回放各自还有模拟器执行，不把这个数字写成整个诊断的总交互数或 PPO
训练预算。本轮来源 Policy 的新增训练步数是 0；24 个独立 probe 共执行
19200 次离线优化更新。

下一项在同一自然数据上定位 CNN 中间空间特征到最终 128 维表示的读出变化，
同时控制输出维度与 probe 参数量，检查是否能保留或恢复局部空间信息。根据
这一步再设计只输入 RGB 的自主动作链对照，保留原有 do 能力，并重新做新种子
完整任务评价。离线高可读性不能代替自主执行；奖励/信用分配和共享生存仍是
后续独立问题，当前没有开始正式训练或 Module 注册。

## 产物与复现

- [锁定配置](../../../configs/crafter_wood3_natural_spatial_probe_cuda_pilot_v1.yaml)
- [数据与拟合运行器](../../../experiments/run_crafter_wood3_natural_spatial_probe.py)
- [独立分析器](../../../experiments/analyze_crafter_wood3_natural_spatial_probe.py)
- [probe 定义](../../../experiments/crafter_natural_spatial_probe.py)
- [分项读出结果](../../../results/crafter_wood3_natural_spatial_probe_cuda_pilot_20261014_v1/analysis_verified/probe_metrics.csv)
- [留出分层](../../../results/crafter_wood3_natural_spatial_probe_cuda_pilot_20261014_v1/analysis_verified/heldout_strata.csv)
- [原行为策略任务终局](../../../results/crafter_wood3_natural_spatial_probe_cuda_pilot_20261014_v1/analysis_verified/native_outcomes.csv)
- [独立审计](../../../results/crafter_wood3_natural_spatial_probe_cuda_pilot_20261014_v1/analysis_verified/verification.json)

已有输入在实验机器上，使用全新 output 路径复现；重复同一协议不是新增独立
实验。完整 RGB、特征、probe artifact 和逐步轨迹留在机器上，GitHub 只归档
代码、协议、测试、中文报告及紧凑结果。

```bash
PYTHONPATH=. PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
/home/zl202621/.conda/envs/kecrl/bin/python -u \
  -m experiments.run_crafter_wood3_natural_spatial_probe \
  --config configs/crafter_wood3_natural_spatial_probe_cuda_pilot_v1.yaml \
  --output results/crafter_wood3_natural_spatial_probe_cuda_REPLAY

PYTHONPATH=. PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
/home/zl202621/.conda/envs/kecrl/bin/python -u \
  -m experiments.analyze_crafter_wood3_natural_spatial_probe \
  --input results/crafter_wood3_natural_spatial_probe_cuda_REPLAY \
  --output results/crafter_wood3_natural_spatial_probe_cuda_REPLAY/analysis_verified
```
