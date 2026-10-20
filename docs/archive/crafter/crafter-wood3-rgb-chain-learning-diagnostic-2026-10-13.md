# Crafter wood>=3 RGB 动作链学习 pilot（2026-10-13）

本轮在[动作链只读审计](crafter-wood3-rgb-chain-audit-2026-10-13.md)后，对一个具体
假设做单 policy seed 学习 pilot：固定 CNN，只改变线性 head 训练状态来源，把
oracle 连续采集路径作为监督数据是否能让 RGB executor 自己完成 wood>=3。

结果为原 actor `12/30 = 40.0%`、原策略状态 head `3/30 = 10.0%`、采集链状态 head
`4/30 = 13.3%`。本次采集链监督没有解决自主动作链，两个拟合副本都出现自然迁移
退化。原 PPO checkpoint 保持不变。这个单 seed pilot 不给出跨 seed 因果结论，
也不是正式 PPO 训练、完整 Module 资格或 teacher-free 学习结果。

## 固定协议

- 原策略来自 seed 0 的既有 `100000` 交互步 CNN-only checkpoint。冻结 encoder、
  critic、原 actor 和 PPO optimizer。仅监督拟合两个相同 Linear(128,17) 的副本，
  训练集标准化后折叠回原特征坐标，mask 外动作行不变。
- 比较 `native_state_fit` 与 `chain_state_fit`：前者由原 RGB actor 自然访问状态，
  后者仍由原 actor 采样，但在数据采集时用 approach/turn/ready-do oracle 连续
  执行。两种来源使用相同环境/action episode seed；按完整 episode 隔离
  train/validation/heldout，每来源分别为 32/8/8 集，共 96 集。
- 每集每类最多取前 6 个不重复 RGB 状态，不按采样是否正确、任务成功或存活筛选，
  不做间隔抽样。监督标签来自局部几何的所有有效动作集合，没有把几何信息输入网络。
  跨 split 精确 RGB 重合被去掉，同一 source 的精确重复也被去掉。
- 对每类使用两来源可用数量与 96 的最小值。本次每个 head 实际训练
  `ready=90 / turn=30 / approach=96`，各 216 状态，初始化及训练损失相同。
  每臂试 `.003/.03` 两个 learning rate、各 1500 更新，共 3000 更新；两臂共 6000。
  共享验证集选择 epoch/LR，选后封存 head；保留集和 fresh episode 不参与选择。
- 收集环境/action seed 为 `181000000/191000000`，验证加 `1000000`，保留加
  `2000000`；fresh 为 `184000000/194000000`，各集加 1，与既有诊断 seed 段分离。
  fresh 三个 executor 各 30 集，共 90 集，同 seed 初始 RGB/snapshot 配对。
- 自然 reset、initial wood=0、目标 wood>=3、每集最多 256 步、environment length
  10000、动作 0..6、categorical sample，使用稳定 wrapper 和真实 CUDA。
- fresh 执行器只收 RGB，没有 ready/turn/approach 动作替换，没有生存干预。snapshot
  只供观察与复核。三份策略各用第一集验证 observer 开关不改变完整 RGB 动作轨迹。

匹配的是**各类别选定训练状态数及优化更新数**。数据采集 episode 数相同，但实际
交互步数、类别的来源 episode 数、状态内容和几何比例不同；不声称训练数据交互预算
相同，也不把这个两臂比较解释为只隔离了“连续性”一个因子。

## 数据采集与模型选择

| split | 来源 | episode | 成功 | 死亡 | 外部截断 | 交互步数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| train | native | 32 | 16 | 16 | 0 | 3854 |
| train | oracle chain | 32 | 31 | 1 | 0 | 887 |
| validation | native | 8 | 4 | 4 | 0 | 1083 |
| validation | oracle chain | 8 | 7 | 1 | 0 | 349 |
| heldout | native | 8 | 3 | 4 | 1 | 1036 |
| heldout | oracle chain | 8 | 8 | 0 | 0 | 89 |

采集共 7298 个交互事件，进入监督数据集的是 1067 个局部状态：train 699、validation
193、heldout 175。采集的成功率含 oracle，不能作为自主 executor 结果。

两臂实际训练的 turn 各只有 30 个状态，分别来自 23/19 个 train episode。各类数量
匹配并没有消除有限覆盖问题；oracle 更快完成任务，反而减少反复出现的转向状态。
两臂选择到的 epoch/LR 分别为 native `1400/.03`、chain `250/.03`。epoch 0
纳入每条验证曲线，保留集和 fresh 不做二次选择。

## 离线局部读出

下表为正确动作集合内的贪心动作准确率（不是 categorical 采样成功率）。保留集共
175 状态：ready 51、turn 30、approach 94，来自此前未参与训练或选择的 episode。

| head | train ready | train turn | train approach | heldout ready | heldout turn | heldout approach |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原 actor | 92.9% | 0.0% | 3.5% | 88.2% | 0.0% | 1.1% |
| native-state head | 95.6% | 100.0% | 100.0% | 45.1% | 30.0% | 60.6% |
| chain-state head | 71.1% | 100.0% | 78.1% | 60.8% | 6.7% | 50.0% |

原 actor 的 train 列使用两来源全部 train 状态；两个拟合 head 的 train 列各使用
自身实际拟合的 216 状态。因此 train 列用于看各自拟合/泛化差距，不作三臂同状态
精确比较；validation/heldout 列则在同一组状态上比较。

保留集 category macro accuracy 为原 actor 29.8%、native 45.2%、chain 39.2%。
局部 turn/approach 有改善，但训练到保留集差距大，原 actor 已具备的 ready/do
能力退化。有限数据上存在可拟合信息，不代表 frozen RGB 表示已足够泛化到自然动作链，
也不能证明 head 容量或 PPO 奖励是唯一原因。

## 关闭全部动作 oracle 的完整任务

| executor | 成功 | 死亡 | 外部截断 | wood1/2/3 达到集数 | 交互步数 |
| --- | ---: | ---: | ---: | --- | ---: |
| 原 actor | 12/30 = 40.0% | 18 | 0 | 23 / 19 / 12 | 4132 |
| native-state head | 3/30 = 10.0% | 27 | 0 | 12 / 4 / 3 | 5002 |
| chain-state head | 4/30 = 13.3% | 23 | 3 | 15 / 7 / 4 | 4616 |

全部死亡及外部截断在分母中，总计 90 集、13750 个交互事件。两拟合臂配对为都成功
1 集、native 独有 2 集、chain 独有 3 集、都失败 24 集。相对原 actor，native 有
11 个原 actor 独有成功、2 个自身独有成功；chain 对应为 10/2。

在各自自然轨迹上观察到的动作有效采样率：

| executor | ready/do | turn | approach |
| --- | ---: | ---: | ---: |
| 原 actor | 54/82 = 65.9% | 21/268 = 7.8% | 455/2588 = 17.6% |
| native-state head | 19/780 = 2.4% | 10/917 = 1.1% | 285/2174 = 13.1% |
| chain-state head | 26/144 = 18.1% | 11/1036 = 1.1% | 232/3171 = 7.3% |

这些分母是不同策略自身访问的相关状态，不能当作同状态因果比较，也不能当作独立
资格 episode。但它们明确显示本轮副本没有形成稳定自主动作链。native 的 3086 次
移动中 2298 次阻塞；chain 的 3318 次 do 中 3292 次没有采到 wood，常见场景为
没有相邻树或朝向错误。离线 readout 指标不能替代实际闭环执行。

详细生存研究尚未开始：保留全部伤害日志，不筛掉失败。三臂终止伤害提示分别为
原 actor hostile/projectile 10、depletion 2、lava 1、mixed/unknown 5；native 为
18/2/1/6；chain 为 17/2/0/4。提示只是归因线索。本安装环境不存在摔落/溺水，
所以不存在可先行剔除的这两类统计。更差的动作链会改变暴露时间与状态分布，不能
仅凭死亡增加判定生存模块的独立贡献。

## 下一项诊断

本轮不扩大到三 seed 重复相同数据方案。先补齐本 pilot 自主失败状态的读出定位：
把三个 executor 访问的 ready/turn/approach 状态交叉输入原 head 与两个副本，
保持同一 RGB、同一有效动作标签，拆开“状态变难”和“head 本身变差”。记录首次局部
失误以及是否在新状态持续无效 do/阻塞移动，先做只读诊断，不重新选 head。

随后根据结果做一个最小学习干预。若主要是自身后续状态覆盖不足，可在**训练段**
迭代采集 learner 状态并标注，单独保护原 ready/do 能力，使用新的验证、保留及
fresh seed 段确认；本次已经查看的 heldout/fresh 不可作为下一次独立最终测试。
若各 head 在相同自然状态上都读不出，则转向 RGB 表示/空间监督诊断。
奖励与信用分配的作用仍未独立识别，teacher 监督可学性不能证明 PPO 也能学好。

局部自主链改善后，即使总成功率未达 0.8，只要剩余失败明确主要由死亡阻断，就
进入跨任务共享生存诊断。wood 是首个 SPI executor 可行性实例；后续 mining、
工具前置条件和 Module 复用仍需各自验证，不能复用 wood 的成功率作资格结论。

## 验证与复现

全量回归 `380 passed`，包括动作 oracle 隔离、split seed/RGB 泄漏、匹配类别预算、
样本选择与失败死亡分母测试。主进程和独立进程各完整执行一次：数据、特征、训练行、
保存 head canonical digest、采集与自然任务 trace digest 必须完全一致；重复不增加
独立训练 seed 数。

独立分析器重放全部采集状态，重新恢复数据/标签与 frozen CNN 特征，重算训练、验证
和保留指标，验证训练集 standardizer 与 head 锁定哈希，再回放全部自然 episode
的 RGB 概率和动作采样。原策略、optimizer、安装 Crafter 包和封存文件保持不变。
这项复核不新增监督更新；拟合确定性由跨进程完整重跑检查。

```bash
PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/zl202621/.conda/envs/kecrl/bin/python -u \
  -m experiments.run_crafter_wood3_rgb_chain_learning \
  --config configs/crafter_wood3_rgb_chain_learning_cuda_pilot_v1.yaml \
  --output results/crafter_wood3_rgb_chain_learning_cuda_pilot_20261013_v1

PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/zl202621/.conda/envs/kecrl/bin/python -u \
  -m experiments.analyze_crafter_wood3_rgb_chain_learning \
  --input results/crafter_wood3_rgb_chain_learning_cuda_pilot_20261013_v1 \
  --output results/crafter_wood3_rgb_chain_learning_cuda_pilot_20261013_v1/analysis_verified
```

runner 拒绝覆盖已有输出。GitHub 归档配置、源码、测试、本文、主摘要及独立分析；
完整 JSONL、RGB 数据、特征、PT 与 source snapshot 保留在实验机器，不随源码上传。

- [pilot 配置](../../../configs/crafter_wood3_rgb_chain_learning_cuda_pilot_v1.yaml)
- [runner](../../../experiments/run_crafter_wood3_rgb_chain_learning.py)
- [独立分析器](../../../experiments/analyze_crafter_wood3_rgb_chain_learning.py)
- [主摘要](../../../results/crafter_wood3_rgb_chain_learning_cuda_pilot_20261013_v1/summary.json)
- [独立验证](../../../results/crafter_wood3_rgb_chain_learning_cuda_pilot_20261013_v1/analysis_verified/verification.json)
- [自然任务结果](../../../results/crafter_wood3_rgb_chain_learning_cuda_pilot_20261013_v1/analysis_verified/outcomes.csv)
