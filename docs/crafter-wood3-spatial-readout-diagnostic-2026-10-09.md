# Crafter 冻结 CNN 的树方位与朝向联合可读性诊断（2026-10-09）

本轮已完成。100k 基线的冻结表示在保留背景上的树方位/朝向线性 readout 平衡
准确率为 **95.2%/95.6%**；MLP64 对“无树/采集/四方向转向”的联合判断为 **93.4%**。
只统计有树场景、以五个局部动作类别的相同口径比较，原 actor 为 **19.9%**，
独立线性 readout 为 **85.5%**，MLP64 为 **95.3%**。相关信息整体可读，但当前
actor 没有稳定把它用于局部动作选择；“encoder 完全没有树方位或朝向信息”
不符合本轮证据。

仍有两个重要限制：少数保留背景的 readout 明显失败；随机 CNN 的 readout 也很强。
因此不能说 encoder 对所有背景都足够可靠，也不能把高可读性归功于 PPO 表示学习。
readout 用的是明确的构造标签、训练集归一化和独立监督损失，不能据此宣布 PPO
的失败已被唯一归因到 actor、奖励或信用分配。

本轮 Policy/encoder/critic/原 optimizer 全部冻结，Policy 训练交互 **0 步**。
构造 oracle 标签只用于独立离线 readout；`policy_teacher_used=false`、
`readout_oracle_supervision_used=true`，范围为 `offline_diagnostic_readouts_only`。
没有更新 Knowledge/SPT、注册 Module 或启动正式训练；所有产物均为非正式诊断。
离线模型从未接管 Policy 或用于环境动作。

## 数据和固定协议

- 使用原生 Crafter worldgen 生成材料背景，将中心 3x3 清为草地，移除其他实体，
  固定 daylight=1、全资源 9、清醒。在中心放一棵四邻接树或不放树。
  原生 Player 更新、转向、采集和 RGB 渲染保持不变；安装包未修改。
- 每背景覆盖 wood0/1/2、树四方向、玩家四朝向，共 48 个有树场景；无树场景
  每 wood/朝向一份，共 12 个。去除假定树方向造成的无树重复，合计 **60 个场景**。
- 场景标签依次为 tree direction（无树/左/右/上/下，5 类）、facing（4 类）、
  joint state（无树/可采/需转向，3 类）、local decision（无树/采集/四方向转向，6 类）。
  “无树”只表示当前四邻接没有目标树，不是整个视窗或世界没有树，也不是自然任务
  下唯一合理动作。背景中仍可见更远的树或其他材料。
- 候选 seed 从 31000000 开始，最多 512 个；仅按可见背景图像去重，接受顺序决定
  split。实际检查 71 个候选，排除 23 个重复，得到 **48 个不同可见背景**。
  去重在拟合和策略表现检查前进行，配置校验排除先前同类 seed 角色。
- 24 个训练背景/1440 张图，8 个验证背景/480 张图，16 个保留背景/960 张图；
  总计 **2880 张图**。同背景所有 wood、朝向、树存在与方位变体属于同一 split。
  检查可见背景重复、完整 RGB 跨 split 重复、同背景变体跨 split 均通过。
- 树方向与 facing 每类数量相同；joint state 为 1:1:3，local decision 为
  4:4:3:3:3:3。因此采用各类别召回率平均的 balanced accuracy，不以多数类预测
  或总 accuracy 掩盖联合判断失败。
- 两 arm（基线与已有 wood-gain 辅助系数 0.1）、三个 Policy seed `[0,1,2]`、
  25k/100k checkpoint，共 12 份冻结 128 维表示。所有模型使用相同场景 manifest。
- 固定 GPU 1 上的 RTX 4090、真实 CUDA、Torch/cuDNN 确定性、稳定排序 wrapper
  `balance-object-insertion-order-v1`；源码、已安装 Crafter 规则及 checkpoint 哈希封存。

首次尝试发现不同 seed 生成相同可见背景，分组审计在拟合任何 readout 前拒绝了数据。
该尝试保留在 `results/crafter_wood3_spatial_readout_cuda_20261009_v1/aborted.json`；
完成版位于 `..._v2`。增加的去重只看背景，不利用验证/保留表现挑选样本。

## Readout 和对照

独立 readout 为线性层或 `Linear(input,64) → Tanh → Linear(64,18)`，18 个输出
拆为四个 target 的 logits。归一化均值/标准差只由训练图计算，std 下限 1e-4。
四项类别加权交叉熵等权平均；Adam lr=0.01，每项候选训练 400 个全 batch epoch。
weight decay 固定候选 `[1e-4,1e-2]`，每 25 epoch 检查验证集，以四 target 平衡
准确率均值选择 checkpoint；分数相同时保留先出现的候选。没有保留集驱动的
超参数调整或模型选择。每个 feature/architecture 使用 readout seed0/1/2。

另外用已拟合的 tree/facing 两项预测按原生规则组合出 joint state/local decision。
组合路径明确使用已知诊断规则，不是策略学出的规划器；它避免把直接线性联合
分类偏弱误读成“两个方向的信息都没有”。线性/MLP 两种结果均报告，不按保留集
分数选择其中一种作为成功模型。

对照包括：

- 固定玩家位置的 3x3 地形 RGB crop（21x21x3=1323 维），不读取目标位置标签选 crop。
- 同一结构、三个固定初始化 seed 的随机 CNN（128 维，完全不训练 encoder）。
- 基线 seed0/100k 表示上的训练标签打乱；四列标签分别置换，保留每列类别计数，
  验证/保留标签不变。它用于检查信息匹配效果，不是置信区间或零假设的完整检验。

共 17 个 feature/control 条件 × 2 架构 × 3 readout 初始化，**102 个独立 readout**。
三种 readout 初始化不是新的 Policy seed；表中的均值保留“三个冻结 Policy seed”
与“三个 readout 初始化”的层级，不把 9 次拟合当作 9 个独立环境实验。

## 保留背景结果

以下是 balanced accuracy；joint 为三类联合状态，decision 为六类局部判断。
组合列使用线性 tree/facing 的预测与已知规则。

| 冻结表示 | 线性树方位 | 线性朝向 | 线性 joint | 线性 decision | 组合 decision | MLP joint | MLP decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 基线 25k | 95.5% | 93.8% | 61.4% | 82.6% | 93.8% | 88.7% | 92.6% |
| 基线 100k | 95.2% | 95.6% | 64.1% | 84.9% | 94.3% | 89.6% | 93.4% |
| 辅助 25k | 97.5% | 97.1% | 64.2% | 92.3% | 96.8% | 94.7% | 95.7% |
| 辅助 100k | 96.5% | 96.2% | 69.4% | 87.5% | 95.2% | 93.8% | 95.1% |
| 随机 CNN | 99.4% | 97.4% | 66.6% | 92.0% | 98.4% | 96.9% | 97.9% |
| 局部 RGB | 100.0% | 100.0% | 73.1% | 100.0% | 100.0% | 100.0% | 100.0% |

打乱标签的线性对照对应为树方位 23.0%、朝向 23.5%、joint 35.7%、decision 19.0%；
MLP 对应为 19.5%、24.9%、30.3%、17.6%。类别平衡随机参考依次为
20%、25%、33.3%、16.7%。有效匹配标签的高分不能由这些打乱对照解释。

随机 CNN 分数很高，说明本组固定视角、模板纹理的人工局部任务对图像映射有
较强可读性，并非必须通过 PPO 才能保存相关信息。本轮没有证明“训练后表示更好”，
也没有测完整自然状态分布或夜间/敌人遮挡条件。

原 actor 与 readout 比较时排除无树标签，以“do、turn-left/right/up/down”五类
相同真值求平衡准确率；预测无树在有树样本上也计错。原 actor 使用 0..6 allowlist，
noop/sleep/错误 do 均按当地构造规则记为动作错误，不改动原策略。

| checkpoint | 基线原 actor | 基线线性 readout | 基线 MLP64 | 辅助原 actor | 辅助 MLP64 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 25k | 19.5% | 82.1% | 93.9% | 20.0% | 95.8% |
| 100k | 19.9% | 85.5% | 95.3% | 19.6% | 95.1% |

这是一步局部决策指标，不是上一轮的八步 wood 增量率，更不是 wood>=3 自然任务
成功率。原 actor 未对准时分给正确树方向的 `p(target | movement)` 在 100k 基线/
辅助分别为 25.04%/25.07%，仍接近均衡四方向固定偏好的 25% 参照。

## 背景差异

高合并分数不能代替各背景的检查。100k MLP decision 六类指标、先平均三个
readout 初始化后，最差保留背景为：

| 冻结 Policy | 背景 ID | 该背景 decision 平衡准确率 |
| --- | ---: | ---: |
| 基线 seed0 | 32 | 34.4% |
| 基线 seed1 | 47 | 59.0% |
| 基线 seed2 | 36 | 89.4% |
| 辅助 seed0 | 36 | 75.5% |
| 辅助 seed1 | 47 | 46.1% |
| 辅助 seed2 | 47 | 87.2% |

16 个保留背景上，三个基线 seed 分别有 12/10/15 个背景该指标达到至少 90%，
辅助为 14/14/12 个。仅按总体平均称 encoder 已可靠处理所有背景不成立。
低分样例与逐背景结果已保存；没有根据这些案例重新拟合、调参或更换保留集合。
分项下降可能涉及背景干扰、readout 外推或表示的稳定性，尚未单独做因果拆分。

## 审计、测试与产物

全量回归 **207 passed**，新增 9 项测试包括去重后的类分布、原生背景中的配对
规则/树移除、原生转向采集、分组与图像泄漏拒绝、平衡指标、训练集归一化、
保留评估不改变拟合状态/选择、actor 与 readout 同口径、数据 digest 和协议边界。
同时将 pytest 默认 `testpaths` 固定为 `tests/`，修复结果目录中归档测试快照被
再次导入导致同名冲突的问题；结果和已封存快照未修改。

12 组 Policy/optimizer 状态完全未改变，全部 Policy gradient 为 None，真实 CUDA
已验证。35 个源码、规则、checkpoint 和配置文件哈希通过；分析另核对数据、
16 份 feature 文件及 102 个 readout artifact 的哈希/digest。

独立 Python 进程重建全部背景数据，并重复基线 seed0/100k 的 feature/actor 输出，
以及 readout seed0 的线性和 MLP64 拟合。图像/标签/记录的 canonical digest、
feature/actor digest、readout 状态/归一化、验证选择、训练/保留指标及逐背景指标
全部完全相同。

主诊断离线 readout 更新共 81600 次，重复进程 1600 次；这些更新仅属于独立
readout，与 Policy 训练步数严格区分。每个背景额外执行两步“turn→do”验证
原生规则，主诊断和重复各 96 步，最初拒绝的尝试 96 步，共 **288 次 fixture
诊断交互**。没有执行 Policy rollout，**Policy 训练交互为 0**。进程已退出，
本轮使用的 GPU 1 空闲。

- 配置：[crafter_wood3_spatial_readout_cuda_v1.yaml](../configs/crafter_wood3_spatial_readout_cuda_v1.yaml)
- 数据：[crafter_spatial_readout_data.py](../experiments/crafter_spatial_readout_data.py)
- Readout：[crafter_spatial_readouts.py](../experiments/crafter_spatial_readouts.py)
- 运行：[run_crafter_wood3_spatial_readout.py](../experiments/run_crafter_wood3_spatial_readout.py)
- 分析：[analyze_crafter_wood3_spatial_readout.py](../experiments/analyze_crafter_wood3_spatial_readout.py)
- 测试：[test_crafter_spatial_readout.py](../tests/test_crafter_spatial_readout.py)
- 主结果：[summary.json](../results/crafter_wood3_spatial_readout_cuda_20261009_v2/summary.json)
- 最终分析：[aggregate.json](../results/crafter_wood3_spatial_readout_cuda_20261009_v2/analysis_final/aggregate.json)、
  [verification.json](../results/crafter_wood3_spatial_readout_cuda_20261009_v2/analysis_final/verification.json)；
  同目录含 readout_metrics/background_metrics/confusions/selections/actor_metrics CSV 及 PNG/SVG 图。
- 最终封存核对：[final_file_verification.json](../results/crafter_wood3_spatial_readout_cuda_20261009_v2/analysis_final/final_file_verification.json)。

![信息可读性与对照](../results/crafter_wood3_spatial_readout_cuda_20261009_v2/analysis_final/spatial_readout_comparison.png)

![100k 的信息与联合判断](../results/crafter_wood3_spatial_readout_cuda_20261009_v2/analysis_final/joint_readout_comparison.png)

![各背景联合判断](../results/crafter_wood3_spatial_readout_cuda_20261009_v2/analysis_final/background_readout_comparison.png)

![低分背景中的配对朝向](../results/crafter_wood3_spatial_readout_cuda_20261009_v2/analysis_final/low_scoring_background_examples.png)

复现命令：

```bash
PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 \
/home/zl202621/.conda/envs/kecrl/bin/python -u \
  -m experiments.run_crafter_wood3_spatial_readout \
  --config configs/crafter_wood3_spatial_readout_cuda_v1.yaml \
  --output results/crafter_wood3_spatial_readout_cuda_NEW

/opt/anaconda3/bin/python -m experiments.analyze_crafter_wood3_spatial_readout \
  --input results/crafter_wood3_spatial_readout_cuda_NEW \
  --output results/crafter_wood3_spatial_readout_cuda_NEW/analysis_final
```

## 下一项

本轮最直接的证据支持先研究**现有表示到 actor 动作选择的拟合缺口**，暂不继续
以更大 CNN、记忆模块或标量 wood-gain 辅助损失作为默认修复。下一项做一个明确
非正式的 actor-only 局部可拟合性 pilot：只在诊断用 checkpoint 副本中冻结 encoder，
以实际 17-action actor、原 allowlist 对局部构造动作进行受限监督拟合；按背景
隔离且使用全新保留背景，再以原生短轨迹检查能否稳定“转向→do”。可用同一背景
协议比较原始 head 与监督后的 head，避免把 readout 的归一化、额外隐藏层和原
actor 学习差异混在一起。该监督必须明确标记 teacher scope，不能称作普通 PPO
性能，也不能直接替换已经保留的无 teacher 基线。

这项干预能检验“现有 head 在固定 encoder 上能否学会局部规则”，仍不足以诊断
完整任务的探索、生存或 PPO 信用分配。若后续需要迁回自然 Crafter，应使用新的
诊断集合检查行为迁移与生存退化，再决定是否开展 PPO 课程或其他训练对照。
本轮没有启动该干预或正式训练，候选 0.8 门槛仍待验证，stone/coal/pickaxe 未重新测试。
代码和文档均为工作区修改，尚未提交。
