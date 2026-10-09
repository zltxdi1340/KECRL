# Crafter 冻结 CNN 的 actor-only 局部可拟合性诊断（2026-10-09）

本轮已完成。冻结三个已有 100k CNN-only checkpoint 的 encoder，复制并监督拟合
原有 `Linear(128,17)` actor，保留原动作 mask。新的保留背景上，有树五类局部
动作 balanced accuracy 从 **20.0% 升至 74.9%**，需转向时的目标方向动作率从
**16.7% 升至 91.0%**；但已对准时 greedy `do` 率仅 **10.4%**，有树八步
greedy 采集率仍约 **8%**。监督可以改善方向映射，尚未形成稳定的采集行为。

这不能证明原线性 head 存在必然的结构限制，也不能唯一归因到 PPO 信用分配。
三个 seed 均选择预算末端的模型，原始特征尺度、优化收敛、联合动作表达和背景
泛化仍未拆开。此前进度更新中的“结构性冲突”表述过强；这里按已完成证据收紧。

本轮是明确的 **teacher-assisted 诊断副本干预**：构造标签进入副本 actor 的监督
损失，范围为 `supervised_actor_copies_on_constructed_local_fixtures_only`。原 Policy、
encoder、critic、原 PPO optimizer 和 checkpoint 文件不变；监督副本未替换
teacher-free baseline。PPO 训练交互 **0 步**，没有正式训练、Module 注册或
Knowledge/SPT 更新。

## 固定协议

- 使用基线 seed0/1/2 的 100k checkpoint；不使用先前表现更差的辅助损失 arm。
- 仅拟合实际 128→17 线性 actor 的副本；输入是原始冻结 CNN embedding，
  不做标准化、不加隐藏层，允许动作仍为 0..6。
- 动作真值为 noop、四方向移动和 do；没有相邻树时 noop 只是 fixture 约定，
  不代表自然任务中的全局最优动作。sleep（动作6）仍是合法输出，但没有正标签；
  只有 masked-out 的动作7..16参数行必须完全不变。
- 每个 head 从原 actor 初始化，Adam 学习率候选 `[0.003,0.03]`，weight decay=0，
  每个候选 3000 次 full-batch 更新；交叉熵类别权重只按训练集频率计算。
- 每100次更新评估验证集，以六类 balanced accuracy 选模型，平分保留先出现者。
  拟合函数不接受保留集输入；不按保留表现重新调参或延长预算。
- 选择后用原生 Player/转向/采集逻辑进行八步 rollout；每场景5次随机采样、
  1次 greedy，原始 actor 与监督副本使用相同环境和动作 RNG manifest。
- 只把采到起始指定的相邻树计为成功；其他树的木材增量单列，避免将游走采集
  错当成指定目标的转向执行能力。oracle 测量发生在 RGB 策略选动作之后。

背景来自 native worldgen，清空中心3×3为草地，固定 daylight=1、初始资源9、
清醒，并移除其他实体；环境安装包未修改，临时稳定排序 wrapper 版本保持
`balance-object-insertion-order-v1`。更远的天然材料和树保留，走出中心后可遇到
其他树或危险地形，因此运行器记录这些结果，不把“无相邻树”当作全世界无树。

候选背景 seed 从41000000开始，检查91个候选，排除37个与上一轮相同的可见
背景和6个本轮内部重复，得到48个全新可见背景。去重只看背景、先于标签和
表现评估；接受顺序决定 split。训练24背景/1440图、验证8背景/480图、保留
16背景/960图，共2880图。每背景覆盖 wood0/1/2、树四方向、朝向四方向的
48个有树场景，以及12个无相邻树场景。全部变体按背景分组，背景和完整RGB
跨 split 重复审计通过。

## 保留背景的一步动作与八步执行

以下均值以三个 source PPO seed 为单位；重复动作与背景不是额外独立 Policy seed。

| 一步动作指标 | 原始 actor | 监督 actor 副本 |
| --- | ---: | ---: |
| 六类 fixture balanced accuracy | 16.7% | 76.0% |
| 有树五类 balanced accuracy | 20.0% | 74.9% |
| 未对准时正确方向动作率 | 16.7% | 91.0% |
| 未对准时 p(目标方向条件于移动) | 25.0% | 82.6% |
| 已对准时 greedy do 率 | 33.3% | 10.4% |

| 八步指定树采集率 | 原始 actor | 监督 actor 副本 |
| --- | ---: | ---: |
| 随机采样：有树全部 | 27.7% | 49.4% |
| 随机采样：已对准 | 53.8% | 56.5% |
| 随机采样：需转向 | 19.0% | 47.1% |
| Greedy：有树全部 | 8.4% | 8.2% |
| Greedy：已对准 | 33.3% | 10.4% |
| Greedy：需转向 | 0.1% | 7.5% |

greedy 的总体采集率没有改善：方向动作改善，却没能稳定切换到 do。原策略的
“有树随机采样”有571次其他树木材增量，监督副本为1次；“无相邻树”场景原
策略仍有7.0%的轨迹采到更远的树。因此必须保留指定目标和任意木材增量的区分。

| Seed | 验证六类 BA | 保留有树五类 BA | 保留正确转向率 | 保留 do 召回 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 83.45% | 76.11% | 91.49% | 14.58% |
| 1 | 83.33% | 78.16% | 96.53% | 4.69% |
| 2 | 74.71% | 70.31% | 84.90% | 11.98% |

三个 seed 均选择 lr=0.03、epoch=3000。训练集 do 召回也很低（14.6%、1.7%、
16.7%），问题并非只出现在保留背景。合法 actor 权重L2范数从约1.60–1.85
增至191–248；训练特征各维标准差中位数为0.0953、0.0252、0.0438。这些是
原始特征拟合难度的线索，尚不是尺度问题的因果证明；预算末端仍是选中模型，
不能声称优化已充分收敛，也不能把低 do 召回解释成已证明的线性不可分。

各背景结果保存在 CSV 和热图中。三个监督副本在保留背景32上的有树一步BA
均为80%，greedy 八步采集率却均为0%；总体分类分数不能替代执行链检查。
所有主诊断短轨迹均无死亡或资源伤害，但它们只覆盖无敌人、白天、短窗口条件，
不能推广到自然世界生存、夜间或长期导航。wood≥3 资格、stone/coal/pickaxe
均未由本轮重新验证，候选0.8门槛仍待验证。

## 审计、测试与计算量

全量测试 **217 passed**，新增10项测试覆盖实际head/optimizer隔离、六类标签、
分组去重和旧背景排除、原生转向采集、指定树与其他树区分、岩浆死亡记录及
协议边界。运行与分析脚本编译通过。

独立Python进程重复全部数据生成、seed0拟合和原始/监督副本的全部保留rollout。
数据/记录digest、冻结特征、actor状态与验证选择/拟合历史、一步指标以及完整
轨迹完全一致。observer开启/关闭的配对轨迹一致。重新加载三个actor artifact
后混淆矩阵一致，CPU概率指标在数值容差内与CUDA结果一致。

原checkpoint、配置、此前数据及已安装Crafter规则共29个路径哈希保持不变；
encoder、critic、原PPO optimizer和动作7..16参数行完全不变。副本评估不更新
任何参数，梯度均为空。actor artifact明确标记 `resumable_ppo_checkpoint=false`，
用于诊断而非直接恢复PPO训练。

| 计算量 | 主实验 | 独立进程重复 | 合计 |
| --- | ---: | ---: | ---: |
| 离线 actor optimizer 更新 | 18000 | 6000 | 24000 |
| 诊断短轨迹 | 34560 | 11520 | 46080 |
| 短轨迹交互步 | 231692 | 73443 | 305135 |
| Observer额外控制步 | 32 | 10 | 42 |
| 原生fixture校验步 | 96 | 96 | 192 |
| PPO训练交互步 | 0 | 0 | 0 |

本轮有真实的监督Policy副本更新，不能写成“全部Policy更新0”；原始Policy与
正式/PPO训练保持不变。全部进程已退出，本轮GPU1空闲。代码和文档为工作区
修改，尚未提交。

## 下一项

先区分**优化条件与联合动作表达**，暂不直接启动自然任务训练：保留冻结encoder
和同样的标签协议，比较原始线性head、训练集标准化后再折叠回原线性head的
拟合，以及小型非线性head；拟合预算和选择条件预先固定，按类报告do/转向
召回与损失收敛，并使用未用于调参的新保留背景验证八步greedy行为。

标准化后折叠仍可表达为同一个128→17仿射head，有助于检验优化尺度；额外隐藏
层则改变表达能力，需要单列。因子化动作头可作为进一步候选，但若其合成使用
已知规则，应明确记录规则/teacher范围，不能将oracle条件直接输入策略而称为
RGB学习。局部贪心执行稳定后，再检查自然背景迁移、探索与生存退化，决定是否
开展后续PPO对照。

## 产物与复现

- 配置：[crafter_wood3_actor_only_local_cuda_v1.yaml](../configs/crafter_wood3_actor_only_local_cuda_v1.yaml)
- 拟合：[crafter_actor_only_local.py](../experiments/crafter_actor_only_local.py)
- 运行：[run_crafter_wood3_actor_only_local.py](../experiments/run_crafter_wood3_actor_only_local.py)
- 分析：[analyze_crafter_wood3_actor_only_local.py](../experiments/analyze_crafter_wood3_actor_only_local.py)
- 重载核对：[verify_crafter_wood3_actor_only_local.py](../experiments/verify_crafter_wood3_actor_only_local.py)
- 测试：[test_crafter_actor_only_local.py](../tests/test_crafter_actor_only_local.py)
- 主结果：[summary.json](../results/crafter_wood3_actor_only_local_cuda_20261009_v1/summary.json)
- 汇总：[aggregate.json](../results/crafter_wood3_actor_only_local_cuda_20261009_v1/analysis_final/aggregate.json)
- 分析核对：[verification.json](../results/crafter_wood3_actor_only_local_cuda_20261009_v1/analysis_final/verification.json)
- 副本核对：[actor_artifact_verification.json](../results/crafter_wood3_actor_only_local_cuda_20261009_v1/actor_artifact_verification.json)

![一步动作与执行](../results/crafter_wood3_actor_only_local_cuda_20261009_v1/analysis_final/actor_and_trajectory_comparison.png)

![拟合曲线](../results/crafter_wood3_actor_only_local_cuda_20261009_v1/analysis_final/actor_fit_curves.png)

![各背景的动作与执行](../results/crafter_wood3_actor_only_local_cuda_20261009_v1/analysis_final/background_comparison.png)

```bash
PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 \
/home/zl202621/.conda/envs/kecrl/bin/python -u \
  -m experiments.run_crafter_wood3_actor_only_local \
  --config configs/crafter_wood3_actor_only_local_cuda_v1.yaml \
  --output results/crafter_wood3_actor_only_local_cuda_NEW

/opt/anaconda3/bin/python -m experiments.analyze_crafter_wood3_actor_only_local \
  --input results/crafter_wood3_actor_only_local_cuda_NEW \
  --output results/crafter_wood3_actor_only_local_cuda_NEW/analysis_final

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/zl202621/.conda/envs/kecrl/bin/python -m experiments.verify_crafter_wood3_actor_only_local \
  --input results/crafter_wood3_actor_only_local_cuda_NEW \
  --output results/crafter_wood3_actor_only_local_cuda_NEW/actor_artifact_verification.json
```
