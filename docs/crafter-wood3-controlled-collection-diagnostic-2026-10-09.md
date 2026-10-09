# Crafter wood 局部转向与采集能力诊断（2026-10-09）

受控测试已完成。移除敌人、资源压力和长距离搜索后，100k CNN-only 基线在 8 步内
采到一份 wood 的完成率为：已朝向邻接树 56.4%，需要先转向 17.1%。相应的贪心
完成率为 33.3% 和 1.9%。固定动作偏好与随机采样能解释大量局部成功，尚未形成
可靠的“根据树的位置和当前朝向切换转向/采集”行为。死亡不是唯一瓶颈。

本轮训练交互为 0；只评估冻结的基线与已有 `do_wood_gain` 系数 0.1 辅助策略。
所有产物为 `formal_result=false`，没有更新 Knowledge/SPT、注册 Module 或启动
正式训练。这里的成功是构造库存从 wood0/1/2 增加一份，**不是自然 Crafter 的
wood>=3 资格成功率**，也不用于检查候选 0.8 门槛。

## 固定场景与评估协议

- 两个 arm、三个训练 seed `[0,1,2]`、25k/100k checkpoint，共 12 组冻结 Policy。
- 每组均衡覆盖初始 wood0/1/2、树的四个方向、玩家四种朝向，48 个几何条件；
  每个条件配一份移除树的对照，共 96 个场景。
- 全地图草地，玩家在中央，最多一棵邻接树；始终白天、无其他实体，
  health/food/drink/energy 均为 9，清醒。环境 length=10000，评估窗口为 8 步。
- 每场景 20 次随机动作采样和 1 次贪心轨迹。每组 2016 次，主实验共 24192 次
  rollout（23040 次随机、1152 次贪心）、174255 次诊断交互。
- scene 环境 seed 从 21000000、动作 RNG seed 从 25000000 开始，按训练 seed index
  加 1000000；按几何条件和重复序号分配。在有树/无树、arm 和 checkpoint 间配对。
  配置检查排除先前训练、开发、资格和自然诊断中的同类 seed 区间。
- 20 次重复只测动作采样；地图没有新的随机背景。无树场景按假定树方向重复，
  其中一些初始图像相同。不能把这些重复当作独立世界或总体泛化样本。
- 输入始终是原生渲染的 64x64 RGB，动作 allowlist 为 0..6。
  策略选完动作后才读取 oracle 标签；标签不作为输入、奖励或动作干预。
- RTX 4090 GPU 1、真实 CUDA tensor、确定性 Torch/cuDNN、稳定对象排序 wrapper
  `balance-object-insertion-order-v1`，参数及 optimizer 状态保持不变。

独立 `QuietCollectionEnv` 只替换 reset 地图、时间和实体生成。Player.update、
移动、转向、采集、生命资源更新及 RGB 渲染都调用已安装 Crafter 的原生实现；
安装包没有修改。初始 wood 对应的 collect_wood achievement/unlocked 状态也一并
初始化，避免把构造库存产生的 achievement 当作新行为奖励。奖励不参与策略更新。

朝树移动先改变 facing，然后被树阻挡；因此已朝向时最短方案为一步 do，
未对准时为“朝树移动→do”两步。每步检查 wood 增量与原生可采规则一致，并核对
全资源、无睡眠、无敌人及无原生终止条件。

## 完成率与方向偏好

以下均为局部构造场景，每 seed 的场景数相同，合并率也等于三个 seed 的均值。
随机模式的已朝向条件每行 720 次、需转向条件 2160 次；贪心模式分别为 36/108 次。

| 策略 | checkpoint | 随机：已朝向 | 随机：需转向 | 贪心：已朝向 | 贪心：需转向 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 基线 | 25k | 68.1% | 21.4% | 77.8% | 0.0% |
| 基线 | 100k | 56.4% | 17.1% | 33.3% | 1.9% |
| 辅助 | 25k | 60.7% | 19.5% | 100.0% | 0.0% |
| 辅助 | 100k | 62.5% | 20.6% | 97.2% | 0.0% |

无树对照没有 wood 增量；这是环境与标签的负对照，不能单凭此认为策略能识别无树。
100k 随机完成率按训练 seed 如下：

| 策略 | seed | 已朝向 | 需转向 |
| --- | ---: | ---: | ---: |
| 基线 | 0 | 72.1% | 20.8% |
| 基线 | 1 | 51.3% | 19.0% |
| 基线 | 2 | 45.8% | 11.4% |
| 辅助 | 0 | 57.1% | 18.9% |
| 辅助 | 1 | 70.0% | 24.3% |
| 辅助 | 2 | 60.4% | 18.8% |

100k 基线需转向场景的 2160 次轨迹中，547 次曾进入可采状态、369 次采到 wood；
1613 次失败从未进入可采状态，178 次进入后仍未采到。初始位置虽已邻接树，
仍有 1772 次曾离开初始位置。辅助策略相应为 692 次进入、446 次采到，失败分为
1468 次未进入和 246 次进入后未采到。短窗口中两个执行环节都有缺口。

100k 的需转向随机完成率按树方向为：

| 策略 | 左 | 右 | 上 | 下 |
| --- | ---: | ---: | ---: | ---: |
| 基线 | 0.7% | 3.5% | 37.2% | 26.9% |
| 辅助 | 32.6% | 18.7% | 24.8% | 6.5% |

基线 seed0 初始贪心动作都是 do，seed1/2 都是 move_up；这些偏好也出现在对应
无树图像中。辅助 100k 的 108 个未对准初始场景中，105 个贪心动作是 do。
高已朝向完成率与极低需转向完成率并存，说明“碰巧面对树时执行偏好动作”与
“按空间状态选择下一动作”必须分别评估。

## 配对动作概率

保持 wood、树位置不变，仅改变玩家朝向，所有 144 个“四朝向”比较组都得到
四张不同 RGB。100k 基线各 seed 的最大单动作概率变化为 1.47/1.66/2.72 个百分点，
但初始贪心动作在全部 36 个比较组内都不随朝向改变。辅助策略为
0.77/3.36/3.34 个百分点，36 组中只有 3 组改变贪心动作；改变不等于正确切换。

初始 p(do) 的合并均值，基线为已朝向 0.396311、未对准 0.396318；辅助为
0.463618、0.463772。个别场景有变化，但没有一致的“真正可采时提高 do”的选择性。
固定树位置的 aligned-minus-other-facings p(do) 差值在基线中范围
[-0.01134, 0.01513]，辅助中 [-0.01722, 0.02962]，不能把合并近零说成所有像素变化
都被完全忽略。

100k 的有树减去同位置无树对照，已朝向 p(do) 平均增加 1.53/1.80 个百分点
（基线/辅助），未对准也增加几乎相同的 1.53/1.82 个百分点；正确目标方向移动的
绝对概率反而平均变化 -0.42/-0.54 个百分点。这包含 do 与移动的占比分配效应，
不能单凭绝对移动概率下降断言方向更差。

因此另外检查 `p(正确目标方向 | action 属于四个移动动作)`：

| 策略 | checkpoint | 有树、需转向 | 配对无树、同一假定方向 | 差值（百分点） |
| --- | ---: | ---: | ---: | ---: |
| 基线 | 25k | 24.993% | 25.008% | -0.016 |
| 基线 | 100k | 25.045% | 24.900% | +0.144 |
| 辅助 | 25k | 24.992% | 24.996% | -0.005 |
| 辅助 | 100k | 25.075% | 25.011% | +0.064 |

均衡四方向时，固定方向偏好或均匀四方向的参考均为 25%；结果与这一参照很近。
需要区分“总体动作分布变了”和“移动内部更多概率分给正确方向”。本轮看不到
足够大的、稳定的方向选择反应。

## 固定动作分布参考

对每个场景用初始七动作概率，解析计算“之后每步都保持该分布”在 8 步内采到
wood 的概率。分别使用有树初始概率与配对无树初始概率，后者无需观察树。
这是人工场景内的理论参照，不向任何真实轨迹强制动作或替换输入。

| 需转向场景 | 实际随机完成率 | 固定有树初始分布 | 固定无树初始分布 |
| --- | ---: | ---: | ---: |
| 基线 25k | 21.44% | 21.24% | 21.23% |
| 基线 100k | 17.08% | 18.65% | 18.49% |
| 辅助 25k | 19.49% | 20.18% | 20.15% |
| 辅助 100k | 20.65% | 20.70% | 20.41% |

均匀七动作的精确参考为已朝向 28.57%、需转向 10.37%。现有策略超过均匀参考，
但偏好 do、少选 noop 等就可以提高该场景的成功率，不能据此认定学会了方向。
实际需转向率与固定无树分布参考接近，与弱配对反应一起支持“固定动作偏好及
采样解释大量成功”的判断。两者接近不是统计等价证明，也不说明策略在自然地图
所有状态都恒定；每几何条件只有 20 次采样，策略途中仍可能对位置变化作出反应。

## 审计、验证与产物

完整回归 **198 passed**，其中新增 14 项测试覆盖所有几何/库存阶段的原生转向和
采集、树移除只改变一格及对应 RGB、无树负对照、全资源与不生成实体、observer
不影响轨迹和 RNG、增量成功口径、固定协议拒绝，以及解析参考与原生两步执行
逐项概率比较、八步均匀参考一致性。

12 组主 observer 对照及重复进程的 1 组对照均通过；基线 seed0/100k 的 2016 次
完整 rollout 在独立 Python 进程中复现，逐步 canonical digest、初始 RGB/概率和
汇总完全相同。12 个 checkpoint 参数与 optimizer 未改变；31 个源码、checkpoint、
配置及已安装 Crafter 文件哈希全部通过。主实验 174255 步、3595 次 wood 增量，
原生采集规则错配和安静条件违例均为 0。分析另核对主实验与重复的全部 13 份轨迹。

重复进程完成 14263 步，13 次 observer control 共 83 步，总诊断交互 **188601 步**；
训练交互 **0 步**。主实验、重复和 observer control 共 26221 次人工场景 rollout。
实验进程已退出，本轮使用的 GPU 1 空闲。

- 配置：[crafter_wood3_controlled_collection_cuda_v1.yaml](../configs/crafter_wood3_controlled_collection_cuda_v1.yaml)
- 场景实现：[crafter_controlled_collection_scene.py](../experiments/crafter_controlled_collection_scene.py)
- 冻结评估：[run_crafter_wood3_controlled_collection.py](../experiments/run_crafter_wood3_controlled_collection.py)
- 分析：[analyze_crafter_wood3_controlled_collection.py](../experiments/analyze_crafter_wood3_controlled_collection.py)
- 测试：[test_crafter_controlled_collection.py](../tests/test_crafter_controlled_collection.py)
- 主结果：[summary.json](../results/crafter_wood3_controlled_collection_cuda_20261009_v1/summary.json)
- 最终分析：[analysis_final/verification.json](../results/crafter_wood3_controlled_collection_cuda_20261009_v1/analysis_final/verification.json)，
  同目录保存 outcomes、wood_stages、directions、paired_responses、direction_probabilities、
  facing_sensitivity、fixed_action_references CSV，以及 PNG/SVG 图。
- 场景图：[controlled_scene_examples.png](../results/crafter_wood3_controlled_collection_cuda_20261009_v1/controlled_scene_examples.png)

![完成率及配对反应](../results/crafter_wood3_controlled_collection_cuda_20261009_v1/analysis_final/controlled_collection_comparison.png)

![100k 方向概率及无树对照](../results/crafter_wood3_controlled_collection_cuda_20261009_v1/analysis_final/controlled_direction_responses.png)

复现命令：

```bash
PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 \
/home/zl202621/.conda/envs/kecrl/bin/python -u \
  -m experiments.run_crafter_wood3_controlled_collection \
  --config configs/crafter_wood3_controlled_collection_cuda_v1.yaml \
  --output results/crafter_wood3_controlled_collection_cuda_NEW

/opt/anaconda3/bin/python -m experiments.analyze_crafter_wood3_controlled_collection \
  --input results/crafter_wood3_controlled_collection_cuda_NEW \
  --output results/crafter_wood3_controlled_collection_cuda_NEW/analysis_final
```

## 判断边界与下一项

本轮与上一轮自然轨迹中的低转向/低机会选择性相互支持，说明基础 Policy 在空间
条件到动作的执行层仍有缺口。纯草地、固定光照、人工库存可能偏离训练分布；
8 步也不测长期恢复、探索、生存或木材任务链。因此不能把这里的比例推广到自然
地图或其他任务，也不能断言 encoder 不含朝向/树方位信息。

辅助策略在部分人工局部指标上更高，但未解决转向，而且此前真实资格成功率更低，
这不足以恢复该训练 arm。继续保留零辅助损失、零死亡惩罚 CNN-only 基线；本轮
不新增训练 arm。下一项应做**冻结 encoder 的树方位与朝向联合信息可读性诊断**：
在多种背景上均衡构造可采/需转向/无树状态，按背景划分训练/验证/保留集合，
同一场景重复及配对变体必须在同一分组，离线拟合受限 readout，区分信息可读但
actor 未使用与 encoder/readout 的信息不足。该 readout 只用于诊断，不能接管
Policy 或冒充 Module。之后才据结果选择方向监督、局部课程或其他训练改动。

后续已完成这项冻结表示诊断，见
[树方位与朝向联合可读性记录](crafter-wood3-spatial-readout-diagnostic-2026-10-09.md)。

候选资格门槛仍待验证。stone、coal、pickaxe 本轮未重新测试。源码与文档为工作区
修改，尚未提交；运行源码、已安装规则、配置和 checkpoint 哈希已保存。
