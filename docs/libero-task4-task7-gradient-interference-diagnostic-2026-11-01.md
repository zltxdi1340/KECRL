# LIBERO task 4/7 固定权重的平移—夹爪梯度诊断（2026-11-01）

本项已完成：六个现有 500-update baseline 的 source-support 梯度读出、原始梯度归档及独立复核均通过。**共享参数整体的局部平移—夹爪余弦中位数在六个模型中均为正，不支持预设的“一致整体梯度冲突”解释。** 少数窗口和 task 7 的逐帧投影层仍有负向结构，不能说没有冲突，更不能由此指定闭环根因。下一步转向 H2 的跨示教条件可分性设计，保留 baseline，不训练新候选。

本项承接[机制证据汇总](libero-task4-task7-mechanism-evidence-review-2026-11-01.md)。float32 v1 因梯度求和数值门槛失败而停止；有界精度审计后另冻结 float64 v2，保持原容差、数据和判断规则。原 v1 配置/结果保留，不将失败前缀与 v2 合并作机制结果。全部阶段新增 optimizer updates、环境 reset/action steps 均为 **0**。

## 冻结实现和数据角色

- 初版配置：[gradient-interference diagnostic v1](../configs/libero-task4-task7-gradient-interference-diagnostic_v1.json)；当前执行版本：[v2](../configs/libero-task4-task7-gradient-interference-diagnostic_v2.json)。
- 脚本：[analyze_libero_task4_task7_gradient_interference_diagnostic.py](../experiments/analyze_libero_task4_task7_gradient_interference_diagnostic.py)。
- 测试：[test_libero_gradient_interference_diagnostic.py](../tests/test_libero_gradient_interference_diagnostic.py)。
- 不修改共享 Policy、数据访问、SPT、旧训练 runner 或旧结果。六个 checkpoint 固定为 task `4,7` × seeds `20261022/20261023/20261024` 的 `source_baseline` 500-update 点；文件及权重 hash 必须匹配历史 budget-sweep 的该固定节点，不加载其他预算或 jitter/局部候选权重。

只解码 source `meta_support` demos `0..29` 的四类公开 Policy observation 和动作，`obs[t-1] → action[t], t>=1`。每条完整 support action 的 gripper 从非负切到负事件定义 ±5 步窗口并集，排除 t=0；标签仅用于损失分解，不作为 Policy 输入或物理阶段。沿用 source role resolver、causal batch reader 和 local mask 实现，10 个输入文件已固定 SHA-256。

不读取 query dataset、source 示教初态属性、private simulator state 或任何 official evaluation state。源 HDF5 的整文件 SHA 校验是字节身份核对，不解码其中的 query/private 字段。训练/推理语义保持 CNN-only；本项范围不包含首动作，也不是完整原 query MSE 重算。

## 计算预算与数值设置

| 项目 | 冻结预算 |
|---|---:|
| 独立 source-support `t>=1` 输入 | task 4 `4,442`；task 7 `4,188`；合计 `8,630` |
| 每 demo 从 t=1 开始的非重叠 10 步窗口 | task 4 `458`；task 7 `432`；合计 `890` |
| 三 seed 主分析前向 | `2,670` 窗口；`25,890` 有效行；`810` padding 行 |
| 重复验证 | 每 task×seed 首个非空局部窗口，6 次前向，至多 60 有效行 |
| 原始分量反向 | 每窗口 T/R/G/base 四次；局部非空时再做 open/weighted2 两次；含重复最多 `16,056` 次 `autograd.grad` |
| 新 optimizer updates、环境 reset/action | 全部 `0` |

CPU、float32 模型、torch threads/inter-op threads 均为 1，batch 为单个窗口，analysis seed `20261022`，启用 deterministic algorithms；点积与范数以 float64 累加。MKLDNN 实际状态、Python/PyTorch/NumPy/h5py 版本、起止时间、Git 工作区、文件 hash 和真实前向/反向调用数记录入结果。CPU 工作不占用训练 GPU。

## 分量梯度和检查

所有项均使用原 `7×valid_step_count` 分母：`L_base=L_T+L_R+L_G`，额外局部项为 `L_open`，已关闭的历史局部 2× 设计在同一固定权重上的代数式为 `L_weighted2=L_base+L_open`。该代数比较不执行更新、不恢复候选分支，也不搜索系数。

用 `autograd.grad` 读取六项梯度，不调用 optimizer、不写参数 `.grad`。空局部窗口不做额外两次反向，局部指标记 null；padding 不计入损失。对 visual encoder、fusion、CNN-only 逐帧 projection 和三者整体分别计算平移梯度与局部、全部、非窗口夹爪梯度的内积、余弦和范数。输出 head 的平移行与夹爪行互不重叠，其梯度内积必须为 0，作为检查，不用于否定共享层干扰。

损失/梯度代数必须成立；梯度求和使用事先固定 `atol=2e-7/rtol=2e-4`，因为不同 float32 求和路径不保证逐位相等。同一窗口重复计算的预测、梯度和标量记录仍要求逐位/精确一致，不放宽输入或权重 hash 门槛。输入不能被修改；每条 demo 后核对全部模型权重，每次 probe 后检查 `.grad` 为空。非有限数、角色/哈希/代数/重复性不符时 fail-closed，保留失败记录和已有 artifact。

原始梯度全局范数给出阈值 1、epsilon `1e-6` 下的虚拟裁剪系数。分别报告同一权重的 base 与 base+open，裁剪本身不改变合并梯度方向。这不是过去训练日志的重算，也不能转换成实际 Adam 更新；历史 optimizer 轨迹未保存。

## 事先声明的结果判断

主指标是每个 task×seed 的非空局部窗口、共享参数整体 `g_T` 与 `g_open` 的余弦中位数。范数 `<=1e-12` 的余弦记 null；余弦 `<-1e-6` 视为负向，避免把舍入量级当冲突。三个 seed 的中位数均负，才保留为该任务终点附近的一致局部冲突线索；部分 seed 为负则记 seed-dependent，全无则记未形成一致冲突。各层、全局/非窗口夹爪及每 demo 结果均保留，不挑负向窗口；不做显著性检验或 qualification 判定。

即使有一致冲突，也只能准备另行冻结的优化边界对照，不能宣布低闭环成功率已归因，或自动使用多头/梯度修正/不同 clipping。若没有一致证据，下一项优先转向 H2 的跨示教条件可分性设计；不继续局部损失权重搜索。历史 500 步中的干扰不能仅凭终点有无冲突判定。

## 运行前验证与 artifact 规则

新专项加上专家误差、局部损失相关检查 **79 passed**。第一次测试中一项把分量梯度求和错误要求为逐位相等，已改为上述预设数值容差；分析脚本及其容差未改变，重复读出仍要求精确相等。新增实际 CNN-only 架构 probe 也验证了共享层、padding、head 分行与权重不变性。`py_compile`、环境检查已通过；更广的相关回归另记。

配置 SHA-256：`e721d00a52c69a32fbca1dca24b373f0a1a8c1ec1831498c2d8de82b50540602`。脚本 SHA-256：`10d059271f6c4fe8d3782a6c10bb81562f2ab230567047e3d606141f5fbedbf2`。协议执行前重新校验两者及全部输入 pin。

结果计划写入 `/tmp/libero-task4-task7-gradient-interference-20261101-v1.json`；artifact 目录为 `/tmp/libero-task4-task7-gradient-interference-artifacts-20261101-v1`。每个窗口标量读出、source 输入/目标/mask 摘要各存 JSONL；六个首个非空局部窗口保存完整分量梯度、公开目标/mask/预测到 NPZ，并记录参数布局，便于独立复算点积和代数。不会复制 checkpoint、RGB 或数据集进仓库，已有输出存在时拒绝覆盖。

本项属于 LIBERO 机制诊断；qualification/validation/held-out/final evaluation 保持关闭，不更新 SPT/Knowledge、不注册 Module。

## v1 停止与有界精度审计

v1 在 task 4、seed `20261024`、demo 0、target `41..50` 的梯度 `base=T+R+G` 检查失败，按门槛停止。共完成 15 次前向、150 有效行、60 次反向，14 个主窗口完成，elapsed `3.06s`；没有完成任何整条 demo，没有选择 checkpoint 或据此判断 H1。失败 JSON 为 `/tmp/libero-task4-task7-gradient-interference-20261101-v1.json`，SHA-256 `6e58526ec6d8ac29098854e569828504158f0f4bc6f838b400e42464aaf238e2`。原 v1 脚本内容另存 `/tmp/libero-task4-task7-gradient-interference-frozen-runner-20261101-v1.py`，SHA 仍为 `10d059…bedbf2`；需要重现旧版时在独立工作副本恢复该源文件及原配置。

随后冻结[两精度审计配置](../configs/libero-task4-task7-gradient-decomposition-precision-audit_v1.json)，由[有界审计脚本](../experiments/audit_libero_gradient_decomposition_precision.py)只重新计算同一个失败窗口、同一 checkpoint。固定两个前向、八个分量反向，零 optimizer/环境交互，elapsed `1.58s`。结果为 `/tmp/libero-task4-task7-gradient-decomposition-precision-audit-20261101-v1.json`，SHA-256 `6b889b9cc097cba7df8b5df6042b5a9199794dd644afca25932c549028da1f33`。

| 精度 | 原门槛通过 | 超限坐标数 | 求和最大绝对残差 |
|---|---|---:|---:|
| float32 | 否 | 1 | `5.5786e-7` |
| float64 | 是 | 0 | `4.3195e-16` |

float64 计算把原 checkpoint float32 数值精确扩展为 double；像素先按原 Policy 做 float32 `/255`，再精确扩展，不添加数据精度或改变输入归一化含义。审计窗口的两精度预测最大绝对差为 `1.31e-7`。结果支持该门槛失败为数值累加差异，没有支持放松门槛、改 loss 或改权重。

## v2 在执行前冻结的修改

v2 仅将离线模型/梯度计算改为 float64，其余数据、窗口、六个固定 baseline、指标、余弦判断及原梯度身份容差均保持。新增 float32 checkpoint 数值摘要检查，以及扩展后的 float64 模型前后摘要检查；不是重新训练或 dtype 不同的闭环 Policy。它检查同一参数值的高精度局部梯度，不承诺逐位重现实际 float32 控制器的预测或 Jacobian。

v2 另存结果和 artifact 到 `…-v2.json` / `…-v2`，重新从全部 support 窗口开始；v1、精度审计和 v2 的实际调用分别记账。原零更新预算与主分析最多 16,056 次反向的上限不变。新增六个 pin 将 v1 配置、原脚本快照、失败结果与精度审计的配置/代码/结果纳入，共 16 个输入 pin。

v2 配置 SHA-256：`92566ee27a6fa14eb4847a0ede6ba79c9395e021e0e105d7bf7cd308ae127e39`；当前脚本 SHA-256：`51f940f48abe12cc5adb1e4dbeeea24ac0b97f9abc4daf32a434c27c818df6e8`。执行前重跑对应数值设置下的相关测试和协议校验，再读取完整梯度。未改变 qualification/held-out 边界或恢复已关闭的局部训练候选。

v2 执行前的相关回归为 **134 passed**，两份脚本 `py_compile` 通过。数值审计只支持失败窗口的精度判断，不是六个模型的 float32/float64 梯度等价证明；饱和或激活边界附近的导数也可能随数值精度变化。另需注意：本项单窗口从 t=1 切分，原训练为 batch 8 且包含真实 t=0；窗口梯度、其等权中位数和虚拟 clipping 均不重建原 minibatch 梯度。CPU 分析耗时不是新的 Policy 学习预算。

## v2 结果：主指标与层间差异

完整覆盖 60 条 source-support 示教、890 个独立窗口及 8,630 个 `t>=1` 输入；每个任务三个 baseline 均评分，共 2,670 个模型窗口、25,890 个有效模型—观察配对。局部 mask 在 task 4/7 分别包含 333/349 个独立步，对应 60/63 个非空窗口；没有空范数导致的额外局部缺失。窗口以外的 null 不伪造为零余弦。

下表为窗口余弦中位数；负向比例的分母只包含非空、可评价的局部窗口。`recurrent` 模块在当前 CNN-only 配置中是**逐帧 MLP 投影**，不是 GRU 或新加入的记忆。

| task | seed | 共享整体局部中位数（主指标） | 局部负向窗口 | 逐帧投影局部中位数 | 共享整体非窗口 gripper 中位数 |
|---|---:|---:|---:|---:|---:|
| 4 | 20261022 | `+0.236817` | `26/60 = 43.33%` | `+0.022151` | `+0.478821` |
| 4 | 20261023 | `+0.198097` | `24/60 = 40.00%` | `+0.064632` | `+0.015516` |
| 4 | 20261024 | `+0.546158` | `18/60 = 30.00%` | `+0.093594` | `+0.015109` |
| 7 | 20261022 | `+0.287915` | `25/63 = 39.68%` | `-0.015997` | `-0.047641` |
| 7 | 20261023 | `+0.147277` | `27/63 = 42.86%` | `-0.046097` | `+0.180837` |
| 7 | 20261024 | `+0.172174` | `20/63 = 31.75%` | `-0.022569` | `-0.306227` |

两个任务均未满足“共享整体主指标在三个 seed 全为负”的预设条件，分类为 `no_consistent_local_conflict_at_fixed_endpoint`。不是因为挑了有利窗口：全部局部窗口的 mean/median/min/max、负向比例、各层及每 demo 结果均保留。

负向窗口仍占约 30%–43%，因此主指标为正不等于每次学习都无干扰。task 7 投影层的三 seed 局部中位数均为负，是事先记录的次级层间线索；视觉/fusion 层局部中位数为正，可能使整体与投影层不同。它不能替代主指标来宣布 H1 成立，也不能通过选中该层直接改网络或恢复候选。非窗口 gripper 在 task 7 也只有两个 seed 中位数为负，没有统一跨 seed 结构，不能把一切冲突限定在 open 窗口。

全部 2,670 个模型窗口的 head 分行内积为 0，梯度求和最大绝对残差为 `1.2268e-14`，输入不变与 `.grad` 为空检查通过；六个 float64 模型前后摘要一致，转回原 float32 数值后的摘要精确匹配各自 500 checkpoint。

## 同一权重上的虚拟裁剪比较

右列是同一 baseline 的 `g_base+g_open`，**不是重训后的局部候选 Policy**。单位为单窗口、固定终点的裁剪激活次数，不是此前 500 个训练 minibatch 的裁剪次数。

| task | seed | base 裁剪窗口 | base+open 裁剪窗口 |
|---|---:|---:|---:|
| 4 | 20261022 | `89/458` | `99/458` |
| 4 | 20261023 | `82/458` | `89/458` |
| 4 | 20261024 | `94/458` | `97/458` |
| 7 | 20261022 | `84/432` | `93/432` |
| 7 | 20261023 | `80/432` | `92/432` |
| 7 | 20261024 | `88/432` | `101/432` |

六组的虚拟裁剪窗口数均增加，说明局部项确实可能改变合并梯度的尺度；它仍不能证明裁剪导致前次训练退化。前次候选沿不同参数轨迹训练，且 Adam 的历史预条件化未重建；方向冲突与裁剪缩放不能相互替代或直接折算成成功率。当前不取消 clipping、不调权重、不增加 updates。

## 实际成本、provenance 与独立复核

| 阶段 | 前向调用 | `autograd.grad` 调用 | elapsed |
|---|---:|---:|---:|
| float32 v1 停止尝试 | 15 | 60 | `3.06s` |
| 同失败窗口的两精度审计 | 2 | 8 | `1.58s` |
| 完整 float64 v2 | 2,676（含 6 次重复） | 11,454（含 36 次重复） | `1,182.61s` |
| 独立来源/算术复核 | 0 | 0 | `2.28s` |

三个计算阶段合计 2,693 次前向、11,522 次反向，elapsed `1,187.26s`；独立复核另计，不与主分析合并为新学习预算。v2 实际 25,950 个有效前向行（含 60 行重复）、810 个 padding 行，少于冻结反向上限。全部阶段新 optimizer/environment steps 均为 0，没有新 checkpoint 或成功率结果。

实际环境：Python `3.10.22`、PyTorch `2.4.1+cu124`、NumPy `1.22.4`、h5py `3.11.0`，CPU、torch threads/inter-op threads `1/1`，deterministic algorithms=true、MKLDNN=true；`CUDA_VISIBLE_DEVICES=''`，OMP/OpenBLAS/MKL threads 均为 1，nice 10。CUDA 训练未启动。FP64 分析使用与各 baseline 同样的 821,704 个参数数值，没有改变容量。

- v2 结果：`/tmp/libero-task4-task7-gradient-interference-20261101-v2.json`，SHA-256 **`222a7aa726beb98fb4b29bf28b418b48e5cde982ffa07f1f39002dff2057eebe`**。
- v2 日志：`/tmp/libero-task4-task7-gradient-interference-20261101-v2.log`，SHA-256 `c878b4329e41341ed56ab97443cd12a6cceb7bb054b8cd6a073f64be3159393d`。
- v2 artifact 目录：`/tmp/libero-task4-task7-gradient-interference-artifacts-20261101-v2`；两份 JSONL 与六份原始梯度 NPZ，共 `95,278,842 bytes`，各文件 hash、参数布局及输入/目标/mask 摘要见结果索引，不含复制的 RGB 或 checkpoint。
- 独立复核：`/tmp/libero-task4-task7-gradient-interference-independent-audit-20261101-v2.json`，SHA-256 **`4bdce49cd405d0eeb9b5dba5126befb2656e12a37c27beeeda33d1f21b0b2174`**。
- 独立复核源码：同 stem 的 `.py`，SHA-256 `7f251fb2a5e3c043817859296896e0fab94d5a192c5581d118ed1041e034036b`。

独立程序未调用模型或原分析的汇总函数。它重读相同 8,630 个允许的公开输入，验证全部 890 个 source 窗口的 causal 索引、目标/mask/输入摘要；重算 2,670 个模型窗口汇总、六个 seed/task 与 180 个 demo/seed 摘要和主指标分类。16 个输入 pin、全部 artifact 及六份 checkpoint 的文件/权重 hash 均核验。另从六份原始梯度重算分量身份、各层点积/范数/余弦及虚拟裁剪系数；该子集身份最大残差 `2.7062e-15`，独立求和次序的标量最大差 `3.1086e-15`，未放宽数据、权重或梯度门槛。

原 v2 执行命令如下；再次执行必须使用新的 result/archive/log 路径，不能覆盖本轮 artifact：

```bash
set -o noclobber
nice -n 10 env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  conda run --no-capture-output -n libero-kecrl python \
  experiments/analyze_libero_task4_task7_gradient_interference_diagnostic.py \
  > /tmp/libero-task4-task7-gradient-interference-20261101-v2.log 2>&1
```

验证为 134 项相关 CPU 测试、脚本编译、环境检查、独立复核及文档/差异检查；未运行含可选 Crafter 组的全量 pytest，不将本轮记为历史 433 项回归重跑。

## 决定与下一步

**保留 500-update baseline；不据本项改 loss、clip、网络、seed 或 checkpoint。** 当前主指标不支持优先用“共享整体稳定冲突”解释失败，投影层负向结果作为次级未解决线索保留。当前分析也未排除早期训练干扰、实际 minibatch 冲突、float32 边界差异或当前 runtime 分布偏移。

下一项按原分支规则转向 H2：先冻结跨示教条件可分性诊断设计，再读取六个现有模型的视觉、fusion、逐帧投影特征，比较同一专家输入下各层的 open/close 条件区分与动作错误。优先在 source-support 内按示教留一建立近邻对照，排除同一 demo 的相邻帧捷径，切换窗口与非窗口分开；在选择邻域规则之前不读取其结果调参。仍不训练 Policy，500-update 预算、原权重和角色边界不变；邻域混叠也不能直接称为真实欠观测、环境因果 Knowledge 或闭环根因。**该 H2 诊断本轮尚未实现或执行**，需要独立冻结配置、预算与判断边界。
