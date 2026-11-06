# LIBERO source 稳定化 0/5 配对诊断（2026-11-02）

承接[晨间报告](libero-overnight-diagnostic-progress-2026-11-02.md)的下一项，检验 source 公共首观察的稳定化设置是否影响固定 Policy 响应及已有 candidate 特征的覆盖距离。只改变 reset 中的固定零动作稳定化步数，保留六个 baseline500，不训练、不执行 Policy 控制，也不新增成功率。

**执行状态：source 采集、离线续算和独立复核均已完成。** 两任务三个 seed 均达到预设描述性 alignment 条件，支持稳定化设置是初始观察差异的一个可复现来源，尚未证明闭环控制改善。原 Path 错误及失败 artifact 保留，未重复采集。

## 执行前冻结的范围

- [配置](../configs/libero-task4-task7-source-stabilization-control-diagnostic_v1.json)，SHA-256 `816ea065533ef9080aaab6b9d4bfa1a1e7c9345dd52c5d4e70912041649ffa02`。
- 原执行 runner SHA-256 `0f44ea3acecb735b9ad52b97b10f9dd4b01d7d26c8c840ce0b5b22b7e551c83a`，完整快照保留为 `/tmp/libero-source-stabilization-capture-runner-frozen-20261102-v1.py`；[仓库 runner](../experiments/run_libero_task4_task7_source_stabilization_control.py)随后修复文件路径类型，详情见执行记录。
- task `4,7`；训练 seeds `20261022/20261023/20261024`；固定原 baseline、CNN-only、500 updates 终点。15 个输入 pin 包含旧覆盖结果/审计、公共 source t0、角色清单、共享代码与 adapter builder；每个 checkpoint 文件/权重及六个候选 feature archive 另按父结果验证。
- 只通过原角色授权器读取 source-support demos `0..29` 的 60 个 `init_state` 属性，用于 source reset；不保存其数值，只记 hash/维数。不读取 HDF5 action/observation 数值，不创建 query dataset，不调用 official 初态加载接口，不读取 live private simulator state。

## 两臂匹配与精确复现门槛

两臂各为每任务单一 adapter，按示教 `0..29` 递增 reset，环境 seed 固定 `20261022`。两臂的 native horizon 和 wrapper max_episode_steps 均为 **300**。builder 均以 stabilization=0 构造；构造后只将 reset 的 stabilization_steps 设为 0 或 5，避免 native horizon 随之变成 305。此行为只在独立 runner 中设置，不修改共享 adapter 或第三方代码。

先完成两个 task 的零步臂，60 个公开观察必须逐元素摘要精确匹配旧 source t0。全部通过后才启动任一五步臂；不按 task 局部通过就提前采集另一臂。每次顶层环境 step 都拦截核验为七维全零动作，记录实际 attempted/completed 次数，异常时关闭环境、保留事件和失败结果。零动作不是“物理状态完全不变”的假设；对照包含这些零动作和物理演化的实际影响。

六个固定 Policy 在每臂各 30 个公开观察上一次批量读出。零步特征/logits/action 与旧 CPU cache 必须精确相同；每模型零步批次再重复一次。四个空间、两种 scaling 下的零步 candidate horizon 汇总也必须与父结果的 runtime_t0 bank 精确相同。任何来源、角色、数值、重复性、权重或预算不符即停止，不增加容差或挑 checkpoint。

## 覆盖对照与解释边界

只复用已审计的 2,400 行 candidate 特征，不重新采集 candidate RGB 或执行候选环境。每个 source 臂都以 demos `0..19` 的首观察作 20 行 bank，以 demos `20..29` 作 10 行 reference；这些是示教编号，不是 qualification 初态编号。

四个空间仍为 visual_concat/fusion/projection/proprio。固定 5NN、float64 平方欧氏距离、稳定的原 demo 行序平局处理、reference95 线性分位数、动作前 index `0/4/9/19`。主分析为 **fusion / fixed_zero_scale**：两臂共同使用仅由零步 bank 拟合的均值/总体标准差，std<=1e-6 用 1；避免换 bank 时同时换尺度。每臂自己的 scaler 仅作完整报告的次级对照，不能替换主结果。

分别报告原始距离与各臂 reference95 超线比例。只有 index0 的平均距离比值 five/zero 在三个 seed 全部 <1，且超线比例 five-minus-zero 在三个 seed 全部 <0，才记描述性 alignment 线索；分母为 0 则比值 null，不发明改善。两臂 reference95 仍可能改变，不能只凭比例下降宣布分布对齐。此条件不触发训练、rollout 或资格评价，没有显著性检验、真实 OOD 阈值或闭环根因结论。

另报告配对 RGB MAE、公开 joint/gripper L2、固定 Policy action 分量差及符号变化。**稳定化后的观察没有新的专家正确动作，不使用原 action[0] 给它计算 MSE。** source 与 candidate 的场景几何、取样分布、原 native horizon 等差异并未因此全部匹配。

## 预算与运行设置

| 项目 | 固定上限／完成时应有计数 |
|---|---:|
| source 初态数值读取（独立 demo 数） | 60 |
| adapters / source resets | 4 / 120 |
| 七维全零稳定化动作 | 300 |
| Policy 控制动作 / optimizer updates / backward | 0 / 0 / 0 |
| 完整模型前向：主批次 / 重复 | 12 / 6 |
| 模型行：主 / 重复 | 360 / 180 |
| 独立 bank scaler / 距离评价组 | 48 / 96 |
| candidate / reference 近邻行 | 38,400 / 960 |

CPU 单线程推理，OMP/OpenBLAS/MKL 均为 1，EGL 离屏渲染，`CUDA_VISIBLE_DEVICES=''`。资源快照显示两张 GPU 有其他负载，本项不启动 CUDA 训练，只作有界渲染；EGL device 1 的启动参数与资源快照另存 `/tmp/libero-source-stabilization-resource-launch-20261102-v1.json`，不修改既有作业。

新专项及相关测试 **46 passed**，其中包括真实 HDF5 仅初态读取、零步不匹配时禁止五步臂、相同 native horizon/reset 顺序、非零动作拒绝、共同 scaler 与三 seed 判断边界。结果、实际计数、独立审计及下一决定在执行后补充。

输出 `/tmp/libero-task4-task7-source-stabilization-control-20261102-v1.json` 及对应 `artifacts` 目录；事件 JSONL、四份公开观察 NPZ、六份特征/预测/近邻 NPZ 保留在实验主机，已有文件拒绝覆盖。qualification、reserved、validation、held-out、final evaluation 保持封存；不更新 SPT/Knowledge，不注册 Module。

## 采集完成后的技术中断与独立离线续算

source 采集实际完成四个 adapter、120 次 reset 和 300 个固定零动作，60 个零步公开观察全部精确匹配旧缓存。四份公开 NPZ 已完整落盘。随后在读取候选特征 archive 的哈希预检处，原脚本将 JSON 中的字符串路径传给要求 `Path` 的 helper，报 `AttributeError: 'str' object has no attribute 'open'`。当时模型前向、近邻统计、backward、optimizer 均为 0。原结果 **status=failed**，SHA-256 `d1f19b66a74a7c2aa351ad44ccc16ab25117db4f191b51ab4b6903856486b71d`，连同错误日志和原代码快照保留；不能当作已完成全部分析。

仓库 runner 仅增加 archive 字符串路径到 `Path` 的验证边界，SHA-256 `54163bd674384bf659264c5f3dfcc3b437b47cae5e9c07cb28dcbc995dfcc576`；[端到端配置 v2](../configs/libero-task4-task7-source-stabilization-control-diagnostic_v2.json) SHA-256 `08d0ea5223e2c1c429125f021efa0876b1a824d15db2534712f26398fae1f5ad`。数值分析、角色和预算均未改，**本轮不执行该端到端 v2**，不重复任何 source reset 或稳定化动作。

另冻结[离线续算配置](../configs/libero-task4-task7-source-stabilization-analysis-diagnostic_v1.json)，SHA-256 `385cd973794e6339042462227040e0a471a40fcdd572d8fa1e3fb092089882f6`；[离线入口](../experiments/analyze_libero_task4_task7_source_stabilization_control.py) SHA-256 `d87de231aecb6ac624afa54ba8877c6c31a6b1da91f59621a54d9761a8df350a`。仅接受这一次已知的 pre-forward Path 错误，要求原采集计数精确完成、四份 cache 齐全、120 个事件顺序匹配、全部 zero gate 通过，重新核验原代码快照/配置/输入/失败进程，再做原预算内的 12 主前向＋6 重复。若原模型读出已开始、采集不全或错误类型不同，一律拒绝续算。

续算新 HDF5 payload、初态读取、环境 reset/step 均为 0；沿用执行前原定的 source pairs、指标、scaler、三 seed 规则和 500-update 权重。结果另存 `/tmp/libero-task4-task7-source-stabilization-analysis-20261102-v1.json`，其六份读出/距离 NPZ 另放 analysis artifacts 目录，原四份公开观察保持不变。

## 结果：共同尺度下的首步距离下降

主分析为 fusion / fixed_zero_scale / candidate 动作前 index0。距离是固定 5NN 平方距离的平均；超线比例每格分母为 20 个既有 candidate，不是错误率或成功率。距离绝对值不跨 seed 相加。

| task | seed 后两位 | 零步 bank 距离均值 | 五步 bank 距离均值 | five / zero | 超线比例：零步→五步 |
|---|---:|---:|---:|---:|---:|
| 4 | 22 | 830.763 | 123.711 | 0.1489 | 1.00→0.10 |
| 4 | 23 | 6,985.827 | 369.896 | 0.0529 | 1.00→0.45 |
| 4 | 24 | 4,597.685 | 1,995.190 | 0.4340 | 0.95→0.00 |
| 7 | 22 | 2,767.523 | 221.431 | 0.0800 | 1.00→0.15 |
| 7 | 23 | 5,122.451 | 262.717 | 0.0513 | 1.00→0.50 |
| 7 | 24 | 8,439.704 | 753.137 | 0.0892 | 1.00→0.55 |

两个任务的距离比值均三 seed <1，超线比例差也均三 seed <0，通过**描述性 alignment 线索**条件。两臂共用零步 scaler，bank/reference 数量和 demo 匹配，主结果不是重新拟合 scaler 本身造成的。将 source 参照改为与既有 candidate 同样的五步稳定化后，初始特征距离更接近；不能把此前所有初始差异都归于 Policy 运行后才产生的 drift，或与稳定化无关的全局 OOD。

阈值反证必须保留：task 4 seed24 的 reference95 从 `140.247` 增至 `25,628.317`，其超线比例为 0 不表示距离为 0 或覆盖完整，平均距离仍为 `1,995.190`。原始距离也下降，但比例下降同时受参照尾部影响。task 7 三组 reference95 则均下降，比例仍下降。完整 reference、per-arm scaler 次级结果及四时间点/四空间均保留；reference 仅十条 source 首帧，尚非有效 OOD 校准。

## 公开观察与固定 Policy 响应

每任务 30 对 source demo 的公开观察差异如下。RGB MAE 除以 255；joint/gripper 用向量 L2，不跨模态相加。

| task | agentview RGB MAE 均值 | eye-in-hand RGB MAE 均值 | joint L2 均值 | gripper L2 均值 |
|---|---:|---:|---:|---:|
| 4 | 0.003006 | 0.037780 | 0.031792 | 0.017165 |
| 7 | 0.003525 | 0.039170 | 0.024033 | 0.017165 |

每种观察在两任务各 30/30 demos 均有变化。eye-in-hand 像素差较大，不直接代表它对动作的贡献更大；两臂同时改变 RGB 和本体状态，尚未分离各自作用。

| task | seed 后两位 | action L2 差均值 | 平移 L2 差均值 | 夹爪绝对差均值 | 夹爪符号变化 |
|---|---:|---:|---:|---:|---:|
| 4 | 22 | 0.103948 | 0.102818 | 0.007698 | 0/30 |
| 4 | 23 | 0.125456 | 0.124553 | 0.005131 | 0/30 |
| 4 | 24 | 0.119030 | 0.118784 | 0.001516 | 0/30 |
| 7 | 22 | 0.338547 | 0.321057 | 0.102323 | 0/30 |
| 7 | 23 | 0.335135 | 0.271418 | 0.191957 | 0/30 |
| 7 | 24 | 0.561528 | 0.506222 | 0.229488 | 0/30 |

固定权重下，同一 source 初始条件的稳定化设置足以改变动作幅值，特别是平移；六模型均没有夹爪符号切换。**这是 action delta，不是有真值的动作误差或改善。** 未用旧 action[0] 给五步观察评分，未测新闭环成功率，未消除 source/candidate 的全部场景和采样差异，不指定低成功率的唯一原因或修改默认执行设置。

## 实际成本、provenance 与验证

source 采集（含末尾预检中断）elapsed `199.8176s`，北京时间 `09:28:52–09:32:12`；离线续算 `1.8502s`，北京时间 `09:36:35–09:36:37`。实际初态读取 60、adapter 4、reset 120、全零稳定化动作 300；主前向 12 次/360 行、重复 6 次/180 行、48 scaler、96 组评价、38,400 行 candidate/960 行 reference 近邻。**新增 optimizer、backward、Policy 控制动作均为 0，没有重复 source 采集或新 candidate reset。**

有效结果 SHA-256：`a2d3df183cb2f830fdcfdd7a1621ac45792920b7ce18a56f8e1ed797b5229115`。原采集与续算 provenance 分开保存；四份公开观察和六份读出/距离 NPZ、六个 checkpoint 文件/权重及原候选 archive 的路径/hash 均可追溯。资源启动记录 SHA-256 `974cbae2a3ea631da41a23fb5bab91ec7f7683e4468217ab2472ae735be70887`；原失败日志 SHA-256 `3b3a49cd818846b9758cf5c974a923b02c74800d234dc93ca22255719a2cb5b9`。EGL device1 的零步图像精确复现旧缓存，Policy 推理仍为 CPU。

独立审计 `/tmp/libero-source-stabilization-independent-audit-20261102-v1.json`，SHA-256 `f183a6ae3417387e3c3508b5c4bc946d5ece58f2dc99853d7c99764601ec62bf`。核对失败/续算 provenance、120 个 reset 事件、60 个零步公开观察、六模型旧特征/预测/coverage 精确一致；独立重算 96 组 bank、全部 38,400/960 行近邻及主判断。汇总重算差为 0，直接相减距离与矩阵公式的最大算术差 `5.59e-9`；elapsed `0.6299s`，0 新模型/环境/HDF5 数值读取。

修复后专项及相关测试 **55 passed**；最终 CPU 回归 **566 passed in 15.55s**，两入口 `py_compile` 通过。继续排除 `tests/test_crafter_*.py`、`tests/test_spatial_gather_spt.py`、`tests/test_smoke.py`，当前环境缺少可选 Crafter 依赖，不声称全量 pytest 或 formal 结果。未改共享 Policy/dataset/adapter/SPT，未提交、清理或覆盖旧 artifact。

最终文件/配置/代码 hash、工作区快照、文档链接/空白检查与已观察测试结果记录于 `/tmp/libero-source-stabilization-final-verification-20261102-v1.json`。历史采集 v1 的 failed 状态保持不变，完整结论引用有效 analysis 结果及独立审计；端到端采集 v2 未执行。

## 下一步应该做什么

最新后续结果：[120对闭环轨迹分析](libero-task4-task7-stabilization-trace-alignment-diagnostic-2026-11-02.md)及[source缓存RGB／本体分解](libero-task4-task7-source-stabilization-input-decomposition-diagnostic-2026-11-02.md)现均完成。source三seed、两参照一致表现为RGB响应大于本体、眼在手大于外部相机，但没有夹爪符号变化；这不是六组闭环根因或新训练依据。保持500 updates与5步，下一项先核对既有增强负结果并冻结合法专家输入上的目标误差对照。下面保留原建议的范围与解释边界，实际执行以新报告为准。

后续更新：用户要求直接比较闭环成功率，现已完成[当前执行环境下两臂fresh对照](libero-task4-task7-stabilization-contemporary-closed-loop-diagnostic-2026-11-02.md)，0/5均为7/120，未形成一致零步改善，保持5步与500 updates。这进一步明确本报告的source特征匹配线索不能直接推出控制收益。当前优先读全部120对公开轨迹及6组相反结果，下面原source RGB/本体分解建议后移，尚未执行。

在本轮 **60 对 source 公开缓存**上分离固定 Policy 的 RGB 与 joint/gripper 响应，再按需要细分两相机。已有七条 outcome transition 的相机敏感性不能直接替代这里的全 source 首帧配对。该项可零环境、零标签读取、零 optimizer 完成，但须先冻结组合、前向预算、交互项定义和解释边界；沿用三 seed 和 500 updates，不选择有利相机或 seed。

若考虑改变 rollout 稳定化设置，另冻 candidate-only 配对协议并保留初始化历史/预算；若考虑训练修改，必须有合法监督。**不能把新五步观察与旧 action[0] 直接组成训练样本。** 当前不自动采用新默认、不进入 qualification 或 SPT/Knowledge 更新。
