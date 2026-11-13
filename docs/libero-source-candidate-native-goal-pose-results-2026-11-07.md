# LIBERO来源candidate原生成功条件与保存终点位姿核对（2026-11-07）

**“靠近盘却未完成”已分离出具体条件：当前原生目标要求root原点XY距离严格小于3cm，并同时满足高度和碗盘接触记录。task7此前重点的三条A臂未完成轨迹在最后6个保存边界均超出XY门槛，其余两项影子代理满足。** 它们分别只多出1.33、18.46、7.96mm，不能因为视觉上接近或已有碗盘接触就算完成，也不能仅凭持续close归因为缺open。

本轮训练决定：保持500 updates、稳定化5步及现有loss，不修改原生成功判据、不统一增加open偏好、不直接实施新训练。后续候选机制优先检查末期平移精度与落盘过程；当前证据尚未证明增加平移loss权重或其他训练变量会修复它，不能直接选权重重训。

这是已知source场景上的只读diagnostic，承接[close/几何对齐](libero-source-candidate-close-geometry-alignment-results-2026-11-07.md)和[本次固定口径](libero-source-candidate-native-goal-pose-plan-2026-11-07.md)。保留task0/1/3/4/7、seed20261106/07/08、A/B、candidate0..3全部120条、60对；原采集150步前缀和提前成功截止不变。不访问validation/held-out/qualification/reserved/final payload，不恢复G准入或运行H，不更新SPT/Knowledge/Module。

## 原生判定实际检查什么

五个source BDDL的目标均为`And(On akita_black_bowl_1 plate_1)`。调用链为[wrapper.check_success](/home/zl202621/third_party/LIBERO/libero/libero/envs/env_wrapper.py:103) → [具体problem._check_success/_eval_predicate](/home/zl202621/third_party/LIBERO/libero/libero/envs/problems/libero_tabletop_manipulation.py:135) → [On](/home/zl202621/third_party/LIBERO/libero/libero/envs/predicates/base_predicates.py:62) → [物理ObjectState.check_ontop](/home/zl202621/third_party/LIBERO/libero/libero/envs/object_states/base_object_states.py:78)。plate_1是物理object，目标没有走SiteObjectState恒True的接触接口；source初始关系不同，不改变最终目标。

| 原生合取项 | 当前源码定义 | 本轮保存字段能做的核对 |
|---|---|---|
| 水平对准 | root原点XY距离严格<0.03m | 计算影子bowl/plate world root XY与有符号margin |
| 高度 | plate root z≤bowl root z | 计算影子bowl z−plate z，等高允许 |
| 碗盘接触 | 两物体contact_geoms在原生contact数组有匹配记录 | 对影子bowl相关记录中plate组做同口径的记录存在检查 |

[robosuite.check_contact](/home/zl202621/.conda/envs/libero-kecrl/lib/python3.10/site-packages/robosuite/environments/base.py:562)不检查dist≤0或force，正margin记录也可满足该接口。与此前双侧手指dist≤0代理不同，本轮同时保存plate全部记录和非正/正margin计数。当前120终点和720个末窗边界中的plate正margin记录均为0，两种contact口径在本批数据没有改变合取结果；这不是下文6条不一致的解释。

3cm来自原生代码，不是根据诊断结果调出的阈值，而且限定的是XY而非XYZ。没有显式的gripper open、手指脱离、固定朝向或持续若干步要求；朝向仍可能经碰撞几何间接影响接触，不能因此排除旋转控制的重要性。原生目标达成也不等于稳定放置契约或Module qualification。

## 全部终点核对

120条中，原生标签仍为19条达成、101条前缀内未达成。以下仅对后101条列互斥的**保存影子条件**缺项，不改判原生结果，不将影子代理当作原生缓存的精确复算。

| 保存终点未满足的项 | 前缀数 |
|---|---:|
| 仅XY | 10 |
| XY与高度 | 11 |
| XY与碗盘记录 | 57 |
| XY、高度与碗盘记录 | 21 |
| 仅碗盘记录 | 2 |
| 无缺项 | 0 |

由此，99/101条保存终点在3cm圆外或边界上；32条高度项不满足、80条没有影子碗盘记录，后三个计数有重叠，不能相加。其余2条XY在线内、根部高度项满足，但没有影子碗盘记录。这个结果说明终点距离小、碗曾升高或存在手指接触都不足以替代完整目标条件；它不说明99条都有同一个平移动作头根因。

所有10个task×arm及30个task×seed×arm格均保留。完整30行人读表见`/tmp/libero-source-candidate-native-goal-task-seed-arm-20261107-v1.csv`；原结果JSON含43组含总体的均值、中位数、范围及组合计数。没有选择seed/场景、显著性检验或阶段阈值搜索。

## task7同seed四初态：具体差在哪里

A_original、seed20261107四个初态均保留。margin=30mm−XY；正值在线内，0仍不满足严格小于。它们的保存高度项全部通过，四个终点都各有1条plate接触记录。

| 初态 | 保存终止步 | XY距离（cm） | XY margin（mm） | bowl−plate z（cm） | 影子碗盘记录 | 原生标签 |
|---|---:|---:|---:|---:|---:|---|
| 0 | 127 | 2.7265 | +2.7349 | 1.9370 | 1 | 已达成 |
| 1 | 150 | 3.1327 | -1.3268 | 1.5946 | 1 | 前缀内未达成 |
| 2 | 150 | 4.8462 | -18.4616 | 1.3623 | 1 | 前缀内未达成 |
| 3 | 150 | 3.7958 | -7.9580 | 1.3076 | 1 | 前缀内未达成 |

初态1/2/3最后六个边界始终有碗盘记录、root高度项通过，XY分别保持在约3.1085–3.1327、4.8406–4.8462、3.7958–3.8040cm，均在线外；不是终点单帧的舍入翻号。初态0在最后六帧XY一直在线内，但直到最后保存边界才出现影子碗盘记录。该例的XYZ距离3.3445cm仍可达成目标，因为原生3cm门槛只约束XY。

成功例从动作41到127一直close，没有close后的open便达成原生目标。这足以反对把本组所有未完成归因为“必须先open”；不同初态不能据此用作释放的因果对照。当前几何也不能逐项证明原生缓存的数值，只能说明保存几何与源码要求之间的具体差项。

## 两条XY已经在线内、仍缺接触的反例

| task/seed/arm/初态 | 终点XY（cm） | bowl−plate z（cm） | 影子碗盘记录 | 原生标签 |
|---|---:|---:|---:|---|
| 7/20261106/A/0 | 1.6474 | 5.2999 | 0 | 未达成 |
| 7/20261106/B/2 | 0.5312 | 7.4995 | 0 | 未达成 |

这两条的保存root几何显示水平位置已进入3cm圆，但碗原点仍高于盘5.30/7.50cm，且影子没有碗盘记录。它们支持另查下降/接触协调，不能简单靠收紧XY就解释全部结果；也不能把这些root高度差当成表面间隙或认定真实夹持力。

## 六条原生成功与影子接触不一致：保留标签与缺项

影子三项合取为true的13条均为原生成功；另6条原生成功在保存影子终点仅缺plate记录。全部原生未完成终点的影子合取也为false。以下6条全部单列，原生19条成功不改成13条，不把114/120的一致数称为新成功率或预测准确率。

| task | seed | arm | 初态 | 保存XY（cm） | 保存bowl−plate z（cm） | 原生目标 | 影子plate记录 |
|---|---|---|---:|---:|---:|---|---:|
| 0 | 20261108 | A | 0 | 1.6636 | 0.2932 | 已达成 | 0 |
| 1 | 20261106 | A | 1 | 1.9223 | 1.0153 | 已达成 | 0 |
| 1 | 20261106 | A | 2 | 1.7306 | 1.7169 | 已达成 | 0 |
| 1 | 20261106 | B | 1 | 0.4616 | 0.7715 | 已达成 | 0 |
| 1 | 20261108 | B | 2 | 1.7035 | 0.7481 | 已达成 | 0 |
| 3 | 20261106 | B | 0 | 2.1973 | 1.0457 | 已达成 | 0 |

这是需要保留的测量限制：原生谓词读取native `body_xpos/contact`；本批sidecar保存的是独立MjData在当前状态上重新计算的位置与碰撞，两者不是同一个缓存。控制循环先forward再step，目标判定后没有为观察器刷新原生缓存；保存的hash只证明observer未改动原生状态，不能反推出原生缓存内容或相应时刻。

其中2条前一控制边界的影子合取为true，另4条最后6个保存边界都没有影子plate记录。控制边界间隔0.05秒，不能用前一帧冒充原生物理子步缓存，也不能据此指定精确2ms延迟。当前未归档原生三项布尔/操作数，无法证明具体差异来自哪个物理阶段、瞬态接触或缓存更新；不能把不一致直接写成环境bug、标签错误或训练失败。若后续确需精确复算原生目标，须另冻有限采集方案，直接归档判定时的最小必要操作数；本轮不追加采集。

## 执行与验证

[分析入口](../experiments/analyze_libero_source_candidate_native_goal_pose.py)只读源码/BDDL，AST核对参数顺序、严格XY边界、等高条件和无dist过滤的contact接口；从未导入或执行原生环境模块。LIBERO revision为`8f1084e3132a39270c3a13ebe37270a43ece2a01`，源码检查时工作区干净；6个已在原采集绑定的相关依赖SHA一致，另固定9个源码和5个source BDDL。

主计算约3.14秒，独立[审计器](../experiments/audit_libero_source_candidate_native_goal_pose.py)约0.86秒，29879项单元格/汇总核对通过。两次均有Path.open读取计数和禁用模型/环境import、checkpoint/初态/数据集路径guard，240份JSON各读取一次；0拒绝尝试、0新增研究调用。后续证据整理只读新CSV，原始geometry数值不再解码。132项重点测试通过（22项新增，9.80秒），三个新Python文件编译通过；没有重复上阶段全CPU回归或声称全环境pytest通过。测试全为合成保存字典/源码，不消耗真实场景预算。

## 产物与provenance

新产物放在/tmp，拒绝覆盖。主结果保留写入时pending-independent-check状态，审计和completion另记通过，不回写原始文件。

| 产物 | SHA-256 |
|---|---|
| 科学口径：`configs/libero-source-candidate-native-goal-pose-readonly_v1.json` | `a15f2cc84c9ecfa98b0b0358eeb70adaafa3f6f866f940e44c8741d17149e063` |
| 执行绑定：`configs/libero-source-candidate-native-goal-pose-binding_v1.json` | `ae4de1bc529ac51e93a0e075daf5f295c0512d45907baee58851794207685d6c` |
| 主结果：`/tmp/libero-source-candidate-native-goal-pose-20261107-v1.json` | `cfc547dd323b8161a1f0d1e4daeadda49ba7291c883ac7a2622f41bf82d27430` |
| 独立审计：`/tmp/libero-source-candidate-native-goal-pose-independent-audit-20261107-v1.json` | `041a19f051fefbfe335cc819a3ba7daaf182e9334a62caa10d1adec5a37f4003` |
| 证据汇总：`/tmp/libero-source-candidate-native-goal-evidence-review-20261107-v1.json` | `017f5f7a150a7dfe67a29d2202dc1aa0489464742211afbc067bb1803d913719` |

三张CSV位于`/tmp/libero-source-candidate-native-goal-pose-tables-20261107-v1/`：

| 表 | 行数 | SHA-256 |
|---|---:|---|
| terminals.csv | 120 | `defc0211c814aba4e0523c5d980c9a1078071a310ed2c28f4e27e4cddf239c4f` |
| terminal_last6.csv | 720 | `723f86d9fa87e33320f30e8ea41db5c52e4c4857de976de59ae59b540b2e93b2` |
| proxy_native_disagreements.csv | 6 | `15c896ce7de54239e56c2e413093b0ebbc4a4de5688bc22c0d7b55c48f878cc2` |

完整30组表的SHA及末窗反例见证据汇总。过程与独立审计计数分别为`/tmp/libero-source-candidate-native-goal-pose-process-20261107-v1.json`、`/tmp/libero-source-candidate-native-goal-pose-audit-process-20261107-v1.json`；专项日志为`/tmp/libero-source-candidate-native-goal-pose-focused-tests-20261107-v1.log`。最终文件/预算/谱系核对另存`/tmp/libero-source-candidate-native-goal-pose-completion-20261107-v1.json`，不修改原采集绑定、历史结果或checkpoint。

## 训练决定与下一步

本轮修正的是解释口径：人看来接近的3–5cm并不满足此版本严格3cm的XY要求，而XY在线内也不能替代接触。由此应优先检验末期平移拟合与落盘协调，当前没有证据直接选择新的loss权重、增加updates、统一提前open，或认定某个head/视觉路径为根因。也不能把影子代理作训练标签来规避6条不一致，或改宽原生成功门槛。

继续500 updates、稳定化5步及现有loss。下一步先核对已有合法专家输入/预测缓存中的末段平移动作目标误差，区分“相同专家观察下已经拟合不准”与“仅自由闭环终点偏离”，再冻结一个单变量的平移训练候选；若没有所需缓存，先明确字段和只读读出预算，不擅自前向或训练。需要精确接触机制时另冻原生判定操作数的有限归档，不能用当前影子接触代替。G停止、H未开展及全部封存角色保持。
