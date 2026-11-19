# task7全部18对共同失败：离线事件统计与归因缺口（2026-11-16）

**已完成纯离线分析：18对共同失败=36条轨迹。MLP4/18、GRU7/18出现交接后的负向开爪命令；25/36从未出现。该结果不能支持“所有共同失败都因主动开爪过早”，但不能排除持续合爪时掉落。现有记录缺少EEF/碗/盘坐标和接触分项，不能给出四类物理失败的可靠占比；v2尚未冻结。**

## 做了什么、配置和实际结果

只读[已审计交接实验](libero-postgrasp-temporal-control-results-2026-11-15.md)中全部18对共同失败，不重训、不运行Policy/环境，不读取HDF5、checkpoint或初态。36条均为300步time_limit，只统计各自交接a后的a+1..300，共7350个控制动作。

输入先锁定719项父链/静态源/新实现依赖，再按原SHA读取36个NPZ；只解码actions、joint_states、gripper_states、rewards、terminated、truncated六个数值成员。实际容器字节463475921、216次数值成员解码、205524项数值/布尔投影；RGB压缩字节随完整SHA读取，但未解码像素，未分析照片。NumPy与独立Python标量/列表公式在同一缓存核对720项指标，通过；不是第二次独立读取或物理复跑。16项人造测试、编译和差异检查通过，未跑全量pytest。

| 交接后的命令事件 | MLP /18条 | GRU /18条 | 合计 /36条 |
| --- | ---: | ---: | ---: |
| 至少一次开爪负向命令 | 4 | 7 | 11 |
| 从未出现开爪负向命令 | 14 | 11 | 25 |
| 末50步至少一次负向命令 | 2 | 2 | 4 |
| 首次负向命令距交接步数 | 18—91 | 16—192 | 不据此命名早/晚 |

9/18对双方都没发过交接后负向开爪命令。开爪命令的首次时间不等于碗释放时刻；Panda按符号累计控制目标，正=闭、负=开、0=保持，正值幅度不等于夹持力。关节差g0−g1仅是公开指关节间距代理，不证明抓稳或接触。

所有36条末50步原生总成功信号均False（1800个保存步）；由该任务单一On目标可知On总体不满足，却不能从False拆出究竟哪一子项失败。没有把这些False臆造为接触False、水平False或高度False。

## 用户提出的三个前提，实际核对

1. **当前不是GAP后端。** ResNet保留512×4×4网格，再Flatten→Linear128；已有joint7/gripper2输入。空间表示仍可能不足，但不能将“替换不存在的GAP”或“补上已有本体输入”直接命名为新干预。EEF Cartesian位置目前未输入。
2. **task7没有独立松爪成功门。** BDDL是On(akita_black_bowl_1, plate_1)，ObjectState.check_ontop检查盘中心z≤碗中心z、碗盘接触、两body中心XY距离严格<0.03m。既没有GripperOpen独立合取，也没有单独放平条件。即使一直闭爪，满足On也会成功；“没松爪”本身不能解释原生失败。5cm不是当前真实成功门。
3. **交接步数关联不是已证实的阶段跨度根因。** 75步的8个状态全属seed20261106，且150步两个状态也全失败。场景、源前缀、seed、接近终点程度混杂；晚交接成功较多不构成同场景随机对照。10帧GRU只净增1例，也不足以排除其他历史结构或500步内的优化限制。

| 交接步 | 状态数/臂 | seed集合 | MLP成功 | GRU成功 |
| --- | ---: | --- | ---: | ---: |
| 75 | 8 | 20261106 | 0 | 0 |
| 100 | 8 | 20261106/07/08 | 0 | 1 |
| 125 | 2 | 20261107 | 1 | 1 |
| 150 | 2 | 20261106 | 0 | 0 |
| 200 | 2 | 20261107 | 2 | 2 |

静态依据：[当前空间编码器](../src/algorithms/libero_frozen_resnet18_policy.py)、[Policy观察契约](../src/environments/libero_adapter.py)、[BDDL目标](../third_party/LIBERO/libero/libero/bddl_files/libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate.bddl)、[On映射](../third_party/LIBERO/libero/libero/envs/predicates/base_predicates.py)、[实际check_ontop](../third_party/LIBERO/libero/libero/envs/object_states/base_object_states.py)。安装的Panda格式映射源也已列入静态SHA绑定。

## 为什么尚不能提供四类归因比例

| 所请求的类别 | 现有记录能确认 | 仍缺少的量 | 本轮数量 |
| --- | --- | --- | --- |
| 半空掉落/松爪 | 有无负向命令、发生步号、指关节变化 | 碗高度/运动、与EEF相对运动、实际接触 | 未能判定，不能填0 |
| 水平未到位 | 归一化平移命令和joint运动 | 碗与盘的世界XY位置 | 未能判定，不能用指令积分代替距离 |
| 已对准但悬停未下降 | 公共关节和动作幅度 | 真实对准、物体高度、目标接触/支撑关系 | 未能判定 |
| 已到位但未松爪 | 是否发过负向命令 | “到位”的几何与接触；且松爪并非独立原生门 | 不作为独立原生失败条件，不能据命令归因 |

36条的厘米级几何/接触原因均未知，未生成伪造的“各占几例”。已归档的整体成功False不足以恢复On三个子项。双相机虽存在，但本数值阶段未解码，未校准像素/深度也不能给出3cm真值。指令非零不代表末端正在朝盘子运动。

末50步7关节累计路径：MLP中位0.811rad、范围0.320—4.163；GRU中位1.727rad、范围0.234—5.109。它们并非关节完全静止；但该路径不是EEF距离，不能据此判断有效搬运、末端悬停或震荡根因。归一化平移指令绝对值均值的中位数为MLP0.07793、GRU0.09781，旋转为0.03817、0.05242；仍只是命令描述，不是几何误差或失败证明。

## 全部36条命令明细

步号为全episode动作步，交接后偏移=首次负向步−a；“无”仅表示交接后从未发负向命令。

| 轨迹 | 交接步a | 首次负向步 | 相对交接步数 | 负向命令步数 | 末50步负向数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| flip00-unchanged-frame_mlp | 75 | 166 | 91 | 4 | 0 |
| flip00-unchanged-gru_window10 | 75 | 无 | 无 | 0 | 0 |
| flip00-confirm5-frame_mlp | 75 | 无 | 无 | 0 | 0 |
| flip00-confirm5-gru_window10 | 75 | 150 | 75 | 2 | 0 |
| flip01-unchanged-frame_mlp | 150 | 无 | 无 | 0 | 0 |
| flip01-unchanged-gru_window10 | 150 | 无 | 无 | 0 | 0 |
| flip01-confirm5-frame_mlp | 150 | 无 | 无 | 0 | 0 |
| flip01-confirm5-gru_window10 | 150 | 无 | 无 | 0 | 0 |
| flip03-unchanged-frame_mlp | 75 | 无 | 无 | 0 | 0 |
| flip03-unchanged-gru_window10 | 75 | 无 | 无 | 0 | 0 |
| flip03-confirm5-frame_mlp | 75 | 无 | 无 | 0 | 0 |
| flip03-confirm5-gru_window10 | 75 | 196 | 121 | 2 | 0 |
| flip04-unchanged-frame_mlp | 75 | 无 | 无 | 0 | 0 |
| flip04-unchanged-gru_window10 | 75 | 无 | 无 | 0 | 0 |
| flip04-confirm5-frame_mlp | 75 | 101 | 26 | 106 | 0 |
| flip04-confirm5-gru_window10 | 75 | 100 | 25 | 52 | 0 |
| flip05-confirm5-frame_mlp | 100 | 无 | 无 | 0 | 0 |
| flip05-confirm5-gru_window10 | 100 | 无 | 无 | 0 | 0 |
| flip06-unchanged-frame_mlp | 75 | 无 | 无 | 0 | 0 |
| flip06-unchanged-gru_window10 | 75 | 136 | 61 | 97 | 0 |
| flip06-confirm5-frame_mlp | 75 | 93 | 18 | 114 | 50 |
| flip06-confirm5-gru_window10 | 75 | 91 | 16 | 33 | 0 |
| flip08-unchanged-frame_mlp | 100 | 无 | 无 | 0 | 0 |
| flip08-unchanged-gru_window10 | 100 | 无 | 无 | 0 | 0 |
| flip08-confirm5-frame_mlp | 100 | 无 | 无 | 0 | 0 |
| flip08-confirm5-gru_window10 | 100 | 无 | 无 | 0 | 0 |
| flip09-confirm5-frame_mlp | 125 | 无 | 无 | 0 | 0 |
| flip09-confirm5-gru_window10 | 125 | 293 | 168 | 6 | 6 |
| flip10-unchanged-frame_mlp | 100 | 无 | 无 | 0 | 0 |
| flip10-unchanged-gru_window10 | 100 | 292 | 192 | 4 | 4 |
| flip10-confirm5-frame_mlp | 100 | 无 | 无 | 0 | 0 |
| flip10-confirm5-gru_window10 | 100 | 无 | 无 | 0 | 0 |
| flip12-unchanged-frame_mlp | 100 | 无 | 无 | 0 | 0 |
| flip12-unchanged-gru_window10 | 100 | 无 | 无 | 0 | 0 |
| flip12-confirm5-frame_mlp | 100 | 189 | 89 | 65 | 50 |
| flip12-confirm5-gru_window10 | 100 | 无 | 无 | 0 | 0 |

## 下一步与v2决定

本轮不直接冻结结构v2：四类物理归因尚未成立，不能以未知类占比或错误前提选模型。SpatialSoftmax、更多机器人本体、action chunking均可作为待论证候选；action chunking须另冻H/k和监督目标/样本曝光/计算公平性。夹爪BCE会改变现loss，不能仅靠尺度对齐称原loss不变；只把同一个线性七维头拆成6+1行且仍用相同MSE，并不自动产生新行为。

若要补齐XY/Z和接触分项，最小方案是[36条已保存动作的几何补录提案](libero-postgrasp-geometry-replay-proposal-2026-11-16.md)。它需要新物理重放，超过本轮纯离线范围：36环境、434reset、2170零动作+10800保存动作=12970native steps，19行初态一次解码，0训练/Policy/权重/HDF5；每帧公开摘要、动作、终止结果须与原44条中的对应记录零容差一致。只有确认补录范围并完成新入口/guard/合成验证与执行绑定，才可运行；本轮未实施。

补录主指标应为碗中心到盘中心的XY距离，EEF到盘仅作解释，因为夹爪抓取偏移会令二者不同。源任务On三子项分别报告，不能强行把有重叠的症状归为互斥原因；物体/目标私有量仅供source诊断，绝不进入Policy或新监督目标。v2性能仍需保留全部22交接状态作回归，不只评价这18个故障状态。

证据：[事前分析协议](libero-postgrasp-common-failure-analysis-protocol-2026-11-16.md)、[科学配置](../configs/libero-postgrasp-failure-readonly_v1.json)、[执行绑定](../configs/libero-postgrasp-failure-readonly-binding_v1.json)、[紧凑证据](evidence/libero-postgrasp-common-failure-analysis-2026-11-16.json)。完整analysis/events.csv/anchor-strata/completion及补录提案JSON在`results/libero-postgrasp-failure-readonly-20261116-v1/`，测试与完整性日志位于同名`logs/`。

Policy500、稳定化5、现loss和所有正式封存保持，未读HDF5、checkpoint、初态，未运行模型、环境、重训或修改旧协议/SHA。LIBERO继续，不更新Knowledge/SPT或注册Module。生成UTC：2026-11-16T09:19:34.328141+00:00。
