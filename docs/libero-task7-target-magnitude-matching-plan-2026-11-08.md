# task7目标幅值分层与同示教匹配只读方案（2026-11-08）

**本轮仅科学冻结，尚未投影真实CSV行或执行新匹配。** [上一轮结果](libero-task7-residual-sign-transition-results-2026-11-08.md)中，负新增组原输出abs小，同时专家目标abs均值0.082653–0.155515，仍对参照组0.507350–0.511327。此处需要检验“给定目标尺度与示教之后，原输出距0的关联是否仍存在”，不能直接据上轮组间差选择loss、残差约束或输入修改。

[配置](../configs/libero-task7-target-magnitude-matching-readonly_v1.json)与[stdlib元数据门控](../experiments/libero_task7_target_magnitude_matching_protocol.py)固定task7、三seed20261106/07/08、A_original/B_fomaml六个原Policy500及辅助500端点。编码器已见全部30条source-support示教，OOF仅为辅助头整demo三折；这是诊断，不是未见泛化或正式评价。保持未来稳定化5步、现有loss、四个父joint cue=false、G失败/H未运行和全部source-query/正式评价封存；不更新SPT/Knowledge或注册Module。

## 唯一输入与角色边界

只授权上一轮新生成且完整独立审计通过的`/tmp/libero-task7-residual-sign-transition-artifacts-20261108-v1/paired-frames.csv`：64764行，SHA `7003a9cbb404abbfd4be1212b603419a4e1751b4c4614f3cf70c0015a95a5ef0`。五份父science/binding/result/audit/completion JSON的SHA用于身份和完成审计验证，不用其中指标payload作新分析输入。

该CSV原41列只投影以下28列：task、seed、arm、comparison、candidate、reference、source_row、demo、action_step、target_sign、tail20、target_2、reference_prediction_2、candidate_prediction_2、reference_category、candidate_category、reference_clip、candidate_clip、transition、reference_error、candidate_error、reference_band、candidate_band、reference_sse、candidate_sse、delta_sse、reference_raw_residual、candidate_raw_residual。其它13列不转换或暴露；所需派生量从授权原始标量独立重算。

不读旧oof-frames/curves、NN标签、其他来源CSV，**不打开或哈希NPZ/NPY/头/checkpoint/HDF5、RGB/特征/公开本体或private state**。无新原Policy/辅助头前向、优化器、训练、环境、初态或近邻搜索。所有数据仅action index2的既有专家动作，不解释为世界z、夹爪命令或物理阶段。

## 固定分层、比较与主匹配

完整保留三组候选−参照：bias_only−original、current_affine−original、current_affine−bias_only；两目标方向、all_common/tail20/complement及四格stays_correct/corrected/new_error/stays_wrong均保留。共同t≥5、abs(target_2)≥0.025共享3598帧（负2273/正1325）；末20负402/正75，补集负1871/正1250。每端点/比较3598行，共64764行；不补读首次4步或neutral。

每个端点×比较×phase×方向固定两种对照：corrected−stays_wrong（参照原本都错）、new_error−stays_correct（参照原本都对）。正纠正和负新增为上轮焦点，仍完整输出其余方向与四格。组由已发生的输出转移定义，匹配不是随机干预或策略效果估计。

目标abs固定七层：**[0.025,0.05)、[0.05,0.1)、[0.1,0.2)、[0.2,0.4)、[0.4,0.6)、[0.6,0.8)、[0.8,1]**。以真实保存float32值转float64后和固定十进制边界比较，不round、不用epsilon；不扫描分位数/阈值、不按结果并层。

**主匹配exact_target**：同一端点、比较、phase、目标方向、对照内，按`(demo, abs(target_2)的IEEE float32 bits)`精确分格。只在同一示教、完全相同目标幅值且两组都非空的格内计算均值差。每格使用两组全部帧，不选最近时间、最近观测或一对一子样本，不展开笛卡尔帧对，也不把逻辑帧对数当新样本数。保留只出现一组的格；双方均空的精确格不凭空创建，但其demo仍在摘要中。

**辅助coarse_bin**：同一示教和固定目标层内按相同方式分格。每个对照全部30 demo×7层均输出，包括双方空格。层内目标不必相同，必须并列两组目标均值/min/max和目标均值差；不能把该结果当作精确控制目标尺度。即使主匹配稀疏，也不能用更好看的粗分层替代主口径。

## 估计、权重、覆盖与缺项

每格12项指标固定为：target_abs、reference_abs、reference_abs_over_target_abs、reference_squared_error、candidate_squared_error、delta_squared_error、reference_relative_squared_error、candidate_relative_squared_error、delta_relative_squared_error、raw_shift、signed_effective_shift、absolute_effective_shift_over_target_abs。relative squared error分母为target²，幅值比分母为abs(target)≥0.025，不加epsilon。raw/effective按CSV两输出及raw残差定义；不恢复仿射头W或bias。

每个格均值差=focus均值−control均值；主要关联量为reference_abs差，另外并列归一化幅值、连续误差和残差位移。精确匹配的target_abs差必须恰0，粗格保留剩余不平衡。负差只说明本条件集合中焦点组原输出幅值较小，不代表输入缺少信息或残差位移的因果作用。

按顺序保存三种描述，避免把权重和支持域变化一概归给目标尺度：

1. 全部focus/control帧各自按帧加权的原始差。
2. 限于两组都出现的格之后，仍各自按帧加权的差；组间格权重仍可能不同。
3. **先在每个demo内对共同格等权，再在有至少一个共同格的demo间等权**。focus/control使用同一组格和同一组权重。exact_target/all_common是主口径；末窗、补集和粗格为辅助。

每个对照保留全部30个demo；报告两组full/matched/unmatched帧数、共同格数、共同demo数及coverage。分母0时率null。缺项分别为both_empty、focus_empty、control_empty、no_common_exact_target或no_common_target_bin；缺项不填0，匹配稀疏不放宽条件、不追加数据。没有“足够匹配”的后验门槛；仅有少量共同格时只能描述该支持域，不声称代表全部3598帧。

全部三个seed单独报告，原始及标准化均值只在三个值均可用时定义；同一arm不与另一arm合成六个独立seed。另固定三seed共同匹配格的交集作辅助：精确格按同demo/方向/float32幅值键，粗格按同demo/层；对交集重算相同的格内/demo等权差，并报告覆盖损失。共同格不保证三seed的focus/control帧成员相同，不能替代各seed自身支持域结果。

## 联接、复现与预算

由父元数据30条示教长度重建source_row→demo/step/tail，要求六模型×三比较×3598行唯一完备、目标/ID跨比较和seed/arm一致。三个mode的预测、raw残差、clip和类别须跨比较严格一致；用original及残差精确复现保存float32(clamp(...))，original残差0。target/预测有限、float32精确、范围[−1,1]，六类别/符号/四格/幅值档/SSE及ΔSSE一致。四格、七层与匹配/未匹配计数和误差分解须守恒。

| 每阶段输入读取 | 固定预算 |
|---|---:|
| 来源CSV数据遍历 | 1 |
| 词法行 | 64764 |
| 投影列 | 28 |
| 投影列值 | 1813392 |
| 原来源CSV/数组/权重读取或字节哈希 | 0 |
| 研究训练/前向/环境/近邻/封存payload | 0 |

主分析＋独立审计共**2遍、129528词法行、3626784投影列值**，是新的派生CSV额度，不复用已消耗的原两CSV四遍额度。输入SHA在前后核对，字节SHA单独记账；本轮冻结只核对这一张派生表字节，不创建CSV reader或投影真实行。

未来输出7张表：frame-strata64764、bin-transitions3024、coarse-bin-cells45360、demo-matched12960、model-matched432、common3-summary144为固定行数；exact-target-cells为确定性观察键并集，最多129528行。上限由6模型×3比较×两次phase访问（all_common和其末窗/补集分拆）×3598帧给出，每个帧在一个参照正确性对照中只属一个target键；不是任意浮动行数，实际必须恰等于重建键并集。总CSV行≤256212，另144份完整三seed JSON摘要。frame-strata保存固定层/目标bits、对照角色、各自all_common及本帧末窗/补集中的精确/粗格匹配标记。完整精确格表支持重新联接三seed交集，无需新增近邻或原数组。

CPU Python3.10.22/NumPy1.22.4，CUDA隐藏、单线程及原环境变量；主/审计活动各≤180秒、总watchdog≤420秒、VmHWM/VmRSS≤512MiB、输出CSV≤256MiB，技术重试0、禁止覆盖。记录配置/代码/父审计/输入输出SHA、seed、设备、PID/软件、时间/RSS、attempted/completed遍数/行/投影、实际格数/覆盖率和停止原因；无环境seed或初态索引。

未来入口、CSV/字段guard、独立审计及运行测试须先实现并绑定七角色和传递依赖；未绑定不得读取真实CSV数值。独立审计全新进程，重读同一派生CSV一遍，不导入主数值实现；独立复现全部ID/标量/精确bits/层/权重/覆盖/空格、三种描述与全部输出、三seed交集。保存float32/键/ID/计数/类别严格相等，派生浮点abs1e−10/rel0。任一哈希、角色、联接、字段、守恒、预算或审计不一致，停止保留partial，不回写旧结果、不重试。

## 决策边界与本轮状态

本方案只检查已观察目标尺度的条件关联。原始→匹配的变化同时受共同支持域和权重影响；几何、历史、时间与其它观测因素仍未控制，outcome-defined分组本身也有结构约束。不做p值、显著性、bootstrap、单帧独立样本假设或新的候选接受阈值；不因某层、seed或匹配均值好看而修改loss、选checkpoint、追加训练、进入闭环或恢复G/H。

本轮只读当前源码/报告和五份父JSON元数据，核对唯一派生CSV字节SHA，创建科学配置、本方案、stdlib门控及合成测试；真实CSV投影、原数组/旧来源CSV读取或哈希、研究训练/前向/环境均0。验证和工作区完整性证据另存/tmp/libero-task7-target-magnitude-matching-freeze-20261108-v1.json，README/roadmap/500计划同步。本轮没有新匹配结论，下一步实现入口、字段guard和独立审计，合成验证后绑定代码再按额度执行。

新增81项合成测试、相关协议共358项通过，py_compile及环境检查通过；覆盖精确float32键/固定边界、近似层剩余不平衡、空格/全30 demo、共同支持域损失、格内/demo等权、目标混合造成的均值差反转、三seed交集与角色/预算/未绑定拒绝。本轮仅元数据和新合成规则，未改变共享训练或运行框架，未重复全量pytest。环境检查的CUDA不可用是本轮CPU验证主动隐藏GPU的结果。
