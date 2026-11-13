# task7原Policy连续动作误差与固定焦点帧对齐协议（2026-11-08）

**本轮冻结方案，尚未执行新的动作误差对齐。** [近邻分解](libero-task7-lag-neighbor-decomposition-results-2026-11-08.md)已经收束：其取舍取决于距离与集合的交互，不能直接指导Policy加历史或改loss。下一问题是这些已保存的焦点，是否与**原Policy对专家动作目标的离线误差**重合、误差属于错号还是幅值，以及同帧其它动作分量是否也有偏差。

原Policy不变，每帧只有一份已保存的七维输出；“读出纠正/新增错误”不是Policy更新后的改善/损害。本分析可确认同帧原Policy是否偏离专家目标，不能据此确认闭环失败由该动作造成；专家动作也不是唯一可成功控制。保持500端点、未来稳定化5步、原均匀七维masked action MSE、G停止/H未开展；source meta_query、validation、held-out、qualification、reserved及final evaluation payload封存，不更新SPT/Knowledge或注册Module。

## 固定端点、来源与访问角色

[科学配置](../configs/libero-task7-policy-error-focus-alignment-readonly_v1.json)及[元数据门控](../experiments/libero_task7_policy_error_focus_alignment_protocol.py)固定task7、seed20261106/07/08、A_original/B_fomaml全部六个现有500端点，source meta_support demos0..29。A/B是G原始/候选初始化适应臂，不是早期baseline/jitter。编码器已见全部support，本结果仍为训练拟合diagnostic，不称未见泛化、正式迁移或新闭环成功率。

主分析与独立审计每阶段只加载：一份task7共享NPZ的targets、demo_indices、action_steps、demo_lengths、tail20五字段；六份原features.npz中的**predictions单字段**。文件名虽含features，本次不得解码features/logits、公有状态或其它字段，不沿用已关闭26次预算的权限。完整七维predictions为旧CNN-only确定性Policy输出；各文件SHA、checkpoint文件/weight身份与capture、legacy bit parity和G500来源逐一匹配。权重仅作JSON元数据，不打开checkpoint/HDF5/RGB或环境。

标签/目标保持`obs[t−1] → action[t]`、t≥1；共享4188行，六模型原始预测25128行。对齐评分集仍为t≥5且abs(target[2])≥0.025的3598共享帧/21588模型帧，负2273/正1325，末20负402/正75；470个共同中性帧不进入方向对齐。第3分量是动作index2，不是世界z或米制位移。全t≥1的4188/600末20/3588补集只用于原来源指标复现，不把其不同分母混成material结果。

另仅读两张已审计标签CSV：旧frame-transitions与最近的lag frames，各21588行。每张只把配置列出的12列转换为有类型值，包含ID、目标符号、末窗、旧当前/历史及HC标签；旧表另提供Policy_sign作独立对齐核对。CSV全文字节/词法读取与授权列数值解码分开记账，不读取gap/距离/逐lag列去继续近邻分析。

联接键为(task,seed,arm,source_row)，须严格一对一、每模型3598行，逐项验证demo/action_step、目标sign/tail20及由长度重建的原始索引。不得仅凭表行号、排序相近或标签相同拼接。原始NN四格由CC/HH及专家符号重新构造，旧current=CC、旧history=HH；sign(Policy预测[2])须与旧Policy_sign逐帧相同，预测恰为0也不能丢。

## 固定焦点、参照与完备分组

| 类型 | 目标方向 | 焦点 | 固定参照 |
|---|---|---|---|
| 主：原读出变化 | 正 | corrected：CC错、HH对 | stays_wrong：CC错、HH错 |
| 主：原读出变化 | 负 | new_error：CC对、HH错 | stays_correct：CC对、HH对 |
| 次：中间变化被抵消 | 正 | CC=HH=−1、HC=+1（端点均错） | CC=HH=HC=−1（同一原始错组） |
| 次：中间变化被抵消 | 负 | CC=HH=−1、HC=+1（端点均对） | CC=HH=HC=−1（同一原始对组） |

主参照匹配目标符号和CC原本对错，次参照额外匹配原端点状态。HC仅作已保存分组记号，0新近邻搜索/距离/窗口扫描。每模型、每时间层、每目标方向固定输出7组：all_same_sign、四个原NN转移、cancelled_internal_change/control。四转移互斥完备；两个次级组仅在对应原始stays_wrong正组或stays_correct负组中分割，不与其它组拼成新分母。

时间层为all_common、tail20、complement；末20是时间标签，不解释成接近/放置/释放。全部30条示教、两个方向、全部seed和所有空组保留。跨模型的相同专家帧和相邻时刻不是独立重复。

## 误差指标与动作分量

所有保存float32先转float64，再计算p−y。主离散量为material目标上的Policy符号错误（严格反号或预测恰为0），同时分别保存wrong_sign与prediction_zero。第3分量的六种互斥类别沿用[已有定义](libero-source-expert-component3-error-types-plan-2026-11-07.md)：错号、预测零、同号不足、同号过大、精确相等、目标零预测非零；主material组最后一类应为0，legacy全源参照仍覆盖零目标。不得加epsilon重标。

主连续量为第3分量SSE/MSE/MAE/bias、目标能量及SSE/目标能量；错误中预测abs(p)<0.025与≥0.025分别计数，区别临界输出和有幅度反向，不搜索新阈值。保存abs目标、abs预测、abs误差的n/均值/中位数/Q25/Q75，使用float64线性分位数。目标能量比是相对零输出的离线参照，不是新Policy或成功率。

同帧保留七个轴各自的MSE/MAE/bias/目标能量和相对零误差，以及维度平均的translation[0,1,2]、其它translation[0,1]、rotation[3,4,5]、gripper[6]、all7 MSE。夹爪符号按np.sign严格相等，目标零/预测零计数分开；不能由某一分量更大就决定新loss权重。完整raw输出与目标及逐轴误差在/tmp逐帧表保存，禁止把近邻二值标签当成新的连续动作计算“改进MSE”。

每组同时记录类分母、Policy符号错误数/比例、六类帧数及SSE贡献、幅值和其它分量指标。空组n=0/SSE=0，MSE/比例null；能量为0的比值null，不填0或删seed。帧占比和误差能量占比不混用。

## 对照、示教构成及三seed

每模型/时间层/方向分别计算两种固定“焦点−参照”差，保留两组n及目标幅值/能量差。MSE差、符号错误率差和目标尺度并列，不只按某一平均分认定问题。**有Policy错误与错误在焦点特异集中是两件事**：参照同样高错不否认焦点本身有错，也不能把存在重合写成焦点更特殊。

逐demo全部30行保持；仅当同demo两组均非空才定义配对差，否则保留null及缺项原因。另给共同demo等权平均差、共同demo数、两组共同帧数和差为正/零/负的demo数；它与按全部帧加权是不同描述量，不代替原始对照。若共同demo为0仍保留模型，不能换参照/任务/seed。三seed均逐个报告及完整均值，方向为higher/lower/tie/null；不作p值、显著性、任意大误差门槛或正式接受。

同一源帧按arm对齐三个seed：保留各seed原NN组、Policy六类、符号错误，以及每个条件组的成员票数、Policy错误票数和二者交集票数0..3。分别报告至少1/all3共同帧和所有分歧；Policy原本共同错误的固定参照须单列。联合all3少可能因为NN成员不一致，不能据此宣称Policy没有共同错误。

解释分三层：同帧是否有Policy错号/幅值偏差；相对固定参照是否更大；相同示教构成下差异是否还在。任何层的结果只提供离线误差关联。没有更新Policy或执行环境，不能声称历史输入改善/损害了Policy，不能确定闭环根因，也不自动推广训练候选。

## 预算、输出、独立审计与停止

| 每阶段 | 精确预算 |
|---|---:|
| 原NPZ/字段解码 | 7/11（共享5＋6预测） |
| 固定标签CSV遍数/行 | 2/43176 |
| 授权投影列值 | 518112（两表各12列） |
| 原始Policy行/七轴平方误差坐标 | 25128/175896 |
| 共享目标能量坐标 | 29316 |
| 对齐material模型帧 | 21588 |
| 全源legacy指标核对 | 216标量（6模型×3层×12） |

主分析与独立审计合计14次NPZ加载、22字段、4次标签CSV遍历；字节SHA核对不计作数值解码但须另记。平方误差先对固定raw预测计算再复用到重叠组，组/帧反复统计不是新增独立样本。原14/26次其它实验额度均关闭，禁止因同文件已读过免计本轮读取。

九表固定行数：frames21588、model-groups252、demo-groups7560、contrasts72、demo-contrasts2160、demo-standardized72、crossseed-frames7196、crossseed-summary72、legacy-parity18，共38990行。来源复现覆盖旧all/tail20/complement的valid_steps、七维/平移/旋转/夹爪MSE、夹爪sign及平移逐轴MSE/bias；独立核对所有对齐ID/标签/Policy_sign、七维误差、分组守恒、参照、空组、逐demo及三seed汇总。

CPU、Python3.10.22/NumPy1.22.4、CUDA隐藏、单线程、PYTHONHASHSEED0；主/审计各≤120秒、总watchdog300秒，当前地址空间VmHWM/VmRSS≤512MiB，CSV≤256MiB，无技术重试。记录配置/代码/输入/输出SHA、seeds、设备/软件、实际文件/字段/CSV attempted/completed、PID、起止UTC、耗时/RSS及异常；环境seed/初态不适用。

独立审计分进程，不导入主数值实现；同7归档/2标签表重新计算。整数/ID/字符/布尔严格一致，浮点abs1e−10/rel0，输入SHA前后不变。缺来源/端点/元数据、重复/缺失帧、私有字段、role/dtype/shape/边界或指标/预算不一致即停止、保留partial，不修旧缓存、不重试/覆盖历史。新结果仅写/tmp/libero-task7-policy-error-focus-alignment-20261108-v1.json及对应artifacts；主pending-audit字节不回写，audit/completion另存。

## 本轮状态与下一步

本轮只读取指定JSON、源码/既有汇总元数据及固定输入的字节摘要，创建冻结配置/方案、stdlib元数据门控与合成测试。**真实预测数组、标签行数值解码、前向、训练和环境交互均0，没有这项对齐的新结果。** 实际验证/哈希与工作区保护证据记入`/tmp/libero-task7-policy-error-focus-alignment-freeze-20261108-v1.json`，README/roadmap/500计划同步冻结状态。

未来执行须实现analysis/audit/guard/runtime_tests并与metadata/metadata_tests/protocol七角色、科学配置SHA和预算单独绑定；未绑定预检必须拒绝。优先验证七轴/零目标/零输出、类别/SSE守恒、CSV一对一及列越权、目标尺度、空组、示教构成反转、三seed成员与Policy错误区别、legacy216指标及无重试。完成绑定后再消费14次新额度，结果另报，再决定单变量训练候选。
