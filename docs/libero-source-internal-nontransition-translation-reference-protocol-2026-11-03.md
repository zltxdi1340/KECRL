# 非切换帧平移误差参照的只读核对协议（2026-11-03）

本协议在本次参照均值和调整后差值计算前冻结，复用[前序事件前后分解](libero-source-internal-event-phase-direction-translation-diagnostic-2026-11-03.md)的全部192个跨seed组及原NPZ。[配置](../configs/libero-source-internal-nontransition-translation-reference-readonly_v1.json)固定task0/1、三个seed、A/B/C全部18个500端点，6954个唯一source-support专家步、62586行已有预测。保持500 updates、闭环稳定化5步及现有loss/观测；新增模型前向、训练、环境、HDF5/RGB/初态/checkpoint调用均0。

参照主口径为保存的`away_from_transitions`，必须精确等于`~(near_open|near_close)`：同时在两类切换事件前后5步窗口之外的全部合法专家帧。task0/1分别2207/3324帧，同一任务所有模型共享相同参照帧，不按预测符号、误差大小、闭环结果或有利seed筛选。每task/arm/seed单独取平移MSE均值`b_s`；不能合并任务或将模型重复行算成新环境样本。

次级口径在同一固定mask内再按被比较帧的**专家**夹爪符号取均值，区分参照帧的命令构成；该类别无参照时保留null，不退回全局均值，不改变原对照帧或seed分组。

原每帧误差`e_s,i`仍为float32预测/目标转float64后前三维平方差的均值，定义残差`r_s,i=e_s,i-b_s`。允许负值，不裁剪、不做比值或重新缩放。严格复用原同帧的错号seed集合W、符号正确seed集合C及全部帧ID：

- 原差`D_i = mean_W(e_s,i)-mean_C(e_s,i)`。
- 参照差`B_i = mean_W(b_s)-mean_C(b_s)`。
- 调整差`A_i=D_i-B_i`，并与`mean_W(r_s,i)-mean_C(r_s,i)`核对abs1e-12。

同seed内同专家命令的错误/正确帧都减去相同常数，均值差代数上不变，不把该恒等式包装成新独立支持证据。本轮主要检验原**跨seed同帧**关联。

主观察组为两个task全部三臂的open/post/专家open预测close/全部窗口；其他事件、pre/at、反向错号、未解析边界及排除另一类事件重叠口径全部保留。逐帧记录原差、参照差、调整差，给出计数/均值/中位数/min/max/正差数；每demo汇总及等权demo均值/中位数为次级。参照缺失不删原帧，报告有效/缺失数；无可用对照为null，不补0。

若调整后均值仍正，只能说**这个预先选定的加性均值参照没有消除关联**，不能声称已排除所有整体拟合差异、已定位局部因果机制或夹爪错误导致平移错误。若改变符号，也不能唯一归因为整体拟合差异。原差和调整差的正均值/符号反转分别记录，无显著性、因果解释比例或训练/qualification门槛。

先固定8个输入hash、代码/测试/本文，再运行。角色、矩阵、mask、参照计数、帧/seed集合、phase归属、有限性、动作范围或恒等式失败即停止；不改旧产物、不放宽门槛。结果独立审计和dated报告另存，保持全部封存角色，不接受活动SPT、不更新Knowledge、不注册Module。
