# 事件前后、错号方向与同帧平移误差只读协议（2026-11-03）

在本次分解前冻结[配置](../configs/libero-source-internal-event-phase-direction-translation-readonly_v1.json)。仅读取[前序逐帧读出](libero-source-internal-fixed500-expert-frame-target-error-diagnostic-2026-11-03.md)已保存的JSON/NPZ，task0/1、seed20261022/23/24、A/B/C全部18个500端点，6954个唯一source-support专家步和62586行既有预测；新增Policy前向、训练、环境、HDF5/RGB/初态/checkpoint调用均0。保留500 updates、闭环5步和所有封存角色，不修改旧产物。

主分析near-open，near-close为预设参照；半径仍为5。仅从已保存的action1及之后目标重建t>=2的专家符号切换，不推断或读取缺失action0。每个原near-event并集帧只关联一次，选5步内最近同类事件，等距选较早事件。lag<0/==0/>0分别为pre/at/post。没有可重建事件的原窗口帧仅允许t<=6，保留为unresolved_boundary，不补标签；已知事件覆盖必须是原窗口子集。输出原事件/示教/帧ID及全部11个lag的有效计数，不对同类重叠窗口重复计帧。

主口径保留全部原窗口；次级同时报告排除原near-opposite-event重叠帧的结果，不替换主口径。open/close两分析可以复用同一帧，不能相加成新的独立样本。

每个phase按严格符号分成四个互斥桶：专家open预测close、专家close预测open、涉及neutral的错号、符号正确。除错误数/phase帧数外，两个方向分别给出错误数/该专家符号帧数；若专家类别根本不存在，条件错误率为null，不能当0错误。phase和标签类别不同可能本身改变错误构成，不能据前后错号方向变化命名为真实提前/延迟或物理释放失败。

平移误差为原float32动作前三维的逐帧MSE，用float64计算，无“高误差”阈值。每seed/phase将方向性错误与**相同专家符号**且预测正确的帧比较；另给同demo内两组同时存在时的均值差、有效/缺失demo数及中位数。两组帧状态仍可能不同，不能当因果作用。

跨seed使用真正相同的专家帧：至少一个seed发生指定方向错误、至少另一个seed符号正确时，计算错误seed的平均平移误差减正确seed的平均平移误差，每唯一帧只计一次。同时保留三个seed共同错误的交集/并集/Jaccard、两两交集和全部共同帧的demo/时刻/lag。共同错误不等于环境机制，模型读出不是独立环境样本。

全部task/arm/event/context/phase和两个方向均保留。只有三个seed都有错误组与同专家符号正确组，且各自平移均值差均严格正时，才标记描述性一致伴随；缺组为null，反向为false，不据此选择loss。B-A/B-C另保留完全相同phase帧上的平移MSE和符号错误率配对。

空组均值/比例为null并记录计数，不补0，不以统计显著性、闭环成败或有利seed筛选。先绑定代码/测试/本文，再运行算术分析；hash、矩阵、合法帧ID、范围、有限性、原窗口关系任一不符停止，旧输出不覆盖。独立审计复算主要表格、分母和交集，新增模型/源数据调用0。执行结果、支持证据、反例及下一步另写报告；不接受活动SPT、更新Knowledge或注册Module。
