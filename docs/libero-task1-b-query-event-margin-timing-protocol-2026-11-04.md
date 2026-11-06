# task1 B query事件margin与时序只读协议（2026-11-04）

承接[完整support/query负结果](libero-task1-b-support-query-history-readout-diagnostic-2026-11-04.md)，本项解释事件夹爪MSE下降但两个seed符号错误增加，保留原joint_query_cue=false和不推进闭环的决定。仅分析现有预测，不将query变成loss、预算或阈值搜索集。

## 固定数据与预算

[配置](../configs/libero-task1-b-query-event-margin-timing-readonly_v1.json)固定task1 B、三个seed20261022/23/24、current_repeat5和causal_history5各500端点。只读取父结果/审计/配置/绑定，以及一个query目标与ID归档、六个query预测归档；数值payload共7份NPZ。仅已有source meta-query demos30..49，不能称未见测试集。新feature、HDF5、RGB、初态、private state和checkpoint payload都不读取。

| 项目 | 固定数量 |
|---|---:|
| 缓存目标／缓存预测行 | 2598／15108 |
| 每模型共同t>=5帧 | 2518 |
| 唯一已知open事件／模型×事件配对 | 23／69 |
| 完整open段的模型×帧配对 | 696 |
| 相对窗口记录：lag−4..5 | 690 |
| 新模型前向、训练、backward、环境调用 | 0 |
| 新源数据、feature、checkpoint payload读取 | 0 |

每个事件始于专家命令非负→负、t>=2；要求全部23事件在已有共同范围内。初始open前缀不伪造事件。696个完整post配对与窗口0..5会重复引用同帧，不能相加作独立样本；69是模型×事件，不是69条独立episode。

## 配对与精确MSE核算

按同一事件当步两个新读出的符号正确性，固定四类：两者正确stable_correct、当前正确历史错误harmed、当前错误历史正确repaired、两者错误wrong_both。负目标下0输出计错并与正输出分开；保留所有69个事件和全部变化案例，不按margin/seed筛选。

每帧保存实际目标y、当前c/历史h的原始夹爪输出、距0绝对值、符号、正确性、delta_g及夹爪平方误差。逐项验证`(h-y)^2-(c-y)^2=(h-c)*(h+c-2*y)`；不假设y恒为−1，实际目标值单列。平移/旋转/七维同帧误差为次级描述，不作因果贡献。

每seed以全部23事件为分母，分别计算四类的`sum(delta_squared_error)/23`，四项必须加回父结果的事件夹爪MSE差；不能用各类条件均值相加。harmed数减repaired数必须精确等于父错误数变化。合并69的量只作固定样本描述，不做显著性检验。

margin不设“近0”阈值，只报告连续值、均值、中位数和范围，保留exact-zero数。空类的条件均值为null；若总体分母23有效，空类对总体的贡献为0，这是空和，不是补出缺失MSE。

## 事件前后和完整命令段

固定窗口lag−4..5，窗口内每个lag保留一条可用性记录。负lag仅取同demo已有预测并保留实际专家符号；缺失不补padding。非负lag仅在同一个expert-open命令段内统计，命令已改变或demo结束须记录缺失原因，不能将其补0或算错。

另跟踪事件到下一个非负专家命令或demo_end的完整open段，不按5步窗截断。调用既有完整段算子，分别保存首次正确lag、未观察到首次正确、正确尾段、最长错段、错误重入及段尾原因；时间缺失保持null，重入保留首次正确的资格分母。固定landmarks为0/1/2/3/4/5/10/20，各自使用可用段分母。

对于harmed当步，区分初始连续“当前正确/历史错误”的长度与历史臂全部初始错误长度。前者可因当前臂后来也变错而结束，不能据此称历史已恢复。repaired按对称规则保存。某段长度1且到边界仍错，不能称已恢复的一帧噪声。段尾是专家命令/示教边界，不是闭环终止或成功。

## 绑定、校验与输出

[分析入口](../experiments/analyze_libero_task1_b_query_event_margin_timing.py)、[专项测试](../tests/test_libero_task1_b_query_event_margin_timing.py)和本文由binding锁定后才执行新增分层。默认命令仅核对元数据，--execute才读归档数组。先核对7项直接pin、父专用完成状态/审计/result/config/binding hash、六个固定读出及7份NPZ hash；再复算父全部既定分层指标，确保动作/索引正确。

输出为`/tmp/libero-task1-b-query-event-margin-timing-20261104-v1.json`。保留每seed及合并核算、全部事件及完整段/窗口、所有缺失和原停止决定。输出或失败存在即拒绝；角色/hash/shape/指标恒等式/固定数量失配立即停并保存完成记录，不放宽规则或重跑模型。

独立审计从原预测重算配对、连续误差、完整时序和边界，指标容差1e-10，不新增神经前向、训练或源数据读取。完成后写dated结果并同步README/roadmap/后续计划。原Policy500、稳定化5步、现有loss、封存角色及SPT/Knowledge/Module边界保持。
