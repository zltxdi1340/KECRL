# 已保存切换近邻的只读分解协议（2026-11-04）

承接[阶段／历史可辨性结果](libero-source-internal-phase-history-observability-diagnostic-2026-11-04.md)，本项解释为什么整体命令读出提高而open切换收益不一致。保持全部18个500端点、稳定化默认5步及现有loss；仅分析已保存的source-support数组，不把历史不足预设为根因。

## 范围与执行边界

[配置](../configs/libero-phase-history-transition-neighbors-readonly_v1.json)在新分层计算前冻结：task0/1、三seed、A/B/C、support demos0..29、共同t>=5、原5NN、当前帧与当前加过去4帧。保留65个唯一open事件和585个模型×事件配对；195个arm×事件的三seed模式均输出。现有近邻、表示、专家目标和Policy输出只读，原结果和代码不改写。

| 项目 | 固定数量／上限 |
|---|---:|
| 全部共同命令帧，按模型计 | 60426 |
| 模型×事件配对 | 585 |
| 两模式事件邻居记录 | 5850 |
| 已选邻居并集，按模型×事件计 | 最多5850 |
| 并集邻居的lag0..4距离值 | 最多29250 |
| 全部进入／离开邻居交叉配对 | 最多14625 |
| 新Policy前向、训练、backward、环境调用 | 0 |
| 新全参照库搜索、k或历史长度搜索 | 0 |
| 新HDF5、RGB、初态、private state、checkpoint payload读取 | 0 |

直接输入9项hash和原执行绑定先核对，再读取NPZ数值；18份表示、18份近邻NPZ按原结果各自hash核对。只用已选两组近邻的并集复算距离，不访问未选择的参照向量来重新排序或搜索。编码器已经训练过全部source-support；本项仍不是未见泛化或正式评价。query、qualification、validation、held-out、reserved和final evaluation继续封存。

## 固定分类与核算

每个open事件按同帧专家符号比较两个已保存命令预测：两者正确stable_correct、当前正确历史错误harmed、当前错误历史正确repaired、两者错误wrong_both。模型×事件与唯一事件的分母分开；没有由本轮近邻变化产生的新Policy动作或闭环结果。

整体共同帧分为初始open、已知open事件当步、已知open事件之后、expert close四个互斥分层。每个模型先核对完整旧指标，再核算错误数变化和BA贡献。分层贡献为`−0.5 × 该分层错误数变化 / 该专家符号的全部共同帧数`；四项之和须等于原整体BA差。不能把各层错误率不加权相加，也不把高频稳定帧改善等同于切换改善。

对两组已保存5NN的并集，记录源demo/step、专家符号、当前/历史rank及进入、离开、保留。邻居自己的命令上下文固定为：初始open；已知open当步；其后1..4步；其后5步及以后；下一次已知open之前1..4步的close；更早的close；没有后续已知open的close；neutral。close仅按自身demo最早后续open锚定，不能误用前一次或其他demo的事件。初始前缀不伪造起点。1..4与5+的划分由既有历史长度决定，不搜索窗口。

记录查询和邻居的当前及过去4帧专家符号，标签只用于分析。用原fold的mean/scale、float64直接平方差、64维块，复算并集每个邻居的5个lag距离；当前距离和5lag均值须在绝对容差1e-8内复现对应已保存距离。

所有进入邻居与离开邻居的交叉配对均保留，记录当前、过去4lag均值和5lag均值距离差。验证进入者当前距离不优于离开者、历史距离不劣于离开者（容差1e-8），以及`ΔD_history=(ΔD_current+4×ΔD_past)/5`。超过容差的严格反转与等距情形分开描述。这个等式解释固定距离规则如何改变邻居，不能认作环境物理因果或真实Policy历史作用。

原Policy夹爪输出、符号和距0绝对值均从旧预测读取，按四类逐模型及六组描述均值、中位数、范围，空组为null。不给margin添加事后阈值，不把5NN与原head的不同表现归为head独立因果贡献。

## 验证、停止与下一步

[分析入口](../experiments/analyze_libero_phase_history_transition_neighbors.py)、[测试](../tests/test_libero_phase_history_transition_neighbors.py)和本协议在执行前由独立binding锁定。默认命令只核对元数据，显式`--execute`才运行只读分析。旧输出或失败文件存在即停止；输入、角色、样本、投票、距离、BA核算或上限不符则保存失败及已完成记录，不放宽规则或重新采集。

完成后独立复核事件分类、邻居身份/符号/时间、距离和整体BA核算，再写dated结果并更新README/roadmap/后续计划。结果只决定是否需要进一步分离读出规则或标签混叠；不自动改训练、延预算、选checkpoint、接受SPT、更新Knowledge或注册Module。
