# task1 B路由闭环60对轨迹只读分解协议（2026-11-04）

用户明确要求分析[已完成120条轨迹](libero-task1-b-routed-history-closed-loop-diagnostic-2026-11-04.md)，单列3条baseline-only，检查第5步后动作、close命令与公开状态漂移。本协议先冻结，再读取本轮episode JSON数值；不重训、重跑环境或读取新输入。

## 输入与范围

固定task1、seed20261022/23/24、current_all7 baseline与history_motion_current_gripper候选的500端点、5步稳定化、环境seed20261022。官方candidate0..19均已用过；不称未见。只读父结果/审计/config/binding与120份已保存JSON，共35609公开状态行、35489动作。读取feature/head NPZ、checkpoint、HDF5、RGB、初态/private payload均0；新模型/编码器/读出、optimizer/backward、环境调用均0。

60个同seed/state配对全部保留，3个baseline-only（seed22/state2、seed22/state10、seed24/state5）只作结果分层，不筛掉57个共同失败；candidate-only及共同成功空组仍输出null统计。先复核全部源hash/父状态/500/5步/角色、每份JSON的端点与父摘要，再重算成功四格。失配拒绝，不放宽阈值或改分母。

## 时钟、动作和输入

action[t]使用公开obs[t-1]，JSON row[t]是动作后状态，row0是稳定化后reset。前4动作、公开状态row0..4与前5个pre-action观察digest必须逐位相同；第5步动作发生于共享观察之后。仅第1..5步标为已有证据认证的完整观察相同；之后即使joint/gripper相同，也不假设RGB相同。

分平移0..2、旋转3..5、夹爪6，计算七维和分量动作差、各臂norm、夹爪符号。首次差异用共同可见步中任意分量exact非零，不事后选epsilon；报告差异幅度，不能把微小数值差异当显著物理偏离。比较候选减baseline，不把动作gap当专家误差。

固定两个早期窗口：1..20与5..20，分别保留共享启动和去掉启动后的运动/命令/path指标。符号切换只在窗口内部相邻动作之间计，不把第4→5步的边界切换算入5..20。完整配对序列到min(两臂长度)，17489个共同动作帧配对；前20步1200配对、5..20共960配对。

同步点固定1/4/5/6/10/20/50/100/200/300，共600条含缺失记录。每点区分动作前、后joint/gripper距离及各自相对reset位移；只有两臂均达到该点才比较，成功结束后不补值。另保留各对共同终点和同钟路径，不用候选300步终点与baseline109/137/143步终点作同钟比较。

## 命令时序与删失

close为g>0，open为g<0，0单列；release只是首次close之后的第一个负命令，不是物理释放。报告首close/open时刻、首close段长度、最长段、符号切换、正/负/0段、全程未close、close后未open、首段到episode结束的右删失。沿用既有纯数值命令指标，移除旧approach/transport/release物理阶段heuristic。

事件缺失为null，不填0/300。双侧已观测才计算时刻差；单侧事件仅在另一臂仍可见的时钟内给出“尚未发生”下界，否则标顺序未知。完整episode时长不同的计数/持续时间仅作描述；主要路径、action norm和close比例差用相同窗口或共同终点。joint/gripper公开位移既不是专家状态误差，也不提供目标距离、接触或任务阶段真值。

按三seed、全部配对及全部结果分层输出total/n/missing、均值/中位数/min/max。3例单列包含事件、step5、早期窗口和同钟终点，完整60对逐步比较全部归档。分层是事后描述，不是独立验证；不做显著性、IID、因果根因或正式SPT/SPI/Knowledge/Module结论。

## 执行与决定

[配置](../configs/libero-task1-b-routed-paired-traces-readonly_v1.json)、[分析入口](../experiments/analyze_libero_task1_b_routed_paired_traces.py)、[测试](../tests/test_libero_task1_b_routed_paired_traces.py)和本文由绑定hash锁定。默认metadata preflight，--execute才读取JSON并分析；输出/tmp/libero-task1-b-routed-paired-traces-20261104-v1.json独占创建，不覆盖原结果或失败。CPU Python3.10.22/NumPy1.22.4。

独立审计只读同一120份JSON和新结果，不导入分析实现，数值容差abs1e-10/rel1e-12，无新科学样本或模型/环境调用。相关测试覆盖共同启动、不同时间轴、缺失事件、零符号、右删失、状态距离与位移区别、同钟窗口、空分层和旧输出保护。

本项保留此前路由候选停止决定，不自动恢复训练或增预算。根据全部支持证据、反例和缺失项决定是否需要进一步受控读出；任何新训练变量须另冻协议。validation、held-out、qualification、reserved及final evaluation继续封存。
