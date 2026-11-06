# task1 B运动历史／夹爪当前配对读出协议（2026-11-04）

用户授权重跑后，先完成[固定头交叉读出](libero-task1-b-crossed-history-readout-diagnostic-2026-11-04.md)：同历史头下重复当前纠正原6个新增错号，但整体运动拟合变差。该结果提出单变量路由假设，本协议在新训练与新query数值读取前冻结。它不是已确认根因或原完整历史候选的挽救；原严格事件改善门槛失败保留。

## 输入、配对与角色

只使用task1 B_fomaml三seed20261022/23/24既有500-update编码器的缓存表示，不重训CNN或原Policy。source-support demos0..29；query30..49已暴露，本设计也受其诊断启发，不能宣称完全独立设计、未见泛化或正式验证。validation、held-out、qualification、reserved和final evaluation继续封存；不读取HDF5、RGB、初态、private state或原checkpoint payload。

新读出两臂：current_all7全部7维使用重复当前；history_motion_current_gripper运动6维使用当前+过去4帧，夹爪1维使用重复当前。两臂均Linear(1280,7)+tanh、8967参数、全零初始化、同seed、同batch/目标/scaler与均匀七维action MSE；单一变化为运动输出的输入路由。夹爪仍参与相同全局clip1，优化耦合可能改变其权重，不保证夹爪预测与baseline不变。

目标均为causal pre-action的t>=5，support3939行、query2518行/seed。历史不跨demo、不填未来帧。scaler只用fit示教合法当前表示，float64 mean/std，std<=1e-6置1，应用所有lag后拼接除sqrt5。编码器先前训练过全部support示教，下面三折只隔离新读出的fit/probe，不能称原Policy未见示教。

## 冻结训练与支持集决定

先做3个整demo probe折0..9/10..19/20..29，每折fit剩余20个。每seed/折/两臂从零独立训练500 updates，共18个新读出、9000更新；np.default_rng(seed+100000*fold)在fit位置有放回采样500×64，两臂完全共享。记录fit/probe 0/100/250/500曲线；主分析是拼接互斥probe帧的OOF固定500预测，不选择折、端点或seed。

Adam lr0.0003、betas(0.9,0.999)、eps1e-8、weight_decay0、batch64、global gradient norm clip1；optimizer设置与实际Adam step/norm/loss都保存。**全部三seed的OOF需同时满足**：open事件错误数不增加，完整已知open错误数不增加，整体平移和七维MSE严格下降。这一新问题允许夹爪持平，不能据此追认旧完整历史候选的严格事件改善门槛已通过。

若支持门槛失败，完整support训练/query读出为0，保存所有反证并停止该路由。若通过，才从零重训三seed×两臂全部30个support示教，共6个新读出、3000更新，每模型仍500；全support采样seed直接使用training_seed，曲线只记support。

全部6个完整fit及归档完成、实际总12000更新后，才读取query目标和三个query feature NPZ数值。query只评价固定500端点一次，使用各自support scaler；不评中间query点，不按query调loss、历史长度、预算或选checkpoint。相同的三seed联合非退化/运动改善规则报告为新query cue，不自动授权闭环。

## 预算和数值控制

| 项目 | support停止时 | 支持通过并完成query时 |
|---|---:|---:|
| 新读出模型 | 18 | 24 |
| optimizer/backward/训练模型前向 | 各9000 | 各12000 |
| 训练有效行 | 576000 | 768000 |
| 训练primitive Linear调用 | 13500 | 18000 |
| support指标前向/有效行 | 72/283608 | 96/378144 |
| support指标primitive Linear调用 | 108 | 144 |
| query模型前向/有效行 | 0/0 | 6/15108 |
| query primitive Linear调用 | 0 | 9 |
| 原编码器/Policy更新/HDF5/环境/初态/private调用 | 0 | 0 |

baseline每次前向1个Linear；候选分别计算历史和当前的Linear，取前6维及最后1维后tanh，每次模型前向2个primitive Linear，不把参数量等同计算量。所有实际计数按两种停止分支精确校验。新的训练只作用于小型监督读出，不是完整Policy重训或SPT更新。

12个baseline（三折9个+完整3个，若到达）必须精确重现已有同设定读出的全部归档数组，包括权重/预测/采样/scaler、loss/norm和Adam缓冲；不一致立即停，不以宽容差修改训练。query baseline预测容差1e-5、夹爪符号相同。独立审计的归档预测数值容差1e-5、指标1e-10；新增算术重放明确单列。

固定CPU单线程、CUDA_VISIBLE_DEVICES为空，Python3.10.22、Torch2.4.1+cu124、NumPy1.22.4；运行前仍记录GPU占用。小读出沿用旧CPU路径保证配对和复现，不做设备自动回退；不用空闲GPU缩短本来很小的拟合。原500更新、闭环默认稳定化5步和原loss保持，本轮环境交互0。

## 执行、归档与停止

[配置](../configs/libero-task1-b-routed-history-readout-diagnostic_v1.json)、[代码](../experiments/run_libero_task1_b_routed_history_readout.py)、[测试](../tests/test_libero_task1_b_routed_history_readout.py)及本文由绑定hash锁定。预先核对所有直接输入及父绑定；默认仅metadata preflight，--execute才训练。所有输出独占创建，旧失败/结果不覆盖。数值非有限、角色/hash、实际预算、baseline复现失配立即停止，保存完成模型、当前部分权重和更新计数，不补跑或放宽阈值。

默认结果`/tmp/libero-task1-b-routed-history-readout-20261104-v1.json`与同名前缀artifact目录，包含全部训练权重/四点预测/曲线/500步loss和norm/最终Adam缓冲、fit/probe/sample行ID、scaler、输入route摘要、OOF和query预测、事件时间线、配置/代码/数据hash和设备成本。所有fit完成时间与query读取时间可核对。

不追加训练变量、闭环、SPT/Knowledge更新或Module注册。即使本项通过，也只说明固定表示上的运动读出改善且夹爪指标未退化，不说明必要记忆、控制成功或正式迁移；下一步须独立检查runtime历史启动规则和合法逐步输入接口，再另冻candidate闭环预算。
