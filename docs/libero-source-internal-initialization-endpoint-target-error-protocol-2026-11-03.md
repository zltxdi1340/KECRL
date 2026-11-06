# 固定专家输入的初始化／500点误差分解协议（2026-11-03）

本协议在本次算术分解前冻结，属于source内部diagnostic。历史训练已经评价过0/50/150/300/500点，本轮复用其中同一source-support输入的0/500点指标，**新增主前向0、重复前向0、训练0、环境交互0**。不重新读取HDF5/RGB、初态或checkpoint文件；原checkpoint身份由已固定的历史独立审计继承，不宣称本轮重新加载或重放验证。

研究动机见[六组配对轨迹报告](libero-source-internal-six-group-paired-trace-diagnostic-2026-11-03.md)。相同reset观察下首动作已有差异，夹爪时序又没有统一成功规律；先区分最终目标误差差异在适应前是否存在、以及500步适应是否扩大或缩小该差异，不改loss或预算。

## 固定范围

- task0/1、seed20261022/23/24、A_original/B_fomaml/C_pooled_bc全部保留；18个Policy的0/500点共36组指标与36份checkpoint身份固定于[配置](../configs/libero-source-internal-initialization-endpoint-target-error-readonly-diagnostic_v1.json)。
- 0点指当前探针的适应更新数为0。A来自原始未训练生成器，B/C已在该fold的其他source任务上训练；不能把三臂0点都称为未训练。500点仍是唯一执行端点，0点不参与checkpoint选择。
- 输入是该探针source `meta_support` demos0..29的全部合法`obs[t-1] -> action[t]`、t>=1。t=0排除，不重建初态。task0为2895个有效目标步/304个窗口，task1为4059步/422窗口；sequence10、stride10、batch8。
- 同观察证据来自固定的原runner：同一fold所有模型和日志点共用相同support缓存、顺序评价，无增强；CNN-only下无跨序列隐状态。历史36组读出共125172个有效模型预测行，6954个唯一专家步；这些是历史复用数量，本轮新增预测行0。
- 保留闭环稳定化5步；本轮无环境，不给离线专家观察额外执行稳定化。官方初态索引与demo编号没有对应关系。

## 指标与判断边界

主指标为平移、旋转、夹爪目标MSE；次级保留七维MSE和夹爪符号错误率（1减原accuracy）。分量分别按3、3、1维归一化；核对七维MSE等于`(3*T+3*R+G)/7`。连续夹爪MSE与符号错误率分开解释，[同号幅值差不直接改变runtime夹爪增量](libero-gripper-action-mapping-audit-2026-11-03.md)。

对每个task×seed、B-A及B-C，全部指标统一用“低为好”：

1. 初始化差：`D0 = E_B,0 - E_ref,0`。
2. 各臂适应变化：`L_arm = E_arm,500 - E_arm,0`，负值表示本臂目标误差下降。
3. 相对适应变化：`DL = L_B - L_ref`。
4. 最终差：`D500 = E_B,500 - E_ref,500`，核对`D500 = D0 + DL`，abs1e-12。

输出六组逐seed、全部三臂绝对值与上述配对分解，每个task内三seed均值/中位数/min/max及严格小于0的seed数。不合并task，不做显著性、相关性或预测有效性检验，不由均值选择seed。已有每模型20条candidate闭环成功数仅作完整上下文，不按成败筛专家输入。

这些是误差差异的代数分解。不同初始化与适应过程相互作用，不能把`DL`当作独立、因果的优化贡献；也不由它反推视觉机制、action head根因、当前自由运行动作正确性或SPT接受。原汇总没有逐帧预测，不能据此给出命令切换/阶段分层或图像对齐；若确需这些数据，须另冻前向预算。

## provenance与停止规则

固定原训练结果、原配置/绑定、历史独立审计、transfer/数据manifest/学习配置、现有六组轨迹分析及其审计、夹爪映射审计共10个输入SHA。执行代码、测试与本文另绑定后才运行算术分解。验证仅打开这些JSON和绑定的代码/文档；checkpoint路径和hash是继承的元数据，不打开其指向文件。

缺失指标、非有限值、负MSE、accuracy越界、角色/矩阵/有效步数/窗口数不符、hash不符或代数恒等式失败立即停止。空值不补零，旧产物不覆盖，不以结果好坏重试或改变规则。保留已有训练预算、模型、loss和观测；不追加训练，不更新活动SPT/Knowledge，不注册Module，validation、held-out、qualification继续封存。

本文件记录执行前定义，执行结果和实际provenance另写dated report；冻结设计不称为formal result。
