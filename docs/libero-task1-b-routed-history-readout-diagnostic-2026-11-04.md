# task1 B运动历史／夹爪当前配对训练结果（2026-11-04）

**已完成新路由候选与baseline配对重训。** source-support三折与source meta-query的全部三个seed均满足新冻结条件：平移/七维MSE下降，open事件及完整open段错号不增加。query三seed均值平移0.049107→0.043379（−11.66%），七维0.049108→0.046585（−5.14%）。但事件错号仍为11/7/10、共28/69，夹爪切换没有修复；这只是固定表示监督读出的离线诊断。

## 为什么改、改了什么

[同头交叉读出](libero-task1-b-crossed-history-readout-diagnostic-2026-11-04.md)显示：固定历史头时，用重复当前表示纠正原6个新增错号，却使整体运动拟合变差。于是另冻单变量路由，运动6维保留历史，夹爪1维只读当前；不同时改loss、历史长度、updates或选seed。

在新训练前冻结[配置](../configs/libero-task1-b-routed-history-readout-diagnostic_v1.json)、[协议](libero-task1-b-routed-history-readout-protocol-2026-11-04.md)、[绑定](../configs/libero-task1-b-routed-history-readout-binding_v1.json)。baseline=current_all7，候选=history_motion_current_gripper；这不是旧baseline/jitter。固定task1 B_fomaml三seed20261022/23/24的500-update编码器缓存；两臂均8967参数、全零初始化、同scaler/采样/目标、均匀七维MSE及原Adam/clip1，每个新读出500 updates。

support demos0..29、query30..49。先做整demo三折：0..9/10..19/20..29轮流probe，fit另20个；18个模型共9000更新。所有OOF条件通过后，从零以全部30个support示教重训6个模型、3000更新。全部24个模型和六个完整fit写入归档后，才读取本轮query缓存目标/表示，仅检查固定500端点。

这次训练只更新读出头；原CNN/Policy没有更新。编码器以前已训练过全部support，三折仅隔离新读出的fit/probe。本设计也受到已暴露query诊断启发，query只作重复source检查，不能当未见/独立最终评价。

## 固定500端点结果

query共同t>=5，每模型2518帧、23个已知open事件、232个完整已知open段帧；同帧配对，不跨demo取历史。下表每项均为baseline→候选。

| seed | 平移MSE | 七维MSE | 事件错误/23 | 完整open错误/232 |
|---|---:|---:|---:|---:|
| 20261022 | 0.049650→0.043324 | 0.047220→0.044401 | 11→11 | 48→48 |
| 20261023 | 0.051194→0.044255 | 0.052335→0.049280 | 7→7 | 37→37 |
| 20261024 | 0.046475→0.042558 | 0.047769→0.046075 | 10→10 | 54→54 |
| 三seed均值/合计 | 0.049107→0.043379 | 0.049108→0.046585 | 28→28 / 69 | 139→139 / 696 |

完整query的平移、旋转、七维MSE三个seed均下降。夹爪整体MSE均值0.189072187→0.189069809，变化极小；事件夹爪MSE三个seed反而略增加+0.000022/+0.000185/+0.000250。所有2518帧/seed的夹爪符号完全相同，输出值并非逐位不变，最大差分别0.000133/0.000408/0.000222。相同夹爪输入不保证相同训练权重，仍有全局clip优化耦合；不把这些小差写成切换改善。

支持集OOF固定500结果同样保留全部seed，事件/完整open错号未增加，平移及七维误差均下降。support每seed34事件、459完整已知open帧。

| seed | OOF平移MSE | OOF七维MSE | OOF事件错误/34 | OOF完整open错误/459 |
|---|---:|---:|---:|---:|
| 20261022 | 0.052473→0.047243 | 0.048102→0.045834 | 23→23 | 109→109 |
| 20261023 | 0.056479→0.048076 | 0.053731→0.050127 | 23→23 | 118→118 |
| 20261024 | 0.050840→0.046992 | 0.048474→0.046809 | 21→21 | 107→107 |

support_comparison与query_comparison的joint_cue均true。新路由问题允许夹爪持平，要求动作拟合改善；**旧完整历史候选要求事件严格改善的门槛失败仍保持**。不拿新规则追认旧结果，也不从旧全历史与新路由之间按query最小MSE挑模型。

## 实际预算与provenance

实际24个新模型各500、共12000 optimizer updates/backward/训练模型前向，768000训练样本行；候选每次模型前向包含两个Linear调用，总18000个训练Linear调用。support曲线0/100/250/500共96个指标模型前向、378144行；query固定500共6个模型前向、15108行。另有144个指标与9个query Linear调用。与冻结分支预算完全一致，无训练失败、补样本或重试。

固定CPU单线程，Python3.10.22、Torch2.4.1+cu124、NumPy1.22.4。运行前GPU0有其他作业，GPU1约259MiB/0%；为精确沿用既有CPU小读出训练路径，预先固定CPU并隐藏CUDA，不做设备回退。入口总耗时9.4905s，其中24个模型循环合计6.5087s，峰值RSS774784KiB。训练耗时短主要因为复用256维表示、每头仅8967参数，不是完整CNN训练或物理交互预算耗时。

新原编码器前向、原Policy更新、HDF5/RGB/初态/private/原checkpoint payload、环境调用均0；仅读取既有source缓存。六个完整fit完成与query缓存打开的时间顺序保留在events.jsonl。不读取query中间端点，不改seed、loss、稳定化5步或checkpoint规则；validation、held-out、qualification、reserved及final evaluation继续封存。

归档保存全部24份读出权重/四点预测/fit-probe曲线、500步loss/norm、最终Adam缓冲、采样ID和scaler；三份OOF与六份query预测，以及配置/输入/代码hash和设备记录。12个baseline（三折9+完整3）的全部旧归档数组逐位复现，三个query baseline亦精确复现；不是只比较最终MSE相近。

## 验证与审计修正

178项专项/相关训练回归通过，包含路由梯度隔离、旧optimizer逐位复现、整demo划分、历史因果索引、query门槛、零调用失败归档、scope/hash与旧输出保护；4份新Python文件编译、CPU环境检查通过。没有运行需要可选Crafter依赖的全量pytest。

独立审计v1将已有`initial_open`标签误写为`initial_open_prefix`，在首模型update0的指标schema检查处停止；完成1批3939行算术重放。失败记录和原脚本保留。仅将审计字段改为canonical标签后另存v2，没有修改训练配置、代码、数值阈值、历史产物或重新训练。

v2不导入项目实现，从原归档独立重建scaler、因果历史、采样、四点预测、五类指标、三折OOF、固定query、两阶段gate与事件时间线。4052292项检查通过，全部24模型、12个配对及12个baseline验证通过；102个审计算术重放/393252行、153个Linear算术调用单列，最大预测差6.78e-7（冻结容差1e-5）、指标容差1e-10。审计没有新增研究样本、optimizer、编码器、HDF5或环境调用。

| 证据 | SHA-256 |
|---|---|
| 科学配置 | `d8f601227ae3052fee78ecfbcff2106477e74d2893e4f0b4fcd1fbfaedd99b81` |
| 执行绑定 | `f15b4303bfd3e7a5c2c81c24cc7ad64dfbd63637404dbeb880cbd76fd4d31e67` |
| /tmp/libero-task1-b-routed-history-readout-20261104-v1.json | `4ab522b340662751779f4c0ef3735847a781691b7a832c74f6a2029ddbd55304` |
| /tmp/libero-task1-b-routed-history-readout-independent-audit-20261104-v2.json | `1dcc7b43f1ae6a0f0bd3c3dae9b1dee6e383ac20ac216131f4ec577214b737a7` |
| 审计v1失败记录 | `ed2d9c66e329d98efa96c6da2360b96756445fa0a42415f0891363a84cb6afbc` |
| 审计v2脚本 | `733210b1b49706253d17b6d13a596e4b020b3988e73f06112f9c3402297204d3` |
| /tmp/libero-task1-b-routed-history-readout-summary-20261104-v1.json | `ce9b52c1746d78e30b30c56a21963454999c1b4484ab5b028f70422e80b14463` |

## 当前结论和下一步

本次支持“在固定表示读出层，可保留历史对运动拟合的收益，同时避免完整历史头带来的额外切换错号”这一受限结论。候选并未纠正baseline原来的28/69事件错误，也未证明实际控制成功。物理阶段、必要记忆、跨任务迁移、SPT或Knowledge机制均未由本项识别。

停止继续扩大离线loss/预算搜索。下一步先固定这6个完整support的500端点，冻结runtime历史缓存、同一pre-action观测的256维表示接口、前4步启动/清空规则及与缓存的数值对齐检查；通过接口核对后，再另冻task1三seed两臂的candidate配对闭环（建议现有0..19、稳定化5步、同环境seed/顺序/horizon）。本次闭环episodes为0，尚无新成功率，不能把建议的下一预算写成已执行。

所有已有工作区、AGENTS.md和.vscode保持；新增代码/配置/协议/报告及README、roadmap、后续计划更新限定在本任务。未提交或推送，原Policy/SPT/Knowledge/Module均未更改。
