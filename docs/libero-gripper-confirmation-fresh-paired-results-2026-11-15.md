# task7确认门：当次新配对真实结果v2（2026-11-15）

**真实240条fresh配对已完成，独立审计通过；原命令13/120，confirm5为14/120，净变化+0.83个百分点。** 未达到本次source开发继续门，停止本confirm5候选。

本结果来自本版本独立的新240条（120对），不混入旧v1完成的179条。使用原6个Policy500完整checkpoint，两类模型、三个seed、已用task7初态0..19；每个模型均比较当次unchanged与confirm5，不训练或扫参数。详见[事前冻结协议](libero-gripper-confirmation-fresh-paired-protocol-v2-2026-11-15.md)。

## 实际效果与完整分层

| 原模型类别 | 原命令成功 /60 | confirm5成功 /60 | 变化（百分点） |
| --- | ---: | ---: | ---: |
| frozen_resnet18 | 9 | 8 | -1.67 |
| layer4_conv_finetune | 4 | 6 | +3.33 |

| seed | 模型类别 | 原命令 /20 | confirm5 /20 |
| --- | --- | ---: | ---: |
| 20261106 | frozen_resnet18 | 1 | 1 |
| 20261106 | layer4_conv_finetune | 3 | 6 |
| 20261107 | frozen_resnet18 | 6 | 6 |
| 20261107 | layer4_conv_finetune | 1 | 0 |
| 20261108 | frozen_resnet18 | 2 | 1 |
| 20261108 | layer4_conv_finetune | 0 | 0 |

120对：两条件均成功7，仅原命令成功6，仅确认门成功7，两条件均失败100。原成功保留率为7/13。完整计入损失，不按成功案例选择模型或初态。

事前继续门要求总体净增至少12/120、每类模型均值不退化、每seed均值不退化；逐门实际值：

- `overall_gain_at_least10pp`：`false`。
- `each_policy_arm_nonregressing`：`false`。
- `each_seed_nonregressing`：`false`。

## 配对、历史参照与旧失败

当次120/120对reset摘要完全一致，120/120对初始raw动作完全一致；6个模型在18次前后检查中权重不变。unchanged逐步原样执行raw，confirm5由独立标量审计重算，其他6维保持本步raw，环境反馈改变后的后续raw允许变化。

历史轨迹仅为事前明确的描述性参照：完全一致118/120；动作有差异的episode 2，有差异的动作步292；公开状态不同步292；终止步与成败均匹配120/120。历史动作最大绝对差0.158429235。没有强制新控制组等于旧成功数。

旧v1在seed20261108/冻结模型/初态19/第129步发生历史动作不符而技术中止，最大动作差5.544722080230713e-5；前128步保存的动作/公开状态完全匹配。原后续RGB缺失，具体来源未确认。旧失败、零容差门、预算、代码/配置/SHA均保留；本v2不是让旧v1通过，而是另冻的当次配对研究。没有增加epsilon、删去该案例、重用旧部分成败或换确认步数。

## 运行、审计与证据

主运行实际240条、68319次Policy前向/动作、69519次native step（含1200次固定稳定化零动作）；12环境、240reset、20行初态单次解码、6次完整checkpoint读取、18次权重检查。内部执行1308.016秒；反向/优化/HDF5/初始化权重读取/额外render均为0。

独立审计1024115项核对通过，重读120份历史JSON和240份新episode；不读checkpoint/初态，不运行模型或物理。主/审计各读历史26041679字节。新绑定663项依赖，161项定向测试通过；最后新增10项复核及4种不符授权拒绝均通过。编译、环境及差异检查通过，未跑全量pytest。

[科学配置](../configs/libero-task7-gripper-confirmation-fresh-control_v2.json)、[执行绑定](../configs/libero-task7-gripper-confirmation-fresh-execution-binding_v2.json)、[本轮授权](../configs/libero-task7-gripper-confirmation-fresh-authorization_v2.json)。结果、实际账本、配对episode、初态访问收据及审计位于`results/libero-task7-gripper-confirmation-20261115-v2/`；日志、旧失败只读核对、测试与完整性检查位于`logs/libero-gripper-confirmation-real-20261115-v2/`。

## 解释边界

确认门同时延后持续合爪并过滤短正脉冲；本次只能评价这一固定规则，不能区分两种贡献，或证明/排除合爪时机是全部失败原因。前两步画面只有1例明确早请求、15例未知，不能把所有微调失败归因此。未过门时停止此confirm5，不扫描更多步数救当前版本。

这是反复使用过的source开发初态上的机制对照，不是正式迁移或完整KECRL验证。Policy500、稳定化5步、现loss保持，query/meta_query/validation/H/heldout/qualification/reserved/final继续封存。不更新SPT/Knowledge或注册Module，LIBERO继续作为主环境。

生成UTC：2026-11-15T02:08:34.261175+00:00。
