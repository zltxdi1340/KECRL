# LIBERO受限视觉微调对照：准备与科学冻结记录（2026-11-14）

**方案、模型、独立入口/核对器和合成验证已准备，科学冻结完成；真实执行尚未绑定，没有本轮效果结论。** LIBERO继续作为主环境，不开展Continual World。详细设计以[冻结协议](libero-restricted-vision-finetune-protocol-2026-11-14.md)及[科学配置](../configs/libero-restricted-vision-finetune-diagnostic_v1.json)为准。

## 决策依据与固定对照

[前一轮](libero-pretrained-vision-control-results-2026-11-14.md)冻结ResNet较CNN增加6.67个百分点，但未过原10个百分点门，task7收益仅1/60。新问题是：同一架构和初始化下，开放最后视觉阶段能否在固定500更新内进一步改善完整控制。它是待检验假设，不能把前一轮结果写成视觉瓶颈已证实。

| 项目 | 本次固定范围 |
| --- | --- |
| 两臂 | 新`frozen_resnet18`控制 / 新`layer4_conv_finetune`候选 |
| 配对 | task4/7 × seeds20261106/07/08；共12个Policy，同全部初始tensor、同500批次 |
| 候选可训练视觉范围 | 仅layer4的5个卷积weight tensor，新增8,388,608参数 |
| 其余冻结范围 | stem、layer1..3、所有BN affine与统计量；BN保持eval |
| 可训练容量 | 控制1,513,352 / 候选9,901,960；同总参数12,689,864 |
| 训练 | 从公共初始化各500更新，不能从旧u500续训；原Adam/clip/MSE和support角色保持 |
| 未来预算 | 6,000次更新；若另获本轮授权，240条fresh episode、5步稳定化、最多300动作/条 |
| 原继续门 | 新候选−新控制≥10个百分点，且每task/每seed均值不退化 |

“受限”指训练范围，新增8.39M参数并不属于轻量适配。两臂训练容量、梯度裁剪频率和计算成本可能不同，均须如实报告。旧16/120不是本轮新控制数据；全部120配对四格、每task×seed及旧成功保留率必须展示，不能只报总体均值。

## 实现与验证

- [新模型](../src/algorithms/libero_restricted_resnet18_policy.py)：复用原初始化，保持所有state key和初始值；两臂只缓存layer3 `[256,8,8]`，每步重算layer4并保留候选梯度。拒绝旧最终特征缓存，冻结前缀用no_grad，BN eval与卷积requires_grad分开控制。
- [协议与角色门](../experiments/libero_restricted_vision_finetune_protocol.py)、[新入口](../experiments/run_libero_restricted_vision_finetune.py)：分别限制训练和闭环操作，另一阶段cap为零；先检查新版本执行绑定，闭环还须本轮新授权且科学配置pin相同。训练只用task4/7 support demos0..29、t≥1，query和正式角色继续封存。
- [独立核对器](../experiments/audit_libero_restricted_vision_finetune.py)：检查完整模型/episode身份、每模型500步Adam状态、完整批次和裁剪记录、不可训练tensor/BN与公共权重、缓存、初态解码范围、reset配对、原生终止、实际预算和Fraction继续门。它不重训、不重放物理环境。

143项定向测试通过：新模型/协议/流程52项，相邻视觉/BC/初态reader等57项，FOMAML和公共执行账本34项；0失败/错误/跳过。实际ResNet合成测试检查初始化逐位配对、五卷积梯度/更新、BN与冻结参数不变、缓存/原像素动作及梯度一致、预准入停止。完整流程用小型人造模型真实执行12×500次Adam更新，并用假物理环境完成240个episode；公共骨干核对在该小模型流程中显式替换，真实ResNet tensor核对在单独测试中运行。它们都是软件验证，不是LIBERO研究结果。

编译、CPU环境检查及差异检查通过。未跑全量pytest、真实CUDA训练或LIBERO物理闭环；环境检查有意隐藏CUDA。依赖源码均为新增版本，未改公共训练内核或旧冻结文件。

## 科学绑定与证据

- [科学冻结绑定](../configs/libero-restricted-vision-finetune-freeze-binding_v1.json)：573项代码/元数据/验证依赖，状态`scientific_frozen_execution_unbound`，执行与初态授权均为false。
- 科学配置SHA：`48868dfe74a51e31e571e9efc58372299f9bc9e8fe283e498f953be39a42bce0`。
- 科学绑定SHA：`6e63ec3a656196480cb4f896d4956f3eeb3a950ad73259d1c9f5bc29729cf397`。
- 日志：`logs/libero-restricted-vision-finetune-preparation-20261114-v1/`，含准备前完整性、模型/流程初验、109项focused与34项shared测试XML/正文、环境检查及最终完整性收据。
- 本轮原始研究payload读取/SHA、真实训练更新、真实episode均为0；真实公共权重和原数据pin仅从前一版配置复制，不重新打开或校验payload。

上轮556项执行依赖和历史配置/结果保持原样；保留其他未提交修改、`.vscode/`及无关phase-gripper报告修改。没有Git提交或push，没有下载、清理或重建旧实验。

## 下一步

先在本版本配置下建立训练执行绑定，核验科学冻结、测试证据和运行环境，再进行source-support训练及独立训练核对。两者入口为`python -m experiments.run_libero_restricted_vision_finetune --stage train`与`python -m experiments.audit_libero_restricted_vision_finetune --stage training`，必须传本轮`--config`、新的`--binding`并按配置设置运行环境；目前训练绑定文件不存在，不能直接运行。

闭环须单独取得本轮task4/7已用source初态0..19授权，再新建评估执行绑定；上一轮授权不转用，其他任务/角色不扩用。之后才可执行240条并核对全部结果。通过开发继续门才准备全source验证，未通过仅收束本版本并继续LIBERO；不自动准入SPT迁移或Module，不改变Knowledge。500更新、稳定化5步、现loss及正式数据封存保持。
