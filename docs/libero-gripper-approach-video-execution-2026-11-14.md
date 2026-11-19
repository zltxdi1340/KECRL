# LIBERO夹爪时机视频诊断：执行技术中止（2026-11-14）

**本次建立新执行绑定后实际运行，在第三段reset摘要不符时停止；只完成2/30段，没有得到夹爪时机结论。** 未重试、跳过案例或放宽一致性门。此结果说明重放尚未复现原流程，不能解释为“夹爪时机没有问题”，也不改变此前冻结臂16/120、微调臂12/120的闭环结果。

本轮承接用户“继续下一步”，范围仍为[已冻结协议](libero-gripper-approach-video-protocol-2026-11-14.md)的15对/30段、每段100个保存动作、10个指定开发初态。新增[本轮授权](../configs/libero-gripper-approach-video-source-initial-authorization_v1.json)及[执行绑定](../configs/libero-gripper-approach-video-execution-binding_v1.json)在初态读取之前写入，核验599项依赖及准备阶段77项测试证据；未转用已经消耗的旧240条授权。

## 实际执行与停止点

| 项目 | 实际完成 / 冻结上限 |
| --- | ---: |
| 完整保存动作前缀 | 2 / 30；第三段已尝试但未通过reset门 |
| 环境构造 / reset | 各3 / 30 |
| 保存动作 | 200 / 3000 |
| 稳定化零动作 | 15 / 150 |
| 受控native step | 215 / 3150 |
| 初态单行加载 / 数值解码 | 各10 / 10 |
| 已保留公开观察 / 两相机帧 | 202 / 3030；404 / 6060 |
| 已保留原始RGB字节 | 19,857,408 / 297,861,120 |
| 采集阶段原episode JSON解析 / 字节 | 3 / 30；650,514 / 64MiB |
| Policy前向、反向、优化更新、模型构造、checkpoint/HDF5读取、额外render | 全部0 |

内部采集计时8.696秒，不含进程启动和后续只读导出；GPU1/EGL1、原包版本、环境seed20261022、128×128双视角和5步稳定化保持。

已完成的是`case00`：task4、seed20261106、global初态0的两臂各100步。按冻结顺序，第三段为`case01/A`：task4、seed20261106、global初态7、冻结臂。它在5个稳定化零动作之后、执行任何保存动作之前报错：

```text
ValueError('source reset RGB/proprioception digest differs')
```

原始证据为`results/libero-gripper-approach-video-20261114-v1/capture-failure.json`及`logs/libero-gripper-approach-video-execution-20261114-v1/capture-console.txt`。没有生成完整`capture.json`、`execution.json`或完整捕获审计通过收据。

## 两段保留画面的有限复核

停止后只处理已经写出的两份NPZ，不再打开初态、构造环境或推进物理步。另重读case00两份原episode JSON，逐位核对200条动作与202个joint/gripper状态，核对reset摘要，重算新画面观察摘要；共3420项坐标/摘要检查。新算摘要明确标为失败后重算，不能冒充丢失的采集记录。后续RGB没有原逐帧对照，不能称原视频已恢复。

- 原始两份NPZ保持字节不变；部分前缀收据为`partial-prefix-audit.json`。
- 导出两路并排MP4和0/100帧上下文拼图；只读媒体审计实际解码各101帧、20fps、768×416，拼图768×832。原画面仍为128×128，放大不增加信息。
- 两段前100步都没有`action[6]>0.1`，没有可判读的合爪前一帧。两条记录均为`not_assessable`，不等于全程从未合爪。
- **没有采集到task7画面。** 未对task7“54/60对更早合爪”作视觉验证，也未估计新成功率或因果效果。

[部分画面浏览页](../results/libero-gripper-approach-video-20261114-v1/partial-review.html)明确标示“技术中止、仅两段、无close事件”。所有导出和收据均使用`partial-*`名称，是对失败后既有文件的单独复核，不是原30段完整媒体阶段通过。

## 已发现的流程与记录缺口

原[闭环执行器](../experiments/libero_formal_rollout.py)对每个模型构造一次环境，在同一环境内按0..19顺序reset并运行各完整episode。本次[视频采集器](../experiments/run_libero_gripper_approach_video_diagnostic.py)对每段重新构造环境，只加载选中的紧凑初态数组；因此初态7成为新环境的第一次reset。两者的环境复用和前序运行历史不同。合成替身没有验证真实环境是否受该差异影响，这是本次重放设计与验证的不足。

**此差异是排查线索，尚非已确认根因。** 初态0的两段成功匹配、首个非零初态7失败，仍不足以证明环境历史是唯一原因。失败日志只保存了组合摘要不符，没有保存当次失败reset的公开数组/分项摘要，无法从现有文件区分RGB差异还是本体状态差异。异常分支还未保存已加载初态的访问收据和前两段完整采集metadata；因此部分NPZ复核不能替代完整来源审计。不得补造这些缺失记录。

另一个纯后处理调用因脚本导入路径失败，原日志保留；在显式设置`PYTHONPATH=.`后只读导出成功。它未推进环境，也不是采集重试。

## 下一步：先核对reset复现，再回到夹爪问题

本v1保留`technical_capture_failure`，不从剩余预算续跑。建议另行准备一个只检查reset的受限对照：同一task4初态7，一路在新环境直接reset7，另一路在同一环境依次reset0..7；各只做原5个零动作，不运行Policy或保存动作。候选上限为2个环境、9次reset、45个native零动作、task4的8行初态解码，比较原初态7 reset摘要及可获得的公开数值，同时在失败时保留分项证据和真实访问收据。

**这只是下一版方案，尚未绑定、授权或执行。** 它会新增初态3/4/5/6的访问和数值读取预算，不能借用本轮10行授权；即使顺序reset匹配，也不能证明完整原动作轨迹均可复现。若两路都不匹配，继续定位初始化/渲染差异；不盲目重跑30段。修正需版本化，保留本v1代码、绑定和失败证据。

本轮没有修改训练行为，不以技术中止决定放弃LIBERO或改变研究假设。Policy500、稳定化5步、现loss、正式数据封存及SPT/Knowledge/Module边界保持。

精简可追溯清单见[执行摘要](evidence/libero-gripper-approach-video-execution-summary-2026-11-14.json)。本轮新增授权/执行绑定、失败分析与部分媒体；无采集源码修改。环境检查已完成，准备阶段77项测试证据重新核验，未重跑全量pytest；真实失败明确表明这些软件测试不足以证明物理重放一致。最终冻结依赖和工作区保留检查记录在`logs/libero-gripper-approach-video-execution-20261114-v1/final-integrity.json`。
