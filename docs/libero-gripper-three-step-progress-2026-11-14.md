# LIBERO夹爪三步推进：执行进展（2026-11-14）

**前两步已完成，第三步已冻结并绑定，正在等待GPU1空闲。** 这里记录等待阶段的状态，不代表240条确认门闭环已经完成。

1. task4初态7两臂各100步原动作复现：2/2通过，280native step，3294项独立核对通过。
2. task7固定8对/16段画面：16/16通过，2060native step，26068项独立核对通过；1段可见对准前的短合爪请求，15段无法可靠判定。明确例来自冻结臂，不能归因于视觉微调。详见[前两步结果](libero-gripper-approach-visual-results-2026-11-14.md)与[视频](../results/libero-task7-gripper-approach-video-20261114-v2/review.html)。
3. 固定6个原Policy500 checkpoint的确认门对照：[协议](libero-gripper-confirmation-control-protocol-2026-11-14.md)、[配置](../configs/libero-task7-gripper-confirmation-control_v1.json)、[651项依赖执行绑定](../configs/libero-task7-gripper-confirmation-execution-binding_v1.json)已完成，用户“按三步推进”在本阶段另建[240条新闭环授权](../configs/libero-task7-gripper-confirmation-authorization_v1.json)。

确认门固定为正请求先保持5步、第6步仍为正再放行，非正请求立即通过；不扫参数或训练。两类模型×三seed×已用task7初态0..19×两条件，共120对/240条。原unchanged条件须完整复现原轨迹，模型权重不得变化。科学继续门事前固定为平均净增≥12/120，且每类模型、每seed均值不退化；保留全部成败配对。

151项定向测试通过，含人造240条/72000前向最大包络、原模型完整state重建输出精确一致、确认门/停止/篡改及元数据权限检查。执行前曾将actual_optimizer_steps的逐参数列表误读为整数；已按旧独立审计恢复“列表长度匹配参数名、每项恰为500”的检查并补3个拒绝测试，研究payload尚未打开，未把该适配错误当实验结果。日志保留在`metadata-schema-correction.json`。

GPU1已有其他任务，当前不占用其计算资源；等待器每15秒检查，最多等待3600秒，空闲后只启动这一次已绑定运行，完成后只运行一次独立审计。原生运行或审计失败不重试；等待超时则保持未启动，不伪造失败结果或成功收据。已取得的公开画面不作为“确认门提高成功率”的证据。

运行与等待日志：`logs/libero-gripper-three-step-20261114-v1/`；第三步目标输出：`results/libero-task7-gripper-confirmation-20261114-v1/`。等待事件追加到`confirmation-job-events.jsonl`，真实运行时另写`confirmation-run-console.txt`；完整研究结果只有`completion.json`与独立审计均通过后才可报告。

Policy500、稳定化5步、现loss及正式数据封存保持；不注册Module或更新SPT/Knowledge，LIBERO继续为主环境。前两步实际1800条保存动作/540零动作，共2340native step；没有模型前向或训练。第三步实际计数须以其真实账本为准，不用计划上限代替。
