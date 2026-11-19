# LIBERO动作复现与task7合爪画面：前两步结果（2026-11-14）

**task4两臂100步核验通过；task7固定8对/16段全部核验并补录完成。16段人工查看中1段为可见对准之前的合爪请求，15段因遮挡/深度不清保留无法判断。** 该明确案例为case09/B、冻结臂、seed20261106、初态0；不是微调独有证据，也没有证明其失败由此造成。

使用[顺序reset执行契约](libero-sequential-replay-execution-contract-2026-11-14.md)及新源码、配置、授权与执行绑定，没有续跑旧失败v1。第一阶段实际2环境、16reset、200保存动作、80零动作/280native step、8行初态；采集17.977秒、3294项独立核对通过。第二阶段实际16环境、92reset、1600保存动作、460零动作/2060native step、15行初态；采集111.190秒、26068项独立核对通过。原动作、每步公开joint/gripper、reset摘要及窗口内终止一致；未读模型、HDF5或正式封存域。

全部18段的MP4及事件图已导出；媒体实际解码各101帧、20fps、768×416，原始128×128画面无损保存在NPZ。130项定向测试通过，包括人造完整两臂和task7最大预算。两阶段分别绑定626/631项依赖，完整运行日志在`logs/libero-gripper-three-step-20261114-v1/`。软件测试和采集审计不代替视觉判断，也不证明私有物理状态等同。

## 判读与命令语义

case09/B帧2的宽视角显示夹爪与炉上目标碗明显分离，手眼画面仍覆盖较大桌面区域、目标位于远处；第3步请求合爪，第4步又请求开爪。此例可记`clearly_before_visible_alignment`，证据帧为2/3/4/8。其余15段均查看双视角事件图，目标虽可见，但指尖遮挡和垂直间隙不足以可靠判定对准/接触，记`not_assessable`；没有推断它们均正确或均过早。

本机PandaGripper实现累积`sign(action)`，正命令收拢、负命令张开、零命令保持原目标；旧分析的`action[6]>0.1`仅是事件锚点，不是原生开合死区。16段中4段的首次正命令比0.1锚点早1步，其余12段相同。旧锚点与冻结记录不修改；新增`sign-command-anchors.json`并绑定实际gripper源码。明确早请求的case09/B两种锚点同为第3步，因此该例判断不依赖阈值差异。

- [task7并排视频与事件图](../results/libero-task7-gripper-approach-video-20261114-v2/review.html)。
- [16条复核记录](../results/libero-task7-gripper-approach-video-20261114-v2/annotations-reviewed.csv)，observer为assistant人工双视角事件图判读；作者已知案例结果，不能称盲测。
- 原始NPZ、真实初态访问收据、观测JSONL、capture、独立审计和completion位于`results/libero-gripper-prefix-replay-20261114-v2/`及`results/libero-task7-gripper-approach-video-20261114-v2/`。
- 视觉记录审计只核验覆盖、锚点与证据字段，不认证标注正确或证明因果效果。

## 第三步的依据与限制

明确的早合爪短脉冲足以支持一次有界确认门试验，但不能把全部失败解释为合爪过早。下一步两类模型均固定原500更新checkpoint，对比原命令与单一5步正号确认门；开爪立即通过，前5个连续正命令改为零，第6步仍为正才原样放行。该规则同时延后持续合爪并过滤短正脉冲，若有收益不能区分两种贡献。

第三步须另冻240条task7 source开发闭环及新模型/初态/调用预算，先实现、测试和绑定才读checkpoint或初态。仍使用已用初态0..19、三个原seed，不开query/validation/H/heldout/qualification/reserved/final，不训练或改loss。第三步的结果在独立报告中记录；当前画面不能当作确认门已改善成功率。
