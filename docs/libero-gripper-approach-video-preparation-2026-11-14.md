# LIBERO夹爪时机视频诊断：准备与科学冻结（2026-11-14）

**下一项已收敛为：看发出合爪命令前一帧，夹爪是否已在画面中对准目标黑碗。方案、采集/审计/浏览代码及人造验证已完成并科学冻结；本轮未实际重放，也没有新的时机结论。** 详细范围见[协议](libero-gripper-approach-video-protocol-2026-11-14.md)及[科学配置](../configs/libero-gripper-approach-video-diagnostic_v1.json)。

已有记录有完整动作，故先按原样重放动作、补录两路画面，不加载模型或重新推理。每一步必须和原公开状态逐位一致，reset还须匹配原RGB/状态摘要；不符即停止，不换例或放宽要求。新画面是当前环境重建，不能称原视频恢复，也不能证明原私有物体/接触状态逐位恢复。

## 已固定的可执行范围

- 15对、30段，覆盖task4/7、三个seed和每组非空的原四种结局；每组只取最小初态编号，不能看过画面后换例。样本属于结果已知的机制案例，不估计新的成功率。
- 每段最多100个保存动作，仍先做5个零动作稳定化；最多3000个保存动作、3150个受控native step。原Policy500更新、原adapter300动作horizon及现loss保持，100步主动停止不标成任务失败。
- 只需10个不同开发初态：task4 `[0,1,2,7,12]`，task7 `[0,3,5,11,14]`。逐个单行解码，显式global/local映射，不解码其余初态。完整容器SHA仍读取全文件字节，不能声称物理字节隔离。
- 两个原128×128 RGB视角，保存6060个camera帧及公开动作/joint/gripper数组，原RGB上限297,861,120字节。模型前向、训练更新、checkpoint/HDF5读取、额外render调用均为0。
- 合爪命令c用原阈值`action[6]>0.1`，重点看帧c−1，另存c−5/c/c+1/c+5的有效帧。窗口内无close、遮挡、深度不清或图像分辨率不足都保留无法判定，不自动填“过早”。判读前需按原任务指令确认目标，不能把任意可见碗当作目标。

## 实现与交付物

[协议/准入](../experiments/libero_gripper_approach_video_protocol.py)固定案例、预算、10行范围与判读要求；[采集/导出入口](../experiments/run_libero_gripper_approach_video_diagnostic.py)只使用保存动作，并在每一步比较原公开前缀；[独立审计](../experiments/audit_libero_gripper_approach_video_diagnostic.py)核对原动作/状态、无损数组、新观察摘要、事件锚点、紧凑初态映射与实际计数，不重放物理环境。

未来原始采集核对通过后，才导出A/B并排MP4、同步帧浏览页`review.html`、命令事件PNG拼图和30行待判读CSV。A/B隐藏模型与原结局，仍不能称前瞻盲测。无损NPZ为原证据；MP4有损且仅为3倍显示，不能创造原128像素之外的细节。判读必须给出observer、说明与证据帧；前两类可判定结果必须包含c−1。仅验证视频/记录完整性不会自动证明时机原因。

77项定向测试通过：新增25项，选择性初态reader及上一轮轨迹工具52项。完整30×100步流程使用假环境和人造RGB/JSON，独立核对通过；全套浏览/判读连接使用明确的媒体IO替身，另以人造RGB实际检验FFmpeg编码/解码、101帧和事件图。测试覆盖1 ULP公开状态不符、reset像素差、动作被改、重复native step、提前终止和超预算拒绝，以及遮挡/缺少合爪前证据帧时拒绝有效结论。它们都是软件验证，不是研究重放或真实画面结果。

编译、CPU环境与差异检查通过；未跑全量pytest、真实GPU/物理环境或研究数据加载。新版本实现未修改原adapter、reader、旧源码、配置、绑定或结果。

## 冻结、证据与执行状态

[科学冻结绑定](../configs/libero-gripper-approach-video-freeze-binding_v1.json)绑定599项依赖，含旧578项、当前源码、所需媒体代码和本地FFmpeg；77项测试证据一并锁定。状态`scientific_frozen_execution_unbound`，执行和初态授权均为false。

- 科学配置SHA：`57796ec806cfed7403eeaa21adf57277cc5d6403eefc71efa27f0d24e61e0f0a`。
- 科学绑定SHA：`1eb17c48be91a23804daa194ecb6749c96e38814610825563c71b4976a13289e`。
- 准备日志：`logs/libero-gripper-approach-video-preparation-20261114-v1/`，含准备前快照、阶段测试、最终77项XML/正文、环境及最终完整性收据。
- 计划的真实输出目录：`results/libero-gripper-approach-video-20261114-v1/`，目前不存在。

用户本轮赞同继续追查方向，本轮据此完成准备；没有将赞同自动记录为这30段新重放授权，也没有转用已消耗的旧240条授权。下一步建立本版本的明确执行范围与授权/执行绑定，再运行`python -m experiments.run_libero_gripper_approach_video_diagnostic --stage capture`及独立capture审计，随后导出media并判读。两阶段都必须传本版本`--config`与新的`--binding`。

本轮真实native step、初态数值读取、真实camera帧和模型读取均为0，其他source及正式数据保持封存；LIBERO继续作为主环境，不修改SPT/Knowledge或注册Module。
