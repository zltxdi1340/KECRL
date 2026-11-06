# task1 B路由读出runtime接口冻结协议（2026-11-04）

用户要求继续下一实验；承接[配对训练结果](libero-task1-b-routed-history-readout-diagnostic-2026-11-04.md)，先检查固定500端点的真实单步输入和历史缓存，再另冻candidate闭环。当前不训练、不访问官方初态或执行环境。

固定task1 B三seed20261022/23/24的编码器500端点及六个全support读出500端点，不能使用OOF模型或换checkpoint。CPU单线程、原均匀七维MSE和闭环默认稳定化5步保持。两臂current_all7/history_motion_current_gripper；每次demo/episode reset显式清空feature历史，单步参数必须从1顺序递增。

启动合同：前4个Policy动作，两臂都使用**同seed baseline头和当前重复输入**。第五步起，baseline仍全部读当前；候选运动6维读当前及过去4帧，夹爪1维读当前，使用自身固定500权重。不是在头部补造零帧、把未来帧或稳定化观察当历史，也不是选不同候选权重。startup选定于任何本轮环境结果之前；无专家action0或physical状态标签。

先在全部既有source-support缓存上逐demo流式读出：4059帧/seed×三seed×两臂=24354次模型读出，含720次共享启动；共同t>=5共23634行须与旧固定500预测绝对差<=1e-5且夹爪符号相同。权重/scaler不变，启动两臂逐位相同，跨demo不能带入历史。

再加载同一合法source-support的前8个sequence10窗口，d0 causal动作t1..80对应stored obs0..79，读取80行公开观察/动作；通过原batch digest和完整HDF5 checksum后才前向。三个固定编码器各做1个batch前向、80次单步前向、首步另重复1次；总246次、483有效行。batch/step/旧cache的256维表示和原动作差<=1e-5，首步重复逐位相同，权重/输入不变。各seed/arm使用单步特征流式读出80次，与旧cache流式动作差<=1e-5，共480次新读出。

总读出24834次，编码器246次，checkpoint加载3次；新增optimizer/backward/环境/query payload/初态/private访问全部0。只使用source demos0..29的元数据和既有缓存，新增RGB/action payload严格限80行；不读取qualification20..39、reserved40..49或validation/held-out/final evaluation。源示教编号与官方初态编号分别管理。

[配置](../configs/libero-task1-b-routed-runtime-interface-diagnostic_v1.json)、[runner](../experiments/run_libero_task1_b_routed_runtime_interface.py)、[runtime helper](../experiments/libero_routed_history_runtime.py)、[测试](../tests/test_libero_task1_b_routed_runtime_interface.py)及本文由绑定hash锁定后执行。metadata/hash/预算/数值不一致立即停止，独占创建结果与失败记录，不覆盖旧产物或放宽阈值。

本项仅检验软件接口及合法输入的数值一致性，不检验物理任务成功、必要记忆或SPT更新。通过后按用户授权继续另冻task1三seed两臂120 episodes配对闭环；失败则保留全部证据、先修复接口，不把未运行闭环写成结果。
