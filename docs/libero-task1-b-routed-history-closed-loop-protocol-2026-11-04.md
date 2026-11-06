# task1 B固定路由读出120 episodes配对闭环协议（2026-11-04）

用户授权继续实验，且[runtime接口](libero-task1-b-routed-runtime-interface-protocol-2026-11-04.md)已实际通过并经独立审计，现于任何本轮初态数值读取/环境构建前冻结本协议。[此前训练](libero-task1-b-routed-history-readout-diagnostic-2026-11-04.md)只证明离线运动拟合改善、夹爪符号持平，本项检验是否对应真实控制收益；仍是重复source candidate诊断。

## 固定模型与启动

task1，训练seed20261022/23/24；current_all7 baseline和history_motion_current_gripper候选，共6个控制器，均固定旧B_fomaml编码器500端点及本次完整support读出500端点。不得用三折模型、中间端点或选有利seed。baseline和候选均全新执行，不复用旧成功率或把原B动作头作为本轮baseline。

两臂共享已审计的CPU单步编码器接口、各自support scaler及确定性动作。每episode reset清空历史，不把稳定化观察存入cache。前4个动作两臂使用同seed baseline读出及当前重复输入；候选第5步开始，运动6维读当前+过去4帧、夹爪1维读当前并使用候选自身固定权重。启动合同在结果之前冻结，不按成败改变。

## 角色、预算与设备

仅task1官方candidate_nonregression初态0..19，均已暴露；qualification20..39、reserved40..49、validation、held-out和final evaluation继续封存。复用既有选择性前缀读取器，完整容器checksum和193字节头核对后，仅数值解码前20×92 float64；不使用torch.load/pickle反序列化初态，不保存原始向量或live private state。

| 项目 | 冻结值 |
|---|---:|
| seed×arm控制器 / 每模型episodes | 6 / 20 |
| 新episodes / 旧成功率复用 | 120 / 0 |
| 每次稳定化 / 合计固定零动作 | 5 / 600 |
| 每episode Policy动作上限 | 300 |
| Policy动作/编码器/读出调用上限 | 各36000 |
| 受控native environment.step上限 | 36600 |
| native horizon | 305 |
| 环境seed | 20261022 |
| adapter数量 | 6，每模型复用20状态，0..19递增 |
| 官方初态数值解码 | 20行candidate；封存行0 |
| reset参照 / 精确比较 | 20 / 100 |
| 共享前四步动作比较上限 | 240 |
| 共享前五个pre-action观察比较上限 | 300 |
| 新训练/反向/HDF5读取 | 0 |

顺序固定seed22/23/24→baseline/candidate→初态0..19。每模型新建adapter，不共享原生环境生命周期；reset后固定5步七维零动作。CPU inference线程1，CUDA_VISIBLE_DEVICES为空、TF32关闭/确定性；保持已核对的单步CPU数值路径。EGL渲染固定物理GPU1，MUJOCO_GL=egl、MUJOCO_EGL_DEVICE_ID=1；执行前GPU1内存<=512MiB、利用率<=5%且无compute作业，不切换设备回退。

Python3.10.22、Torch2.4.1+cu124、NumPy1.22.4、robosuite1.4.0、MuJoCo3.2.3，LIBERO revision8f1084e3132a39270c3a13ebe37270a43ece2a01。native step是受控Python调用，constructor/reset内部及物理积分子步不伪称Policy动作。

## 配对门槛与记录

每state的首个baseline reset公开双RGB/joint/gripper摘要为参照，其余5个模型必须逐位相同；不一致在该episode首动作前停止。相同seed/state的候选前4个动作必须与fresh baseline逐位相同；前5个pre-action观察亦须逐位相同，因为第5个观察发生在4个共同动作后。合法提前成功按可见前缀比较，不填不存在的动作或硬凑上限。

每episode保存终止结果、全部7维动作、公开joint/gripper及相对post-stabilization reset偏移、逐步历史长度/索引/route、前缀观察摘要、配置/端点hash和环境seed。额外NPZ保存每步pre-action公开观察导出的256维表示与执行动作，供离线审计；它不是private simulator state，不保存RGB像素。包括attempted/completed计数、模型加载与时间事件。

只接受adapter公开成功或300步time_limit；异常、NaN、稳定化成功、边界/配对/hash/预算失配都属于technical failure，保存完成及部分episode并停止，主汇总无效，不算普通失败、不补样本、不放宽门槛。输入/权重/scaler必须保持不变。

主结果仅在120条全部完成后给出：各seed两臂成功数/率、candidate-only/baseline-only/both/neither四格以及三seed是否均严格多成功。合计分母60/arm是同20个状态跨训练seed重复，不做IID、显著性、未见泛化或qualification主张；不因中途结果选checkpoint、改启动/预算或追加训练。

[配置](../configs/libero-task1-b-routed-history-closed-loop-diagnostic_v1.json)、[执行入口](../experiments/run_libero_task1_b_routed_history_closed_loop.py)、[测试](../tests/test_libero_task1_b_routed_history_closed_loop.py)与本文绑定后才执行；runtime helper/adapter/reader等依赖也全部pin。默认只核对metadata，--execute执行。输出位于/tmp同名前缀20261104-v1，不覆盖历史。无论成败，不自动更新SPT/Knowledge、注册Module或开放封存角色。
