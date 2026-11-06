# task1 B同一候选权重双输入配对闭环协议（2026-11-04）

用户要求另冻同一候选权重、真实历史／重复当前的配对闭环。本协议在本轮环境构建、初态数值解码及研究前向之前冻结；当前仅冻结和检查入口，未执行闭环。[共同第5步交叉读出](libero-task1-b-step5-crossed-readout-diagnostic-2026-11-04.md)发现固定候选头的内容项常较大，但尚不能判断这种输入差异是否改变控制结果。本项独立检验此问题，仍是已暴露source candidate上的diagnostic。

## 两臂与唯一改变项

task1 B_fomaml，训练seed20261022/23/24，固定原编码器500端点、已完成完整support的候选读出500端点及support scaler。每seed两臂使用完全相同的编码器、候选头、启动头和scaler；不重新训练，不选择checkpoint，不使用三折读出或中间端点。六个运行控制器对应三套固定权重和两种输入路由。

两臂每次reset清空历史；稳定化观察不入历史。前4个动作共同使用同seed current_all7启动头及重复当前输入，保持父闭环启动合同。第5步起：

| 输入臂 | 运动6维输入 | 夹爪1维输入 | 权重 |
|---|---|---|---|
| candidate_real_history | 当前及过去4帧，顺序[t,t-1,t-2,t-3,t-4] | 当前重复5次 | 同一候选500头 |
| candidate_repeat_current | [t,t,t,t,t] | 当前重复5次 | 同一候选500头 |

各256维表示先按同一support scaler标准化，五块串接并除sqrt(5)，float32 CPU Linear+tanh；不使用未来帧、其他episode或递归hidden state。真实历史臂每成熟动作两次Linear（运动历史、夹爪当前），重复当前臂复用一次当前Linear输出。运行计数分开记录控制读出与内部Linear，不将重复算术当额外环境动作。

重复当前可能偏离候选训练时的自然历史分布。该对照只识别指定路由在当前执行合同下的控制差异，不能独立证明必要记忆、物理环境机制或失败唯一根因。两臂分叉后观察会随控制改变，后续夹爪差不表示夹爪历史被直接干预；首个分叉前的同观察夹爪必须完全相同。

## 角色、预算与设备

仅task1官方candidate_nonregression初态0..19，已被此前多次使用。qualification20..39、reserved40..49、validation、held-out、final evaluation封存。不读取HDF5、专家目标或新query。未来执行沿用选择性前缀读取器：校验容器字节及193字节pickle头，只数值解码candidate前20×92 float64；不用torch.load/pickle解码初态，不保存原向量或live private state。冻结检查不打开初态容器。

| 项目 | 冻结值 |
|---|---:|
| 训练seed / 输入臂 / 初态 | 3 / 2 / 20 |
| fresh episodes / 复用旧结果 | 120 / 0 |
| 每episode稳定化 / 固定零动作合计 | 5 / 600 |
| 每episode Policy动作上限 / native horizon | 300 / 305 |
| Policy动作、编码器、控制读出上限 | 各36000 |
| 内部Linear调用上限 | 53760 |
| 受控native step上限 | 36600 |
| adapter / 编码器加载 / 读出bundle加载 | 6 / 6 / 6 |
| 同seed权重身份参照 / 比较 | 3 / 3 |
| candidate初态数值解码 / 封存行解码 | 20 / 0 |
| reset公开观察参照 / 精确比较 | 20 / 100 |
| 共享前4步动作比较上限 | 240 |
| 共享前5个pre-action观察及表示比较上限 | 各300 |
| 同观察第5步夹爪比较上限 | 60 |
| 新optimizer / backward / HDF5 | 0 / 0 / 0 |

固定顺序seed22/23/24→真实历史/重复当前→初态0..19；每seed×输入臂新建adapter，复用20次且递增reset。两臂全部fresh，不能用旧真实历史0/60替代本轮任何轨迹。环境seed20261022、图像128×128、稳定化5步七维零动作；native step计受控Python调用，不计为物理积分子步或constructor/reset内部调用。

CPU inference单线程、CUDA_VISIBLE_DEVICES空、确定性/TF32关闭；EGL固定物理GPU1（MUJOCO_GL=egl，MUJOCO_EGL_DEVICE_ID=1）。执行前检查GPU1内存<=512MiB、利用率<=5%且无compute作业，忙则保留技术拒绝，不自动换GPU。Python3.10.22、Torch2.4.1+cu124、NumPy1.22.4、robosuite1.4.0、MuJoCo3.2.3，LIBERO revision8f1084e3132a39270c3a13ebe37270a43ece2a01。

## 门槛、记录、停止与解释

各seed两臂的固定模型/scaler身份hash必须一致；每state的首个reset公开双RGB/joint/gripper摘要作参照，其余5模型必须精确一致。相同seed/state的前4动作、前5个pre-action观察及256维表示必须逐位相同，第5步夹爪输出逐位相同；检查在该动作执行前完成。合法提前成功只比较实际可见前缀，不补不存在的帧。

逐episode保存公开success/time_limit、全部动作、公开joint/gripper及相对post-stabilization reset偏移、逐步真实历史索引和实际输入索引/route、前缀digest、端点/scaler身份hash。NPZ保存所有pre-action公开观察导出的256维表示、执行动作与步号；不保存RGB像素或private state。记录配置/代码/输入hash、环境seed、设备版本、实际attempted/completed计数、UTC事件、加载和时间成本。

完整120条才允许主汇总：每seed及合计两臂成功数/率，both/real-history-only/repeat-current-only/neither四格，repeat-current减real-history成功数，以及三seed是否均严格增加。另逐对描述动作/终止长度是否相同，防止同为失败就声称控制过程没变；无需新增前向。合计每臂60是20状态跨训练seed重复，禁止IID、显著性、unseen或qualification主张。

不因中途成功率提前停止或追加预算。异常、NaN、初态/动作/历史/路由/权重/hash/配对/预算不符、稳定化提前成功均为technical failure，保留已完成及部分轨迹、失败原因和attempted/completed，不混为普通time_limit，不替换样本、不放宽阈值。任一臂技术失败则本轮主结果无效。成功/失败都不自动恢复旧路由候选接受，不训练或更新SPT/Knowledge，不注册Module。

本[配置](../configs/libero-task1-b-fixed-head-input-closed-loop-diagnostic_v1.json)、[入口](../experiments/run_libero_task1_b_fixed_head_input_closed_loop.py)、[新运行helper](../experiments/libero_fixed_head_input_runtime.py)、[测试](../tests/test_libero_task1_b_fixed_head_input_closed_loop.py)及本文独立hash绑定；旧已执行的代码/配置保持不变。默认命令只作metadata preflight；未来执行另加--execute，输出位于/tmp独立同名前缀20261104-v1，排他创建、不覆盖历史。当前状态与验证见[冻结报告](libero-task1-b-fixed-head-input-closed-loop-diagnostic-2026-11-04.md)。
