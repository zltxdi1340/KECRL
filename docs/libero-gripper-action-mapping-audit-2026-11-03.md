# LIBERO 夹爪动作映射只读核对（2026-11-03）

**已完成。** 当前 pinned Panda runtime 的夹爪增量使用 `sign(action[6])`，同一控制累积历史下，同号幅值差不会改变该夹爪控制增量。[上一轮分层](libero-task7-expert-camera-component-tradeoff-diagnostic-2026-11-03.md)的连续夹爪 MSE 增加，因此不能直接解释为闭合命令变弱。本项收束后转入 source 内部 SPT/SPI 可行性对照，不继续按该幅值热点搜索 loss。

## 核对范围与来源

只读源码和已有 task7 预测缓存，未加载 Policy、创建环境或读取 HDF5/初态/checkpoint。配置：[只读审计 v1](../configs/libero-gripper-action-mapping-readonly-audit_v1.json)，脚本：[audit entry](../experiments/audit_libero_gripper_action_mapping.py)。配置逐一 pin adapter、LIBERO wrapper、Panda robot、SingleArm、Manipulator、PandaGripper 及既有缓存的完整 SHA-256；任一变更拒收。

实际链为：adapter 原样检查并转 float32 → wrapper 传 action → SingleArm 切出 controller 后的夹爪分量 → `Manipulator.grip_action` → `PandaGripper.format_action` → actuator 范围的仿射映射。

当前 formatter：

```text
current_action = clip(current_action + [-1,+1] * 0.01 * sign(gripper_action), -1,+1)
```

`0.01` 是每次 formatter 调用的增量，不是每个 Policy step 的物理位移或力度；控制循环可能在同一 Policy action 下调用多次。零动作保持累计控制量。代码中的内部控制累计量在本项仅用虚拟数组检验，没有读取 simulator 私有状态。

## 结果与解释边界

从安装源码 AST 隔离 formatter、speed、dof 后，在零/饱和/近饱和五个合成累计量下测试正、负、零动作，共15组探针全部通过。同号值例如0.1/0.9具有相同结果；相同动作在不同累计历史下仍可不同。

task7三seed、四种弱相机干预共12组缓存也逐帧核验同号等价。seed20261024眼在手弱空间干预中：4188帧有4122帧符号相同，其中3934帧两者符号均正确；66帧符号不同。4122帧在五种相同历史条件下的20610次控制比较均一致。

这解释了为何“66.7%的正向连续 MSE 增量来自符号仍正确的帧”不能直接作为物理控制恶化证据。不同符号、先前累计历史、机械臂六维动作和接触状态仍可能产生不同物理轨迹；本项没有证明符号翻转是闭环失败根因，也没有否定视觉几何的重要性。

## provenance 与后续

结果：`/tmp/libero-gripper-action-mapping-audit-20261103-v1.json`；实际elapsed0.04784s，Python3.10.22、NumPy1.22.4，新增模型/训练/环境/HDF5/初态/checkpoint调用均0。结果本身记录配置和脚本 hash；5项新增审计测试、最终739项CPU回归通过（排除可选Crafter依赖组）。

下一步按[SPT/SPI source 内部协议](libero-source-internal-spt-spi-protocol-2026-11-03.md)执行 A/B/C，保持每个探针适应500 updates，现有闭环默认稳定化5步不变。source 内部离线结果与正式迁移、Knowledge Evolution和Module qualification分开。
