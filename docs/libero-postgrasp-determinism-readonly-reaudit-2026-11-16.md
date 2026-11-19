# 366步对照的只读审计v2：区分采样通道（2026-11-16）

两臂均完成153步/各183 native steps，公开旧摘要与新A/B物理字节门全部由主程序通过，但旧审计新增了未验证的额外断言：`raw qpos.astype(float32) == cached public joint/gripper`。它不是执行协议规定的“A/B原生状态逐位一致”或“旧/新公开观察一致”关系。真实A/B从第0帧起所有154帧均有相同的跨通道差异，审计据此错误拒绝。

原审计文件、测试、执行绑定、driver失败与全部原始产物保持原内容；不回写原audit/completion，不重跑物理。新增独立文件`audit_libero_postgrasp_determinism_v2.py`复制原审计，唯一准入修正是将此跨通道等价断言改为描述性差异统计；所有真正的零差门保留：原RGB/公开state/action/reward/native flags、新A/B float64 robot轨迹及目标端点、查询前后记录的物理状态。1e−12仍仅用于独立碰撞外包络公式。

源依据：robosuite `environments/base.py` 每控制步有多次内部sim.step与`_update_observables`，默认`_get_observations(force_update=False)`读observable缓存；`robots/robot.py`的joint sensor按control_freq采样；`utils/observables.py`按采样周期更新值。步后直接复制sim.data.qpos是另一采样通道，不能假定等于缓存。代码可解释为何两者不必相等；本轮没有记录完整传感器采样时间，不能进一步声称已测得固定几毫秒延迟。

本只读角色读取既有prepare、A/B result及保留观察/physics/endpoint/geometry，并选择性重读同一源数值前缀，0初态bank/环境/模型/HDF5/训练/额外render。新增独立只读配置/绑定在读取前冻结，新输出放单独目录。3项人造反例测试：缓存与raw不同但各自A/B相等应通过；新物理差1 ULP或RGB差1字节仍拒绝。保留旧测试未涵盖缓存时序的缺口，不把87项历史通过说成已覆盖这次错误。

若审计v2通过，只能准入本次两个新进程首环境短前缀，并完整披露旧审计实现错误和独立修正；不能将旧driver改写为从未失败、不能认定纯EGL或证明连续前6条上下文、不能自动放宽36条几何补录的门。
