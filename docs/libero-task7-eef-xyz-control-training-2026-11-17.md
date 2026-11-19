# task7缓存EEF XYZ模型v2：配对训练记录（2026-11-17）

6个模型、3000次实际Adam更新已完成，独立训练审计3972项核对通过，另通过30份schema/432窗口/XYZ缓存审计。训练进程130.206秒、审计4.192秒；本记录不推断闭环收益。配置与结论边界见[冻结协议](libero-task7-eef-xyz-control-protocol-2026-11-17.md)。

两臂总参数均12690632、可训练1514120；每seed完整初始state_dict及500步minibatch相同。ResNet18/BN及不可训练状态不变，Adam所有可训练参数step=500；控制臂XYZ三列权重和一/二阶矩保持不变。原值世界米XYZ输入的候选臂每seed500步均有非零XYZ梯度，证明通道参与了本次优化，不代表尺度最优或已改善控制。

| seed | 臂 | 前50步均loss | 后50步均loss | 非零XYZ梯度步数 | 平均XYZ梯度范数（裁剪后） |
|---|---|---:|---:|---:|---:|
| 20261106 | zero_xyz_mlp | 0.079578 | 0.024767 | 0 | 0.000000 |
| 20261106 | cached_xyz_mlp | 0.078446 | 0.026783 | 500 | 0.005120 |
| 20261107 | zero_xyz_mlp | 0.076694 | 0.026117 | 0 | 0.000000 |
| 20261107 | cached_xyz_mlp | 0.077067 | 0.022640 | 500 | 0.005311 |
| 20261108 | zero_xyz_mlp | 0.078549 | 0.026925 | 0 | 0.000000 |
| 20261108 | cached_xyz_mlp | 0.077620 | 0.030991 | 500 | 0.004925 |

XYZ各seed末段loss并非一致优于控制，因此不据loss选seed/模型/端点。唯一checkpoint是last500，后续仍须全部22个状态两臂配对评估并独立审计。

实际仅task7 meta_support demos0..29、432因果t≥1窗口、4188有效行（12564个XYZ坐标）；所有30份官方obs/ee_pos字段shape/dtype通过检查，demo0为[155,3]/float64，输入统一float32。2次HDF5打开、2592总切片（含432 EEF），120冻结编码器调用、6像素/缓存前向核对；训练阶段初态/环境/物理步为0。独立审计不再读HDF5，不增加训练或物理步。

证据：[结构化汇总](evidence/libero-task7-eef-xyz-training-2026-11-17.json)，原始`results/libero-task7-eef-xyz-control-20261117-v2/train/`，权重`checkpoints/libero-task7-eef-xyz-control-20261117-v2/`，日志`logs/libero-task7-eef-xyz-model-v2-20261117-v1/`。928项训练依赖锁定、163项定向回归通过；未跑全量pytest。Policy500/稳定化5/现MSE及全部正式封存保持。
