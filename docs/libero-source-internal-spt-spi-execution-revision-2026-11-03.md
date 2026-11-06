# SPT/SPI source 内部诊断的确定性执行修订（2026-11-03）

科学设计沿用[原协议](libero-source-internal-spt-spi-protocol-2026-11-03.md)，500-update 适应、两 fold/三 seed/三臂及全部停止规则不变。v1 入口、配置、代码绑定、失败 event log 和 partial JSON 保留，不将其记为完成的研究实验。

v1 在第一批 support 的第一次 CUDA backward 因 `adaptive_avg_pool2d_backward_cuda` 无强制确定性实现而停止：一个 Policy forward 完成，一次 backward 尝试未完成，全部 optimizer 更新为0。`actual_counts` 是已完成整步的计数，额外的首次失败调用在 `/tmp/libero-source-internal-spt-spi-v1-failure-audit-20261103-v1.json` 补记；不得写成无模型调用。

v2 继续开启 `torch.use_deterministic_algorithms(True)`。固定128px输入经原CNN得到16×16 feature map，AdaptiveAvgPool到4×4的每个分区恰为互不重叠的4×4块，改用普通 AvgPool2d(kernel=4,stride=4)计算同一平均值。没有新增参数，不改变初始化、形状、loss、数据角色或预算，也没有关闭检查。仅支持整除网格，其他尺寸拒绝执行，不用近似代替。浮点求和实现可能改变末位，因此作为显式执行版本重新绑定，而非宣称位级等价于 v1。

CPU测试核对池化和整个Policy的输出/梯度及 state_dict 键值不变。每次 v2 启动前，另在合成 feature grid 上核对CPU原自适应池化与CUDA普通池化：输出、梯度各以abs1e-6/rel0比较，两次CUDA结果必须位级相同；2次CUDA池化前向/反向和1次CPU参照前向/反向作为执行验证单列，不计成Policy训练更新。没有新Policy前向、source数据或optimizer用于该合成门槛。

所有臂从原 seed 重新初始化，没有沿用失败梯度或状态。[v2配置](../configs/libero-source-internal-spt-spi-diagnostic_v2.json)、[v2执行入口](../experiments/run_libero_source_internal_spt_spi_diagnostic_v2.py)、[等价池化实现](../src/algorithms/libero_deterministic_pool.py)单独留存。仍只属于source内部离线可行性诊断，不开放validation、held-out、qualification或Module注册。
