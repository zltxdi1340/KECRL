# task7原Policy误差/焦点对齐执行契约（2026-11-08）

本契约落实[已冻结科学方案](libero-task7-policy-error-focus-alignment-plan-2026-11-08.md)，不改变其任务、六个端点、数据角色、阈值、参照或预算。原科学配置SHA保持`39fd9bbe7c578ddd80191602762a68ceeebc15e1731f6117ce531c5453690f30`，原冻结方案、元数据门控及测试均保留。新增七角色执行绑定另存于`configs/libero-task7-policy-error-focus-alignment-execution-binding_v1.json`；绑定前不得对真实数组或标签行作数值解码。

执行入口为`python -m experiments.analyze_libero_task7_policy_error_focus_alignment`，默认仅预检；`--mode run`在两个新CPU进程中依次执行主分析和独立审计。Python3.10.22/NumPy1.22.4、CUDA隐藏、单线程、PYTHONHASHSEED0；每阶段120秒/当前地址空间512MiB，总watchdog300秒，技术重试0，已存在launch、结果或partial即拒绝。子进程中断时只终止本次启动的进程组。配置、七角色、依赖和本契约SHA在每阶段预检，运行命令、PID、UTC、退出码和成本另存provenance。

每阶段固定7次NPZ加载、11字段解码：共享5字段和六份features.npz内的predictions单字段。仅允许path形式的`np.load(..., allow_pickle=False)`，重复加载/成员、features/logits、mmap/pickle/fromfile、直接Zip成员、其它文件读取及模型/HDF5/环境入口均拒绝；数组返回只读。该guard是研究访问契约，不称操作系统安全沙箱。

每阶段两张固定CSV各1次词法遍历、各21588行，只向数值实现暴露冻结的12列；原header须与父manifest完全一致且无重复，未授权列保持词法字符串且不暴露。CSV文件只在投影函数打开句柄时授权，yield后不得留下原文件读取权限。投影字典的列集合、整数/布尔编码、角色/符号均核对；主分析和审计分别重建因果source_row及一对一ID、端点/转移、旧Policy_sign。字节SHA在阶段前后及加载时核对并单列记账，不以哈希读取替代数值额度。

九张CSV共38990行；列schema由绑定的实现固定。空数值保存为空字段，对应JSON null；计数/SSE为0、布尔为True/False。原float32先转float64，逐轴平方误差与共享目标能量先计算一次再复用。每轴用固定float64列求和及分母，动作组、等权demo与seed汇总用`math.fsum`，避免零附近求和顺序改变精确higher/lower/tie标签；不增加epsilon或统计阈值。六种类别、相对目标能量、目标幅值和同帧其它轴保留；NN二值标签只分组，不当作连续Policy动作。

独立审计不导入主数值实现，分别实现ID/标签重建、逐帧六类与误差、分组、参照/示教构成和跨seed票数，并逐字段流式核对九表及三seed JSON汇总。整数/ID/布尔/字符严格一致，浮点abs1e-10/rel0；全源legacy共216标量，夹爪sign准确率严格相等。主pending-audit JSON保持原字节，audit/completion单独写入。任一校验失败即停止并保留partial，无覆盖、自动修缓存或实际重试。

绑定前测试发现并修复两项工程问题：合成数据上的不同求和顺序导致零附近方向标签不一致；访问单测共享完整pytest进程时继承Torch驻留内存，触发资源上限。前者统一算术；后者只在访问规则单测注入独立资源桩，完整合成主/审计仍在新进程中实测512MiB/120秒。原失败日志/记录保留于`/tmp/libero-task7-policy-error-focus-alignment-prebinding-*-failure-20261108-v1.json`及相应测试日志。此时真实研究数组/标签行解码和重试均0；最终验证、执行及结论另见结果报告。

保持现有500 updates、未来稳定化5步、原loss和G失败/H未运行；不新训、不前向、不运行环境或近邻搜索，不读取source meta_query或正式封存payload，不更新SPT/Knowledge或注册Module。结果只描述原Policy对专家动作的离线误差关联。
