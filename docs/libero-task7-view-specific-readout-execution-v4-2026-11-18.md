# task7视角专用读出v4：执行契约（2026-11-18）

落实[975项依赖的科学冻结](libero-task7-view-specific-readout-freeze-v4-2026-11-18.md)。当前用户明确授权新执行层、6模型×500更新及44条原交接状态闭环；本轮驱动止于交接审计和继续门，**即便门通过也不自动运行120条**。不修改冻结模块/科学配置/旧绑定，不沿用旧44条许可或历史控制成绩。

## 新入口与阶段准入

- `libero_view_specific_readout_execution_v4.py`：独立阶段准入与逐视角ComputeLedger。
- `run_libero_view_specific_readout_v4.py`：五字段source-support缓存、6模型配对训练/交接入口。
- `audit_libero_view_specific_readout_v4.py`：独立优化器/初态/分支/采样/熵/归档/原生端点与预算审计。
- `execute_libero_view_specific_readout_v4.py`：训练审计通过后新绑handoff，等GPU1，执行44条及独立审计；两种门结果均full0。

科学`require_execution_binding()`继续无条件拒绝。新执行preflight另核科学冻结、全部代码/验证闭包、当前用户授权、stage/caps/字段/来源/封存列表。train绑定在源HDF5/public weights的SHA与内容访问前建立；handoff绑定在初态库SHA/解码、checkpoint加载及物理步前建立，引用已完成的训练与独立审计证明。绑定拒绝科学绑定冒充执行绑定。来源读取契约不是操作系统安全沙箱。

## 训练与逐视角计算

完全沿用科学配置：task7 meta_support demos0..29、五字段、t≥1、432窗口/4188有效行；每臂三seed，各500 Adam3e−4更新、uniform masked七维MSE、batch8/seq10/stride10/clip1、CPU单线程/CUDA隐藏、last500。只加载科学配置指定的冻结ResNet18公开权重；支持缓存/冻结特征只构建一次，前后hash不变。HDF5禁止EEF、demo30+、属性、整体隐式数组读取与read_direct。

训练每个seed两臂共享非读出初态及逐批ID。候选记录每支initial state hash，由审计从固定offset独立重建；记录两支卷积梯度范数（有限非负，仅诊断，不按是否非零选点），两视角熵从既有缓存前向取出且先于parity/optimizer。XYZ梯度为0；冻结骨干、BN/RGB/温度/网格前后不变。无追加模型前向来记诊断。

每次候选前向严格依次`agentview`、`eye_in_hand`各一次，并核实际模块对象身份。缓存训练不得额外跑CNN；parity/闭环前向各两次冻结编码。每次调用先同时准入聚合及该视图预算，再调用分支。错误顺序、重复/缺失视图、错模块、额外CNN/读出或任一预算耗尽立即拒止。每视图/聚合的调用、图像行、热图、概率/坐标/熵值都独立记账且审计重算；双支与Policy熵快照上限81920。

检查optimizer全部参数实际step=500、Adam完整状态/设置、模型/分支形状与参数数目、权重/冻结buffer、零XYZ权重/一二阶矩、500条trace与sampler逐项一致。独立训练审计最多6初始模型+6checkpoint+1公开权重读取，0前向/训练/示教/初态/环境步；不把一致性审计称为独立重训或loss数值重演。

## 44条交接与停止

固定原22来源、seed覆盖12/8/2、原前缀/9次丢弃预热、两臂各一条；每条环境按0..目标reset，各5零动作，horizon300。初态库只选择性解码0..18共19行一次。最多15980物理步：2780稳定化+4700前缀+8500新策略动作；GPU1/EGL1、线程/版本/环境seed及所有计数按冻结配置。

严格旧公开状态/摘要及新成对前缀RGB/joint/gripper门保持；不得拿几何补录的±1像素门来放行。warmup、episode、partial、failure和完整账本写入新results，checkpoint在checkpoints，证据日志在logs；不覆盖旧产物，不自动重试或扩大预算。每次原生动作先检查命令、次数和预算；权重前后hash相同，私有EEF/物体状态查询、训练及额外render均0。

全部44条及审计通过后才报告原门：相对本轮新训练flat净增≥4/22且各seed不退化。独立标量公式重算门；12仅在控制8时成立，不借历史8或11作新门基准。新驱动记录门结果后结束，无full绑定、无120条。仅报告端点与已有熵，不宣称直接量出了新模型几何改善或完成正式迁移/Module资格。

## 验证与原始证据

定向人造执行测试覆盖6个小模型/3000实际Adam更新及checkpoint独立审计、全部44条最大15980步包络/独立归档审计；五字段封存、来源窗口元数据、熵篡改、科学绑定拒执行、逐视图错误及预算拒止，且继续门通过和失败均不启动full。小模型与模拟Adapter仅验证执行接口，不是LIBERO研究结果；冻结阶段的空白真实ResNet模块测试另保证生产结构/梯度。

运行前另做CPU环境与GPU1资源核对，写出train新授权/绑定，再训练与审计；完成后才生成handoff新授权/绑定并评估。完整测试、阶段启动/结束、实际资源和完成凭证保存在`logs/libero-task7-view-specific-readout-execution-20261118-v4/`及科学配置指定的新results。若发现代码/协议偏差，保持fail closed并保留失败现场；本契约不授予任何封存角色访问权。
