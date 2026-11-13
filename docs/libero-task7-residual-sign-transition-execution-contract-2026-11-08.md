# task7残差错号转移与幅值分解执行契约（2026-11-08）

落实[科学冻结方案](libero-task7-residual-sign-transition-plan-2026-11-08.md)，科学配置SHA保持`7de0233952c2913c50b59071d7af79706dc26477f4db635d1bd6606a7aac0c0a`。原方案、配置、元数据门控及104项元数据测试保持原字节。七角色、传递源码依赖、本契约及预算先写入独立execution-binding，再允许真实CSV投影。不是新训练、正式评价、成功率或SPT/Knowledge/Module实验，不恢复已结束的辅助候选或G/H。

入口`python -m experiments.analyze_libero_task7_residual_sign_transition`默认只预检；`--mode run`依次启动全新CPU主分析和独立审计进程。每阶段真实活动≤180秒、外层等待≤195秒，总watchdog≤420秒，VmHWM/VmRSS≤512MiB、全部输出CSV≤256MiB。CUDA隐藏、单线程、固定Python3.10.22/NumPy1.22.4及科学配置环境变量。launch/pending/artifacts/process/partial任一存在即不重启；异常保留记录并停止，真实技术重试0、无覆盖。仅中断本次启动的进程组。

## 访问与数据角色

每阶段恰读两个已固定CSV，一表一遍：frames75384行各17列；curves4608行各9个ID，只有500/probe576行再投影14指标。0/100/250与fit的其它字段保留为未解释词法字符串，不转换为数字。源header/行宽与manifest完全一致；源写表代码确认target_direction为−1/0/1规范整数，tail20/clip为True/False，辅助mode、fold、role和phase严格枚举。其它六动作轴不解析；不是将action index2解释为世界z或close/open。

字段guard只交付已授权列；未列CSV、重复遍历、原始CSV读取、直接reader、NPZ/NPY/checkpoint及其字节SHA、numpy.load/fromfile/loadtxt/genfromtxt/read_array/memmap/Zip与低层文件入口、Torch/HDF5/环境/Policy/优化器调用均拒绝。只开放输入字节哈希的小作用域，以及本次输出和/proc内存遥测。它是研究访问契约，不是OS安全沙箱。源码及五份父JSON只在payload前按固定SHA核验，不递归调用旧数组访问入口。

从30条长度分别重建causal source_row/demo/t/probe_fold/tail20，要求三mode、六模型逐帧唯一完备，目标跨mode/seed/arm相同。原输出残差0；另独立重建保存float32(clip(p0+raw_residual))，clip与保存error/类别一致；bias_only每端点/fold为同一float32常数。仅共同t≥5、material目标3598帧进入三组四格，neutral与首4步仅用于ID和probe摘要复现。

## 数值和独立审计

主分析与审计分别从输入CSV重建，不共享新主分析的联接、分组、统计或数值函数。共享冻结口径、访问/词法转换和既有纯IO/资源限制。审计独立计算目标方向、六种类别、float32预测、bias常数、576个probe摘要、四格及计数/SSE守恒、九种符号组合、3×3幅值格、clip四格、示教构成、三seed票数、头间4×4交集和72份seed汇总。输出CSV逐行逐字段核对，保留空格及全部30个demo；ID/计数/类别/保存float32预测严格相等，派生浮点abs1e−10/rel0。审计进程拒绝导入新主分析数值模块。

gap=effective_shift−raw_shift明确同时包含clip与float32舍入；clip原标记另报。参考恰0的幅值比null、不加epsilon。组间corrected−stays_wrong和new_error−stays_correct固定双向，报告尺度和缺项，三seed均值只有三个值均可用才定义。不做显著性、因果归因或候选选择。两套独立学得头的差异不能拆成同一个仿射头的Wz与自身bias。

主结果pending-audit字节保持不变；audit/completion另存。源CSV SHA在阶段前后核对，字节哈希不混计数据遍数；审计读取新生成12表用于核对，另记12输出表遍历，不占来源CSV的4遍预算。主/审计各8064个probe标量核对；总4次来源CSV遍历、159984词法行、2662128投影列值，12输出表115728行及72份seed汇总。没有数组/头/权重读取或哈希，没有新前向、拟合、环境交互、source-query或封存评价payload。

## 验证和结果记录

绑定前先执行访问/标量合成测试，再用与真实同规模、其它轴及未授权曲线指标含不可数值转换哨兵的合成CSV，在两个全新进程中检验完整主分析与独立审计。实际数据只在绑定和合成流程通过后执行一次；绑定前发现的实现错误及修正保留在测试日志和结果报告，不能抹去失败或借机更改冻结科学口径。

执行绑定记录测试、软件/环境、依赖SHA、运行命令及git状态；实际结果/审计记录各阶段PID、时间、RSS、attempted/completed遍数/行/投影列值、probe与全表核对数及失败/重试。dated results报告与README/roadmap/500计划同步。所有父joint cue仍false、原Policy500/未来稳定化5步/现有loss、G停止与source-query及正式评价封存保持；结果只用于支持/反证/未解决问题汇总，不自动进入训练。

绑定前实际验证：72项新增运行测试、349项相关协议/运行测试、2388项CPU回归通过；编译和环境检查通过。第一次扩大回归仅忽略test_crafter*，遗漏test_spatial_gather_spt.py的可选依赖，记录2399通过、18项缺crafter失败和4跳过；补全排除后2388全部通过，原失败日志保留。没有修改Crafter测试、安装依赖或将环境问题归为LIBERO失败。两个完整合成流程均通过，无合成数值/访问失败、无真实投影或真实重试。源写表代码的整数方向标签及/proc/self符号链接路径在绑定前已核对；无科学口径变动。
