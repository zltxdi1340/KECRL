# task7 负裕度有限步只读分解：执行与审计契约（2026-11-11）

执行[科学方案](libero-task7-m-margin-finite-step-decomposition-plan-2026-11-11.md)及[843依赖科学冻结](libero-task7-m-margin-finite-step-decomposition-freeze-2026-11-11.md)，不修改原配置、分类、标量公式、固定分层、预算或解释边界。用途为已保存source-fit辅助优化的diagnostic，不进行原Policy或辅助头训练，不作正式评价。

入口`experiments/analyze_libero_task7_m_margin_finite_step_decomposition.py`默认只预检；`--mode run`核对执行绑定，依次启动主分析和审计两个独立进程。API也须先核对科学配置、科学绑定、全部执行角色和依赖闭包，之后才允许访问源JSON或创建实验目录。没有新执行绑定，旧训练绑定或科学绑定均不能授权本次运行。

`experiments/libero_task7_m_margin_finite_step_decomposition_access.py`仅放行本轮指定P/U JSON的单次解析、各前后一次SHA和当前进程内存监测。路径统一resolve，`/proc/self/status`仅对应本进程实际PID；不放宽至其他进程或私有状态。标准库CPU运行，禁止未列文件、原NPZ/CSV/checkpoint/HDF5及替代IO、数值/模型/模拟器导入；源JSON完整物理解码，随后只投影白名单标量，原始对象丢弃。只读映射拒绝未投影字段与修改，不将物理解码声称为未解码数组。

每源核对status、formal=false、science/binding及U.primary，再对states/trials逐行取arm标签；控制行其余字段不投影。候选只投影科学配置26个state列和22个trial列，身份、分母、状态k0..50、50轮、连续j、首个接受恰为最后一条、全部18头和精确行数均检查。每阶段17358个标签、1836候选状态、14034候选trial、357972次字段投影；两阶段合计4源JSON解析/8SHA、715944字段投影。审计另外读取新主输出一次及前后2次SHA，不能重复读取源或读旧角平分历史。

主分析使用已冻标量参考算术；`experiments/audit_libero_task7_m_margin_finite_step_decomposition.py`自行实现M/P/T、严格Armijo/M500决定、有限步余项、比值、trial/state对齐、逐头累计、均值/中位数/范围、等折后等seed以及全部分层。审计禁止导入主数值或汇总模块，不调用主实现或冻结参考中的数值函数；只共享字段、身份、IO与预算契约。独立重读同两源并重建全部19846行，再逐项检查主结果，浮点绝对1e−10/相对0、类型/布尔/字符串/null/结构精确；不同参数路径P/U仅描述差异，不套同参数门。

每实现900接受步原始trial差分必须仍407 up/493 nonincrease，保留down/zero及next-state符号差异；余项和补偿用保存的精确右导数与实际α，不计算新α、梯度、Hessian或Policy动作。比值无正分母或溢出时null及原因，无epsilon。所有接受及此前拒绝trial均保存，端点分支计数不能证明步内未跨界。已存试步的M≤当前M只作标记，不推演替代轨迹、预算或成功率。

固定输出steps1800、saved_trials14034、rounds1800、heads36及科学方案规定的分层表，共19846行/阶段，主/审计39692行。空组null和计数保留，不据新结果挑选头、窗口或步长。先逐头/折、seed、来源汇总，等权结果与按步数加权结果分别有明确分母；720个同头/窗/粗j组分层缺一类则比较null。步长与结果事后条件化，不作因果、显著性或新训练接受判断。

P/C记录配置/绑定、输入pin、真实命令/PID、UTC、CPU/环境、读取和字段计数、各头覆盖及实际时间/RSS。原Policy500、未来稳定化5步、现有λ1 loss/ρ/κ/M500门、原0/18、G/H及正式角色封存保持。每阶段120秒/512MiB，总300秒、源文件各≤96MiB、输出各≤32MiB；有界流式写入、0技术重试/覆盖，产物全部留/tmp。异常fail closed，仅保留已有计数和已写partial，不新增排错源读出、不改绑定自动重跑。

绑定前以人造数据验证完整18头、源状态/试步/字段精确预算、全部固定输出及空分层，并在独立进程中禁用主/参考数值helper验证审计。还验证字段/IO/预算绕过、错身份/分支/证书、严格M500边界、三类拒绝reason、符号留痕、类型/浮点篡改及时间/内存/输出上限。最初发现的监测路径resolve错误与测试检查数预期错误在真实源读取前修正，失败日志/XML保留；不计研究运行重试。最终测试/环境凭证与入口、guard、审计、运行测试、本契约、科学绑定及完整科学闭包哈希固定后，主分析和审计各运行一次，完成审计后再报告支持、反证、未解决问题和下一步。
