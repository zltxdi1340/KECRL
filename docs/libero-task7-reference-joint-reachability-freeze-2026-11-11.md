# task7 原四参照M/T共同可达性：有限见证科学冻结记录（2026-11-11）

已冻结[固定表示下的有限见证方案](libero-task7-reference-joint-reachability-plan-2026-11-11.md)，状态为scientific_frozen_execution_not_bound。**本阶段没有新实验结果**：6份源数值JSON、NPZ、checkpoint/CSV/HDF5及封存数据读取和SHA均0，实际研究前向、梯度/方向、试步、优化更新与环境交互均0。测试全部使用元数据与人造标量/向量；方案中的预算不是实际消耗。

## 冻结了什么、为什么

[上一轮只读结果](libero-task7-joint-t-tradeoff-gradient-results-2026-11-11.md)中，18/18联合T候选相对角平分都是M较高/P较低，14/18虽T降低仍不能满足原四参照条件；剩余梯度也不证明延长预算能解决。现在固定同一当前256维表示和λ1 loss，检查是否有**同一257维辅助头参数**同时满足：

```
M(theta) <= min(M_control, M_margin, M_zero, M_bisector)
T(theta) <  min(T_control, T_margin, T_zero, T_bisector)
T = M/7 + P
```

M为动作第3分量专家MSE，P为相对原Policy误差的正超额均值/7；其他6维保持缓存原Policy动作。保留四个原参照和全部并列活跃约束，不新增P上界，不混合不同参数的M/T或合并P/U不同历史。所有通过门按原值比较，M可等于、T严格更低；1e−10只用于数值重演，不放宽通过门。

## 未来怎么查、结论能说多远

task7全部18头（A_original/B_fomaml×20261106/07/08×fold0/1/2）均保留。每头固定四参照端点，以及6条两两连线各31个内部点α=1/32..31/32，合190配方；同一own20 fit标准化坐标、固定Float64操作次序，不优化、不去重、不提前命中停止、不细化网格。

四个计算角色P_numpy/P_torch/U_torch/U_numpy，各3420次辅助线性头前向，合13680次、38194560 fit帧；0训练/梯度/autograd/试步/原Policy或CNN/环境调用。原源JSON合6解析/12SHA，41220白名单投影值；审计另读新主JSON一次/2SHA。NPZ合50加载/250成员/100SHA，源head仅解码6个fit/归一化/预算成员，不解码其weights/bias，四参照参数来自历史端点JSON。整文件JSON解码/SHA仍覆盖未投影字节，归档整体SHA覆盖未解码字节，不称物理字段隔离。精确数据契约、预算及CPU时间/RSS见方案和配置。

同参数由两实现独立计算M/P/T，还必须同时优于原缓存四参照及同角色新重演的四参照，阻止自参照的舍入伪改善。原caps不换成重演值，同参数门分歧即使小于数值容差仍保留，不记经验证见证。参照校准、同参数数值审计或预算/角色等不符均技术停止，不自动重试。

找到只记verified_endpoint_witness；未找到只记not_found_on_frozen_grid。**有限边网格未命中不能证明整个表示不可达，正见证也不能证明原500起点经50步方向/Armijo/M500/逐步M门可达**。不以此替代原科学进展/路径判据，不采纳新Policy或选择checkpoint。主0/18、G停止/H未开展保持，原Policy500、未来稳定化5步及现loss不改。

## 验证及证据

新增79项与相邻只读分析69项，共148通过，0排除；首轮73项通过日志保留。覆盖全部四参照/caps等价、nextafter原始边界、不同参数不能拼接M/T、P不另加约束、参照重演与舍入伪见证、同参数坐标hex/门分歧、完整190配方/溢出/重复参数、真实存在窄可行区间而网格漏解的人造反例、预算守恒、数据封存、完整父闭包、科学冻结不能授权执行，以及缺入口/预算/包络/代码篡改拒绝。测试拦截Path读取和SHA的open入口，验证元数据工厂不碰研究payload。

py_compile、scripts/check_environment.py、git diff --check通过。运行环境Python3.10.22、PyTorch2.4.1+cu124；未启动训练。此次未改共享src/训练代码，未运行全量pytest，不能把定向验证说成全仓回归或真实数值独立审计。

- [协议及元数据门](../experiments/libero_task7_reference_joint_reachability_protocol.py)、[人造协议测试](../tests/test_libero_task7_reference_joint_reachability_protocol.py)。
- [科学配置](../configs/libero-task7-reference-joint-reachability-diagnostic_v1.json)，SHA-256 `f66a06736891529a45fa2cd5b791ca6841163ab99760e1e273c11f1f5880d145`。
- [科学绑定](../configs/libero-task7-reference-joint-reachability-freeze-binding_v1.json)，`613dbf8a471f600ce30433aa4598be0666fee9848b5e4957c3fac79fe994136d`；937项依赖，包括完整父929及完成元数据/报告、新协议/测试/方案/配置，不包含未来源数值payload的文件SHA操作。
- `/tmp/libero-task7-reference-joint-reachability-tests-final-20261111-v1.log`及同前缀XML：148通过；initial日志/XML：73通过。环境日志同前缀environment；绑定保存全部验证证据哈希。
- `/tmp/libero-task7-reference-joint-reachability-freeze-initial-20261111-v1.json`：1642文件初始快照，SHA `7619fdbfd48a6b756ce8147a0ef5e289ea0a0e65d1c6368e83acf706c67e8403`。937依赖绑定前，1642旧文件哈希全部未变；本阶段结束仅给README/roadmap/500计划追加本条状态，原正文及其他工作区修改保留。

旧研究配置、绑定、产物、AGENTS.md、.vscode/和无关phase-gripper报告不改。新验证文件留/tmp；不删除实验文件、不提交checkpoint、不进行git提交或推送。所有正式评价、query、validation、held-out、qualification、reserved、final和其他环境数据继续封存；不更新SPT/Knowledge、不注册Module，不把有限source-fit见证当作正式KECRL验证。

## 下一步

实现主入口、字段/归一化坐标/预算guard和独立审计，用完整18头合成流程及最大预算包络核对计数，完成**新的执行绑定**后再按冻结预算主/审计各执行一次，报告每头/每历史的见证、未命中及数值分歧。当前执行文件仅列为future路径，尚未实现或绑定；不得从本科学绑定直接开始数值读取。训练保持500 updates，不先改loss或追加预算。
