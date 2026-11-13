# task7 负裕度有限步分解：科学冻结记录（2026-11-11）

**已完成科学冻结和843依赖绑定，尚未实现或绑定执行入口，没有本轮研究分析结果。** [完整方案](libero-task7-m-margin-finite-step-decomposition-plan-2026-11-11.md)、[科学配置](../configs/libero-task7-m-margin-finite-step-decomposition-readonly_v1.json)、[科学绑定](../configs/libero-task7-m-margin-finite-step-decomposition-freeze-binding_v1.json)、[元数据/人造标量参考](../experiments/libero_task7_m_margin_finite_step_decomposition_protocol.py)、[测试](../tests/test_libero_task7_m_margin_finite_step_decomposition_protocol.py)。

依据[上一轮已审计结果](libero-task7-m-descent-margin-direction-results-2026-11-11.md)，固定每实现全部18头、900接受步，407个M上升与493个未上升步共同分析。目的在于核对负一阶证书与有限步M变化、惩罚P补偿和原M500余量之间的关系；不据次数直接认定原因，也不预设应减小步长或改变loss。

未来从两个固定P/U JSON按白名单投影标量，主分析和独立审计各读每源一次，合计4次源JSON解析/8次源SHA；审计另读主输出一次/前后2次SHA。每阶段1836候选状态、14034已试trial、357972次标量字段投影，包含1800接受及12234拒绝；每阶段固定19846输出行。原NPZ/CSV/checkpoint/HDF5/图像/私有状态及其SHA均0，不再读角平分历史或补算未试α。JSON可物理解码完整文件，未投影字段不得作语义使用，不能把这等同没有物理解码其中数组。

主分类沿用原运行接受trial相对前态的原始ΔM，另存下一state差及原始符号分歧。固定分解ΔT=ΔM/7+ΔP、一阶预测αs与有限步余项、M500余量、P补偿比例；核对原始Armijo/M500决定。守恒绝对1e−10/相对0不用于放松符号或决定。零分母/溢出比例null并保留原因，不加epsilon。端点分支一致不能排除步内跨界，不把余项直接称曲率或根因。

保留每头、seed、来源和P/U，固定五个10更新窗及20级j，额外720个同实现/头/窗/粗j组分层；两结果类缺一则差值null、计数仍保留。步长由原line search和状态共同决定，407/493为事后分层，不作p值、独立样本或步长因果结论。仅标记已存试步是否满足M≤当前M，不重放新接受门、推演轨迹或选择checkpoint。

验证：50项新增通过（0.71秒）；新增与相邻检查117 passed、1 deselected，最后执行绑定输出位置条件补齐后同117项再通过（1.22秒）。排除的是已完成父协议中“研究产物不存在”的旧断言，未改旧测试；本轮缺少新执行绑定时的拒绝仍通过。人造算术覆盖负sM而有限步M升/P补偿、严格M500拒绝、trial/state微小符号差异留痕、全部拒绝reason、余项守恒和空/溢出比例。Python编译、CUDA隐藏环境检查及git diff --check通过。未改共享训练实现，未重复全量CPU/Crafter回归。

| 凭证 | SHA-256 |
| --- | --- |
| 科学配置 | dc244649ef278a8b4b667a06675e1382c330e5fd0aa893dcfffc7fb85c475bcb |
| 843依赖科学绑定 | 72d10a1b3d3724f324d6c6b17eb01285d3389facf230dba971a190beeddaf43c |

验证日志/XML、编译和环境凭证SHA列于科学绑定validation。初始工作区快照及最终元数据凭证保存在`/tmp/libero-task7-m-margin-finite-step-freeze-20261111-v1.json`和`/tmp/libero-task7-m-margin-finite-step-freeze-initial-20261111-v1.json`。本轮P/U数值JSON读取/SHA、真实标量分析、原归档访问、前向/梯度/优化/环境调用均0。原有文件除README/roadmap/500计划增加状态外保持；AGENTS/.vscode、phase文档、已有冻结文件和实验产物未动。

原Policy500 updates、未来稳定化5步、现有λ1 loss和M500门保持；source-meta-query及正式评价数据继续封存，父实验0/18、G/H及SPT/Knowledge/Module状态保持。CPU每阶段120秒/512MiB，总300秒、输出各≤32MiB、0技术重试/覆盖已冻结。

**下一步：实现主分析入口、字段/角色/读取预算guard、独立审计和运行测试，完成执行代码哈希绑定后，按冻结预算主/审计各一遍。** 科学绑定不替代执行绑定；完成审计并报告支持、反证和未解决问题后，才决定是否另冻一个训练或接受变量。
