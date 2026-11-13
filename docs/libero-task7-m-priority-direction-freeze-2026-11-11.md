# task7 M优先方向：冻结完成记录（2026-11-11）

已冻结[公式、两臂、预算及停止规则](libero-task7-m-priority-direction-plan-2026-11-11.md)，完成元数据/人造向量验证；本轮没有真实前向、梯度、试步、优化或环境调用。科学绑定状态为`scientific_frozen_execution_not_bound`，没有执行绑定或新实验结果。

## 做了什么、为什么做

基于[上一轮只读端点分解](libero-task7-stepwise-m-reference-tradeoff-results-2026-11-11.md)，冻结仅改变方向构造的候选：控制为已含逐步M门的T优先负裕度方向，候选为本臂当前`−||G_T||×G_(M/7)/||G_(M/7)||`。原λ1目标T=M/7+惩罚P、M证书ρ=.01、T/Armijo、M500及逐步M门、表示和数据角色保持。候选没有κ投影参数，construction_margin_fraction=null；κ=.02仍只属于控制构造。

该方向在同范数球内给出最大的M线性下降，但T未必下降。人造向量验证了负M方向与T冲突、而控制仍存在联合下降方向的情形；候选应科学停止，不能推论全局无可行解或另换投影方向。没有新增T裕度、λ扫描、网格扩展或预算回补。

覆盖task7的全部18个原辅助500头（A_original/B_fomaml、seed20261106/07/08、三折），两臂都从原500副本开始，各≤50辅助接受更新，20级α=2^-j、j0..19。原Policy仍500 updates，未来稳定化5步。本轮冻结不是原Policy训练、正式SPI/SPT或Knowledge验证。

## 得到什么：冻结内容

控制须在P/U各自历史上精确复现900更新、918状态、7691试步、36个500/末点、18头及0个M上升接受步。7691来自旧guarded候选；旧7017/407属于未加逐步门的另一臂，不能误作本次控制。

跨臂只要求相同原500参数、三梯度、分支及表示；方向可从k0不同，原“逐步门否决前共享路径”约束不适用。控制自身的历史全路径仍须严格核对。

保留四参照：本次guarded T控制，以及原负裕度、旧零裕度、旧角平分三个固定端点。主线索要求更新、自身T进展、M执行不变量，并相对四参照均T严格更低且M不高；每来源各9/9、两来源、P/U及局部稳定规则保持。没有因换控制降低旧科学标准，也不把受强制单调单独作为改善证据。

未来主/独立U总计≤3600辅助更新，S0更新；P/U/S合计≤5508梯度状态、11016 autograd、77073 trial、89925线性形式、251647425线性行。读取/写入仍为68次NPZ/574成员/172归档SHA、36归档/630成员写，CSV≤69058行。P300秒/512MiB、C900秒/1GiB、总1500秒、0技术重试或覆盖；详见配置的逐角色账。

## 证据在哪里

- [科学配置](../configs/libero-task7-m-priority-direction-diagnostic_v1.json)，SHA-256 `6f13044f54f06e19fcd745625f368daf6104dd1d2a88d6bab62a82e4b35b402f`。
- [科学冻结绑定](../configs/libero-task7-m-priority-direction-freeze-binding_v1.json)，SHA-256 `3b7bd5e303022135073c8693019607371d9fec971eb6bf1445f2ec90bfa2d9b7`，881项依赖，含父873闭包、新协议/测试/计划/配置及完成元数据。
- [协议参考实现](../experiments/libero_task7_m_priority_direction_protocol.py)与[测试](../tests/test_libero_task7_m_priority_direction_protocol.py)。新增69项、相关133项，共202通过；排除1项旧“产物不存在”断言，未修改旧测试。
- 最终日志 `/tmp/libero-task7-m-priority-freeze-tests-20261111-v3.log`及同前缀XML，哈希记录在冻结绑定；py_compile、环境检查和git diff --check通过。未运行全量pytest，本轮未改共享src。

验证覆盖控制算术不变、负M同范数/线性最优性、人造方向冲突、selected/exact原始符号、零梯度/零Gb/下溢/溢出、证书伪造、双臂原接受门、完整20拒绝、实测M不升、四参照、全18/来源覆盖以及缺执行绑定拒绝。工厂测试禁止源NPZ/CSV/HDF5/checkpoint和当前历史P/U、前次只读analysis/audit数值JSON的读取或SHA，创建配置仅使用仓库及完成元数据。

最终绑定前复核将父优化引用由旧负裕度配置对齐到实际控制的逐步M门配置，补断言后202项复核通过；公式、数据、接受/停止和预算未变。未绑定草案保存在`/tmp/libero-task7-m-priority-unbound-config-20261111-v1.json`，SHA-256 `7e2b4bb45ddbb43620e30abb663d2d1f640398244c504f8131da054e14f2fba8`；未改历史协议或重跑研究结果。

绑定前1594个既有文件哈希保持，AGENTS.md、.vscode/和原有未提交工作保留。此次新增协议、测试、计划、科学配置、科学绑定及本记录；README/roadmap/500计划仅追加当前状态。

## 下一步

实现主入口、方向/字段/角色/预算/历史控制guard和独立U/S审计；先通过完整18头人造流程和新最大预算包络，再绑定实际执行角色哈希，按冻结预算各运行一次。本次没有创建这些执行角色、执行绑定或产物目录。

source query、validation、held-out、qualification、reserved、final及其他环境数据继续封存，原G停止与联合0/18不变。来源仅own20 support fit、encoder见过全部30示教，不是新泛化结果；不得在实现/审计或诊断改善后自动修改原Policy、更新SPT/Knowledge或注册Module。
