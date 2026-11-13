# task7 终点梯度/分支只读几何：冻结完成记录（2026-11-11）

已冻结[分析对象、解析判据、字段与预算](libero-task7-terminal-gradient-geometry-plan-2026-11-11.md)。当前只完成元数据和人造验证，没有读取研究数值JSON、计算真实终点几何或运行新优化；实际线性可行头数仍未知。科学绑定状态`scientific_frozen_execution_not_bound`，没有执行绑定。

## 做了什么、为什么做

基于[上一轮M优先方向结果](libero-task7-m-priority-direction-results-2026-11-11.md)，保留全部18头，区分“缓存负M方向与T冲突”和“缓存梯度/半径/M界内是否存在联合线性方向”。另将已知6个零更新终点与缓存saved500控制方向交叉核对；其余12个更新后终点单列，不能借用控制在不同参数点的结果。

未来仅投影上一轮P/U两JSON的head/endpoint路由与头元数据，18个候选终点和zero6的6个saved500点体；每点38个标量及G/Gb/Gp/已存方向四个257维向量。不读取参数、原始输入/特征/target、states/trials或其他端点body，不生成替代方向、试步或更新。

线性模型固定为||d||≤缓存r、7Gb·d≤缓存B<0、G·d<0。以缓存binary64坐标为精确实数，Fraction主参考及独立2幂分母整数证明核对容量C和可行差D的精确符号，不使用epsilon、cosine夹紧或近零投影门，不把审计容差加到半径。原程序浮点门与该线性模型的边界差异另报，不回写旧结果。

clip/hinge计数用于区分“无tie的充分光滑线索”和“替代方向exact未知”。即使线性可行，也始终保留未验证替代方向exact门、未验证固定网格有限步的标记；不据此认定当前loss、表示或任务全局不可行。

## 得到什么：未来冻结预算

主/审计各读P/U一次，合计4次源JSON解析/8次SHA。每源24点（18终点+6起点），每阶段48点，总96点；105936个投影字段/坐标值。两阶段共≤384 norm、576 float dot、288精确内积归约及74016精确坐标乘积，不做任何模型前向、梯度或新方向向量。

每阶段82输出行（36终点、12zero对照、12来源×seed、4来源、18双实现对比）；审计另读新主结果一次/2SHA。CPU、Python3.10.22，无NumPy/Torch计算进程，CUDA隐藏、单线程；每阶段120秒/512MiB，总300秒，0技术重试/覆盖。字段/身份/provenance、非有限、分支计数/守恒或独立核对失败均fail closed保留partial，全部新产物留/tmp。

原Policy500 updates、未来稳定化5步、现有λ1 loss/表示/停止与四参照规则保持；当前主0/18、G停止/H未开展不变。source为own20 support fit、encoder见过全部30示教，不称未见泛化。query、validation、held-out、qualification、reserved、final和其他环境数据封存；不更新SPT/Knowledge或注册Module。

## 证据在哪里

- [科学配置](../configs/libero-task7-terminal-gradient-geometry-readonly_v1.json)，SHA-256 `a4759c468a1bf10cac7941e83ddd01c138c6a9e85a083f57407fe95ac138eadc`。
- [科学冻结绑定](../configs/libero-task7-terminal-gradient-geometry-freeze-binding_v1.json)，SHA-256 `2991673d9774f90300cca1c053b06b2f7b2b05fee39fcc542cfa2141948b97fd`，896项依赖，含父887闭包、完成元数据及新协议/测试/计划/配置。
- [人造数学与元数据参考](../experiments/libero_task7_terminal_gradient_geometry_protocol.py)、[测试](../tests/test_libero_task7_terminal_gradient_geometry_protocol.py)：新增51项、相关116项，共167通过；排除1项父实验的历史“产物不存在”断言，旧代码/测试未改。
- 最终日志 `/tmp/libero-task7-terminal-gradient-geometry-freeze-tests-20261111-v2.log`及同前缀XML、环境日志均在科学绑定中保存哈希。py_compile、环境检查、git diff --check通过；未运行全量pytest，本轮未改共享src。

验证覆盖正交/同向/反向、空M半空间、零梯度、严格零边界及nextafter两侧、极端正尺度不变性、独立整数证明、576组人造二维约束最小值、分支缺项/计数/未知界、预算守恒、数据角色与配置篡改、数值payload读取/SHA禁止以及缺执行绑定拒绝。初次50通过，补充二维交叉验证后167项复核通过，无研究运行重试。

冻结前1607个既有工作区文件哈希保持；AGENTS.md、.vscode/及原有未提交修改保留。此次新增协议、测试、计划、科学配置、科学绑定及本记录；README/roadmap/500计划仅追加状态，历史科学文件不修改。本次真实源JSON/数组读取及SHA、研究几何计算、前向/梯度/试步/优化/环境调用均0。

## 下一步

实现主入口、字段/坐标/角色/预算guard和独立整数审计，验证完整人造投影流程与字段预算，完成实际执行哈希绑定后主/审计各运行一遍，再依据全部18头的线性证书、分支及双实现差异决定后续。当前没有创建执行角色、执行绑定、产物目录或新的训练配置。
