# task7 有界约束可行性诊断：科学冻结记录（2026-11-12）

已完成[搜索域、求解器、预算与停止/接受规则](libero-task7-bounded-constrained-feasibility-plan-2026-11-12.md)的科学冻结，状态scientific_frozen_execution_not_bound。**本轮研究参数/归档/数值JSON读取和SHA、实际SLSQP/搜索/梯度/训练/环境调用均0，新可行头数未知。** 只读取已完成元数据、报告、源码与安装包代码，做小规模人造协议/梯度/预算算术验证，尚未运行真实solver全链路或完整18头最大调用包络。

## 做了什么、为什么

[上一轮有限网格](libero-task7-reference-joint-reachability-results-2026-11-11.md)P/U各0/18见证，不能证明连线之外或全空间不存在共同解。新诊断固定当前表示，以原四参照M/T条件为目标，在有界257坐标域内进行单起点局部约束搜索；不继续加密旧网格，不以失败直接调整原Policy训练或loss。

全部task7 18头、A_original/B_fomaml×20261106/07/08×fold0/1/2保留，P/U历史分开。原Policy500 updates、未来稳定化5步及原Policy训练loss保持，当前辅助λ1目标T=M/7+P也保持；新SLSQP主迭代/提议是另计辅助可行性搜索，不是增加原Policy训练steps。只用own20 source-support fit、t≥1含neutral；encoder已见全部30 support，不能称未见泛化。

## 冻结的范围与算法

每头、每历史固定以原bisector参数为center，逐坐标radius=max(2^-20, 四参照与center差的绝对值向外nextafter)，lower/upper=center±radius。以u∈[-1,1]^257搜索，theta=center+radius*u，保留Float64操作顺序。无随机起点、额外起点、半径/ftol扫描或失败后扩域；域不可表示则技术停止。实际radius数值尚未计算，未来由已冻结原输入和确定公式在求解前记录。

单次SciPy1.10.1 SLSQP，从u0=0开始；min原T，硬约束M≤min(原缓存四参照M_cap、该角色重演M_cap)。固定maxiter100、ftol1e−12，目标/约束均解析jac，禁有限差分、autograd和额外eps探测。clip边界inclusive1、hinge相等取0的既有解析约定保留。全部回调统一精确clip原生u到[-1,1]，记录投影，按规范u bytes精确缓存；只评价实际域内参数，不对M/T加epsilon。安装的SciPy包装器与原生扩展代码SHA已固定。

SLSQP ftol仅为算法停止设置，不是M/T接受容差。同参数两实现需分别满足原缓存和各自新重演四参照M不升/T严格更低；校准/数值核对绝对1e−10、相对0，门分歧保留，不弱化原caps、不新增P上界。找到候选不提前丢弃其余状态或头，不选checkpoint。

## 未来预算

每问题：≤100主迭代、256不同规范u求值、1024原生调用、4096总回调；单一起点、单一域。P/U各18个问题，共36，S只核对实际同参数，不运行solver。

四角色P_numpy_search、U_torch_search、S_P_torch、S_U_numpy合计：≤18720前向（含288参照校准）、52266240 fit帧、36864解析梯度向量、55584注册线性形式；≤3600主迭代、36864原生调用、147456回调。全量轨迹、缓存/投影/原生状态及实际消耗保留，未用预算不跨头/历史转移，不停止后追加uncached终点求值。

原源6JSON/12SHA、50NPZ/250成员解码/100SHA；旧源只投影4212标量和37008参照参数坐标。审计另读新主JSON一次/2SHA，向量投影≤4774032坐标，state/callback/native及头信息标量合≤1710630。物理解码/整文件SHA仍覆盖未投影字节。禁读原checkpoint/CSV/HDF5/RGB/初态/私有和封存数据，禁写新NPZ/checkpoint或可部署Policy。

CPU主≤900秒/1GiB，审计≤1800秒/2GiB，总≤3000秒；NumPy1.22.4/SciPy1.10.1/Torch2.4.1+cu124，CUDA隐藏、hash seed0、OMP/BLAS及Torch线程各1。详尽预算/字段/输出体积/实际计数规则在配置中；这里全是**未来上限，不是本轮实际执行量**。

## 结论规则与边界

阳性只记`verified_primal_witness_in_bounded_domain`。阴性只记`not_found_within_frozen_domain_and_budget`；**域内全局不可行及全空间不可行均未解决**。本版本没有负证书，status4“不可兼容”、status0“成功”、零梯度/KKT样残差或预算耗尽均不能证明不可达。clip目标不默认全局凸，P/U共用SLSQP包也不是两种独立全局算法。

见证不证明原500端点经旧50步方向/Armijo/M500/逐步M门可达，不证明闭环成功、跨任务迁移或资格通过。原主0/18、G停止/H未开展保持；不更新SPT/Knowledge、不注册Module，所有正式评价数据继续封存。

## 验证、证据与工作区

新增72项、父协议79项共151通过，0排除；初轮63项及扩展151项日志保留。覆盖四参照域包含性/向外舍入的精确有理核对、连线与凸包外参数、上下界和signed-zero、解析梯度与尺度链、人造clip/hinge边界、局部零梯度但域内另有可行点的反例、原始门/同参数审计、预算停止/缓存命中、源读取/SHA阻断、完整943父闭包与未来执行门。py_compile、环境检查、git diff --check通过，未跑全量pytest。纯协议测试不是实际求解/全预算执行，不用测试阳性代替研究结果。

- [协议/纯元数据门](../experiments/libero_task7_bounded_constrained_feasibility_protocol.py)、[人造协议测试](../tests/test_libero_task7_bounded_constrained_feasibility_protocol.py)。
- [科学配置](../configs/libero-task7-bounded-constrained-feasibility-diagnostic_v1.json)：SHA-256 `e345f813a492edf823e7a39cd518e511081d2541659d1d097f57c80fa41064c1`。
- [科学绑定](../configs/libero-task7-bounded-constrained-feasibility-freeze-binding_v1.json)：`26ea46b5324112c3583d13f3e923566f5039cc38d4a8c0dbfe83ac12581bdc1a`，953项依赖，含完整父943、完成元数据/报告、新协议/测试/方案/配置和SciPy包装器/原生代码。
- `/tmp/libero-task7-constrained-feasibility-tests-bound-20261112-v1.log`及XML：151通过；同前缀initial/final日志/XML保留。environment日志及1655文件初始快照均有哈希存入绑定。
- `/tmp/libero-task7-constrained-feasibility-freeze-initial-20261112-v1.json`保留初始git状态与文件哈希；绑定前1655旧文件未变。结束只给README/roadmap/500计划追加本状态，旧正文、AGENTS.md/.vscode/及无关phase-gripper修改保留，无删除/提交/推送或旧实验产物改写。

科学绑定在协议层明确拒绝研究访问：新执行绑定不存在，未来必须精确包含科学闭包及指定新角色/契约，不能把额外研究/封存文件作为待哈希依赖混入。

## 下一步

实现主入口、field/domain/cache/native/budget guard与独立U/S审计；用完整18头合成流程核对实际solver、缓存、投影、计数及最大预算包络，完成新的执行哈希绑定后再按预算运行。本科学冻结不授权数值访问。继续保持原Policy500 updates、未来稳定化5步、现有loss，query、validation、held-out、qualification、reserved、final及其他环境数据封存。
