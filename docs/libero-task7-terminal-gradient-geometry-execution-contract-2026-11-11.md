# task7 终点梯度/分支只读几何：执行契约（2026-11-11）

执行[冻结方案](libero-task7-terminal-gradient-geometry-plan-2026-11-11.md)及896项[科学冻结依赖](../configs/libero-task7-terminal-gradient-geometry-freeze-binding_v1.json)，不修改旧协议、缓存或接受/停止规则。原Policy500 updates、未来稳定化5步、表示与λ1 loss保持，主科学0/18、G停止/H未开展不由线性几何晋级。source为own20 support fit、encoder已见全部30示教；query/validation/held-out/qualification/reserved/final及其他环境继续封存。

主入口`experiments/analyze_libero_task7_terminal_gradient_geometry.py`默认仅预检，`--mode run`顺序启动主分析与独立整数审计各一次。每个子进程在读取研究数值JSON或创建产物目录前核对科学配置、科学绑定、完整父闭包、本轮入口/guard/审计/运行测试及本契约；缺执行绑定即拒绝。执行绑定在合成验证通过后创建，包含902项依赖及验证日志、环境、初始工作区快照的哈希。本轮源码与本契约绑定后不改。

`experiments/libero_task7_terminal_gradient_geometry_access.py`只允许冻结P/U的JSON解析各一遍、前后各一次SHA。路由36头/54端点身份必须完整、不重不漏；元数据核对控制900更新/7691试步、候选33/266、全部18头停止及固定zero6。只投影18候选终点和zero6的6个saved500点体，每点38标量和G/Gb/Gp/缓存方向四个257维有限binary64向量。映射与坐标以只读缓存交付，拒绝未列字段/向量、其他body及重复源读取；JSON物理解码包含未投影内容，在访问记录中明确。禁止额外NPZ/CSV/HDF5/checkpoint/输入/特征/target/参数/私有/封存读取（含SHA），禁止NumPy/Torch/模型/仿真导入、未列IO与写入。

每阶段精确计账：1080路由值、2544标量body、49344坐标，合计52968投影值；48点、192 norm、288 float dot/74016坐标乘积、12336梯度和坐标核对、9252个zero梯度逐位比较、144精确内积/37008坐标乘积。操作guard逐点登记且每操作限固定顺序，拒绝外来点、重复或漏算，不挪用预算。两阶段合计105936投影值、384 norm/576 dot/288精确归约。输出固定82行/阶段：36终点、12zero控制、12来源×seed、4来源、18历史双实现对比；不生成研究CSV。

检查缓存T=M/7+P、G≈Gb+Gp、范数及selected斜率（绝对1e−10、相对0）；保持原始G、半径r和负M界B，不重算方向/半径或以容差改变符号。ρ=.01、B的原表达式、cached selected/exact原始布尔和分支计数精确核对。所选M方向终点的selected/exact T正、M证书通过，与其他方向是否存在分开。描述性范数和点积各由stdlib重新计算，余弦原值不夹紧，未定义/越界单列。

主分析以Fraction.from_float对缓存坐标精确求A、B₂、z、Gram、C和D符号；独立审计不导入主入口、不调用其数学实现或冻结Fraction证书，而分别将向量转换为共同2幂分母整数，独立归约三个内积并交叉相乘证明C/D符号。两者均只判断||d||≤缓存r、7Gb·d≤缓存B、G·d<0的存在性，不生成任何新方向。空半空间、零G、严格零边界独立分类；不加epsilon、不用余弦近零门。审计自行重建分支、终点及来源×seed/来源汇总；分类、符号、身份和布尔零容差，描述性浮点按原1e−10核对。P/U历史路径分类或分支差异只报告，不要求不同参数点相同。

zero6逐后端比较候选终点与saved500三个梯度float.hex、n/M/P/T/分支及零更新，继承父实验已审计的同起点参数/零更新不变性，不读取参数。缓存控制selected/exact证书另列，不声称本轮试了有限步或其方向一定满足精确半径模型。分支无clip tie/hinge zero仅作充分光滑线索，出现tie时替代方向exact未知。无论几何结果如何，alternative_direction_exact_gate_checked=false、finite_trial_checked=false始终保留，不能据此认定优化/闭环或qualification通过。

独立审计另读新主结果一次、前后2SHA，核对来源/配置/绑定、全部82行及访问/运算实账，写独立结果与completion。CPU/Python3.10.22新进程、CUDA隐藏、hash seed0、OMP/BLAS各1，0新前向/梯度/autograd/优化/trial/环境及新方向。每阶段≤120秒/512MiB、总≤300秒，源JSON≤192MiB、输出各≤4MiB。所有新数值产物、日志、launch/process/completion/partial留在/tmp；x创建，0覆盖/技术重试。字段、坐标、预算、数值守恒、资源或独立审计失败保留partial与日志并停止，不改绑定自动重跑。

绑定前完成全18头人造P/U投影、主分析、独立新进程审计及驱动器日志/完成链，验证精确边界/nextafter/极端尺度、未列IO/字段/坐标/身份/分支、损坏来源和新结果、逐位zero、更改输入SHA、计账和资源上限及无覆盖/无重试。初轮完整计算通过，发现测试错误地要求描述性均值逐位相同，改为冻结浮点核对规则后复核；原精确分类与符号门未改，初始日志留证。实际研究数值在合成通过并绑定后才读，各阶段仅运行一次，再按18头真实分类和分支边界决定下一步，不自动改训练。
