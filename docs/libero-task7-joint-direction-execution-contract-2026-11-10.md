# task7 同loss/同MSE上界方向对照：执行契约（2026-11-10）

执行[已冻结科学方案](libero-task7-joint-direction-plan-2026-11-10.md)，科学SHA `3a586c3270b9834889848e3dd175aa54ff8fd4d693f83ba7005fa55f7ec09251`保持。18个λ1辅助500头配对−G_T与等范数双目标角平分方向；两臂20级/最多50新增辅助更新，loss、表示、固定500误差上界不变。原Policy500、未来5步稳定化、旧cue/G停止/H未运行及正式封存保持，属于fit优化机制诊断。

## 执行与输入不可变性

`experiments/run_libero_task7_joint_direction_diagnostic.py` 提供preflight/evaluate/run，默认只校验绑定。run以CPU单线程/CUDA隐藏的新进程顺序启动主计算与独立审计，记录命令/PID/UTC/退出码/总耗时。主≤300秒/512MiB，审计≤600秒/768MiB，总≤960秒；已有结果、产物目录、launch/process或审计记录即拒绝。技术重试0、不覆盖，失败保留partial。主JSON保留pending，以audit/completion及两个exit0表示完成。

CLI和两个payload API都先校验新科学配置、七角色/本文及父774依赖闭包，再创建研究目录/读取研究归档。研究源头仅允许原λ1辅助500权重索引3，不能从旧fine停止/末点起步。源500参数、scaler、数据角色和注册fit只读对象保持；两臂只要求相同起点及fit，后续路径不作共同前缀或优劣排序约束。

归档guard沿用既有np.load/成员/ZIP/raw open/Path/io.FileIO/os.open及替代读取、Torch模型/优化器与环境导入限制。每阶段25原NPZ/161字段一次准入，主写18新NPZ/126成员，审计另读18/126；全程68读取/448解码/154归档SHA。原归档与新输出按冻结安排前后核SHA，不额外重读源CSV/优化NPZ、λ0、Adam状态、原.pt、HDF5/RGB或封存角色。源feature缓存物理解码全30个support示教，计算只能用本头20个fit示教的注册切片，encoder已见全部30示教的边界不变。

每头复用一次标准化表示，共18次重建/18次复用、50256标准化行；每个梯度状态仍两线性形式/三梯度，每个试步一次线性形式且不求梯度。原参数/上下文用内存摘要和只读标志核对；新NPZ只存[500、控制末点、候选末点]，保存前检查已完成两条路径、参数/scaler来源及七成员schema。

## 方向、精确导数和预算guard

`experiments/libero_task7_joint_direction_access.py` 的DirectionWork在注册状态的第一线性形式和三梯度之后准入，每状态恰好一次构造；norm/dot/vector操作按真实调用计数，分别最多4/2/4，向量坐标257。主的稳定范数/逐坐标构造和审计的Torch Float64范数/构造都经相同计账API，但不共享数值构造函数。额外方向归约不是新模型前向或梯度，不得漏记。

Jobs核对方向构造与state的G_T/Gb摘要、方向向量和status字段一致，控制逐坐标为−G_T，候选有效方向范数匹配绝对1e-10。精确sM/sP/sT共用第二次方向切线形式，核对sT≈sM/7+sP；候选在selected共同方向与精确sM≤0、sT<0时才允许试步。零/近反向/数值unresolved和夹点方向异常按冻结顺序正常停止，不改方向或阈值。

试步必须沿当前记录d、从j0顺序到j19、首接受即停，不跳点或加第21级。两臂共同接受严格T下降、精确sT的Armijo及Mtrial≤自身固定M500，无epsilon和逐步MSE门。上界内M上升合法并计入科学失败；所有试步保存当前M/ΔM、固定M500、M/P/T/F、两条件和拒绝原因；接受后核对新状态与ΔM账本，0更新保留，末点唯一。最坏每阶段1836状态/36000试步/1800更新及方向代数上限按science.budget核对。

旧fine20控制必须逐项复现416状态/3509试步/36端点/18头、398更新；历史只从旧JSON投影冻结标量字段，规范化控制标签，排除历史向量/参数参与计算。历史JSON完整解析的物理向量解码不伪称未发生。不存在换方向后要求7旧完整候选路径不变的检查。

七CSV最多51840行/64MiB，新18NPZ≤64MiB。主和审计分别记录归档/计算/标准化/方向计账、运行版本/环境/训练seed；环境seed null、初态为空，无环境交互。拒绝计数、科学线索失败和正常停止都不触发追加预算或选择checkpoint。

## 独立审计与严格判据

`experiments/audit_libero_task7_joint_direction_diagnostic.py` 从自身源500副本，用独立Torch autograd得到三梯度、独立构造方向/切线/条件/路径与逐头线索，不导入主数值实现、不调用science.direction参考函数。共享的配置和guard只核对角色、状态机与预算。新表、完整末点参数/梯度、逐状态标量/clip-hinge分支、全部试步和接受决定及摘要逐项核对；数值绝对1e-10、相对0，计数/分支/决定/科学cue精确一致。Float32锚点动作1e-7/目标1e-6仅作诊断，不放宽接受。

逐步MSE不增、与控制末点比较、与自身500比较分别独立重建；候选末点T更低/MSE不高且路径每次MSE不增才有主配对线索，所有18头及旧6/5/7分层、空组和三折原值保留。每来源9/9条件及旧T/G联合线索分别报告，不自动晋升原Policy/SPT/Knowledge/Module或开放query/正式评价。

特别保留严格排序的数值边界：若两臂几乎等价、末点T差仅为浮点舍入量，主/审计的严格cue可能不同。此时仍技术停止，不能用1e-10容差放宽Tcandidate<Tcontrol或MSE门，不能借用主结果的cue替代独立决定。绑定前首个零目标完整合成即触发该保护，失败日志保留；后续改为有非共线梯度取舍的合成夹具，生产算法和冻结判据未改。实际研究若出现同类分歧，同样不重试。

## 绑定前验证与启动

新增局部覆盖：方向范数匹配、相同/正交/反向/近反向/零梯度、clip/hinge精确导数、每项方向操作超额/时序拒绝、固定上界内局部M上升的科学失败、归档旁路和历史标量投影；完整两进程合成证明审计不调用direction参考；最大1836状态/36000试步+方向操作资源包络分别验证NumPy/Torch。随后CPU回归、环境、编译、git diff检查；具体XML/log/包络/选择清单与已知历史夹具排除随新binding.validation记录。

历史只读测试中“默认已完成输出必须不存在”的断言已在上一冻结记录，保留旧文件/哈希；本轮CPU回归显式排除该旧用例，由新协议的已有产物不变检查覆盖。不得删除已完成结果以使旧断言通过。

仅合成测试可在隔离子进程替换validate_call，并检查所有源归档和历史JSON均在当次pytest目录，生产无绕过开关。源码/方案/配置均不得在真实payload后修改；失败不就地改条件。

```bash
conda run --no-capture-output -n libero-kecrl python -m experiments.run_libero_task7_joint_direction_diagnostic --mode preflight
conda run --no-capture-output -n libero-kecrl python -m experiments.run_libero_task7_joint_direction_diagnostic --mode run
```

固定产物 `/tmp/libero-task7-joint-direction-20261110-v1.json` 与对应artifacts目录。独立审计完成后另记dated结果与provenance，再更新README/roadmap/500计划入口；不修改已冻结科学文件或历史结果。


完整合成验证记录：v1零目标夹具产生等价两臂的严格末点cue翻转，按冻结规则失败并保留；v2非共线夹具已通过全路径数值/判据，但最终IO汇总引用了不存在的metadata.NEW_HEAD_MEMBERS命名空间，已修正为冻结finer.NEW_HEAD_MEMBERS；该实现错误未改数学规则或科学配置。v3完整合成通过，1032401项独立核对。局部55项通过；最大包络主12.431秒/57.133MiB、审计10.402秒/472.555MiB，实际5508次autograd，方向1836构造/6426范数/3672点积/4590向量形式/1179630坐标，均在冻结上限内。研究payload当时尚未启动，合成重跑不计研究重试；原v1/v2日志与XML一并记录在binding.validation。
