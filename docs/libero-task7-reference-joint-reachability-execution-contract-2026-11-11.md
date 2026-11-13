# task7 原四参照M/T有限见证：执行契约（2026-11-11）

实现[已冻结科学方案](libero-task7-reference-joint-reachability-plan-2026-11-11.md)，不改科学配置/旧绑定或数据预算。原Policy500、未来稳定化5步、λ1 loss、四参照和原主0/18保持，G停止/H未开展；query、validation、held-out、qualification、reserved、final、初态及其他环境继续封存。本轮是source-fit有限参数诊断，无优化器、梯度、trial、原Policy/CNN前向或环境调用，不注册Module或更新SPT/Knowledge。

## 执行文件与门

- 主/驱动器：`experiments/run_libero_task7_reference_joint_reachability_diagnostic.py`。
- 访问/字段/归一化/参数与前向预算guard：`experiments/libero_task7_reference_joint_reachability_access.py`。
- 独立审计：`experiments/audit_libero_task7_reference_joint_reachability_diagnostic.py`。
- 完整人造流程/预算包络测试：`tests/test_libero_task7_reference_joint_reachability_diagnostic.py`。

保留冻结协议及79项科学协议测试不改。新的execution-binding须包含完整937项科学闭包、科学绑定自身、新四文件及本契约（合943项）；检查所有哈希与完整18头合成/最大预算包络通过记录，才能接触原研究数值（包括SHA）。`--mode preflight`只验证元数据，不读payload。科学绑定不能替代执行绑定。

真实运行命令为`python -m experiments.run_libero_task7_reference_joint_reachability_diagnostic --mode run`，使用libero-kecrl的Python。驱动器不做研究前向，以固定环境启动两个新进程：主`--mode primary`，随后审计`--mode audit --primary-sha256 <驱动器取得的新主产物SHA>`。审计只对新主JSON做一次解析/前后2SHA；驱动器为传递新产物provenance另做输出文件字节SHA，属于新产物完整性元数据，不是额外源研究数据读取。记录命令、UTC、PID、git HEAD、配置/代码/数据哈希、wall/RSS与退出状态。已有launch/process/artifacts则拒绝；任一阶段失败即停止，不重试、不覆盖、不扩大预算。

CPU/Python3.10.22、NumPy1.22.4、Torch2.4.1+cu124。CUDA隐藏，hash seed0，OMP/OPENBLAS/MKL各1；审计Torch intra/inter-op各1、deterministic、inference_mode/no-grad。主≤300秒/512MiB，审计≤900秒/1GiB，驱动器总≤1500秒。VmHWM/VmRSS监测使用当前进程解析后的`/proc/self/status`，不以宿主pre-exec rusage峰值替代限制。

## 访问与字段

每阶段只读同一25个source归档各一次，前后SHA各一次；只解码125个冻结成员。head的8项必要核心schema是投影契约；**来源归档还含训练trace/预测等附加成员**，源码核对可见原HeadWriter按`libero_task7_residual_harm_protection_protocol.HEAD_MEMBERS`写入完整schema。执行按该既有生产者全schema核验，只解码冻结6项（fit_rows、fit_demo_indices、scaler_mean/scale、updates、optimizer_steps），weights/bias及其他成员均不可解码。shared/current只要求冻结成员存在，附加logits/proprio等不解码；源文件整体SHA仍覆盖其字节。这是实现纠正，不增加科学字段或读取预算。

每历史仅原latest/stepwise/zero_T三JSON，前后SHA。整文件物理解码后只投影54×6字段header/源和选定参照端点body/参数，latest另18×19 pairs；保留四参照身份、condition、n、更新数与原M/P/T缓存。原端点参数应匹配latest锚点，caps始终取原缓存，不取重演值或较弱参照。P/U历史单列。每阶段162 headers、72 bodies、18 pairs、2106标量与18504参数坐标；两阶段合41220源投影值。审计另投影新主JSON的18504参照参数坐标，总55512。

Guard封装Path/builtins/io/FileIO/os.open、NumPy load/NpzFile成员/ZipFile/底层数组读取及source SHA作用域，拒绝重复归档、重复/非白名单成员、raw open、mmap、替代CSV/HDF5/checkpoint读取和源文件写入。新CSV仅允许固定新路径排他创建。读出期间拒绝环境/模型模块导入、Torch checkpoint/Module/optimizer/autograd/backward和非注册矩阵前向。所有新数值写/tmp，不写checkpoint/NPZ，不读原CSV或新主CSV。独立审计以新主JSON为数值记录，重算其全部原始门和配方；CSV由同一行记录写出，人造完整流程验证格式和行数，不宣称真实审计再次解析了CSV。

Guard是源码和调用层研究契约，不声称操作系统安全隔离。物理解码含未投影字节的事实保留；SHA尝试与完成、JSON尝试与实际解析、已完成前向和部分投影分别记录，失败不冒充完整预算已执行。

## 读出、同参数与接受规则

每头own20 fit、t≥1含neutral，重算均值/总体std、scale≤1e−6置1，与保存scaler按原1e−10核对；normalize后Float32再Float64。所有四参照使用同一不可变fit上下文，guard核对shape、角色、内容摘要及头结束时未改变。encoder已见全部30 source support，不记未见泛化。

主按原四端点、6对×31内点固定顺序190槽计算P_numpy，NumPy einsum实际调用逐次入账。独立审计有自己的配方、插值、Torch/NumPy loss、caps及逐参照原始门代码，不导入主程序计算公式。P_torch从保存四端点独立重建，与主实际配方参数摘要逐一相同；U_torch生成参数后，U_numpy复用同一不可变参数对象。两者禁止用独立优化路径替代同参数核对。Torch linear实际调用也经guard入账，参数无梯度，原表示/参数不可变；每probe恰一次核心线性前向。上下文与参数格式转换不计新优化或新的逻辑参数配方。

每角色先做4参照校准（计入190），超原绝对1e−10则技术停止。全部190槽都保留，不因成功早停或按指标去重。见证须同一θ满足原四参照M≤、T<以及同角色新重演四参照M≤、T<；原caps不改。独立审计对P保存的门用其实际保存M/P/T及校准值逐项原样重算；再比较同参数跨实现M/P/T（绝对1e−10/相对0）。两边原值布尔门分歧即使在容差内也保留，不能用epsilon抹平或判见证。

## 精确总预算与输出

四角色各3420前向、9548640 fit帧、2444451840特征乘法；总13680、38194560帧、9777807360乘法。参数配方构造P_numpy/P_torch/U_torch各3420，U_numpy复用0；总10260向量/2636820逻辑坐标写入。物理类型转换、数组内部重复校验不伪装成新的源字段投影或训练。

四角色每probe原/重演共20次门比较，另重算P保存门3420组，合17100组/342000比较；每头每角色缓存/重演M/T各2最小值归约，加P保存核对，总360归约/1440值。实际计算预算由Jobs逐次前向、配方及门/归约作用域计数，不以声明代替执行。完整联合包络逐项核对原预算，包括6旧JSON/12SHA、50NPZ/250解码/100SHA、新主JSON1解析/2SHA和全部投影/参数量。

主`P-numpy-probes.csv`3420行；审计P-torch/U-torch/U-numpy各3420行，总13680，不能增加参数CSV。新JSON≤64MiB，每CSV≤64MiB。新主JSON包含18头四参照参数、校准、caps/原始门和3420probe；审计JSON包含10260probe、6840同参数比较、18头/来源/seed摘要、精确计数和最大误差。结果只给`verified_endpoint_witness`或`not_found_on_frozen_grid`，不选best recipe/checkpoint。

若P/U各自有经两实现确认的见证，同头可记双历史都有见证；不声称两历史见证θ相同。全空间不可达、全局最优和原50步路径可达均未由本扫描证明。原Policy500和原主0/18不因见证而变化。技术失败与网格未命中分开，完整保留反例/门分歧/已消耗预算，无训练修改或正式晋级。

## 绑定前验证记录

完整18头/4角色/190槽人造clip目标经真实NumPy/Torch计算、IO guard、主/审计新进程和生产驱动器日志/停止链；阳性和重复参数阴性头均保留，核对全部最大逻辑/IO预算及CSV格式。定向测试另覆盖源字段/角色/哈希/归一化、同参数对象/hex、原始nextafter边界、额外前向、资源门、真实入口缺绑定拒绝与子进程失败不重试。

首轮人造测试38通过、1失败/4 setup错误：telemetry解析路径白名单遗漏，另测试错锚点同时破坏守恒而未到预期错误分支；均在真实访问前修正，日志保留。第二轮43通过；扩展驱动器与门检查后204项相关测试通过（新增56+科学协议79+相邻只读69、0排除）。最终绑定保存最新测试/合成包络证据和环境检查哈希；上述测试数不能替代随后真实源的独立数值审计。
