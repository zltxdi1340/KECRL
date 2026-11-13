# task7 固定表示下有界约束可行性诊断：科学方案（2026-11-12）

**本阶段只冻结，不执行研究搜索或读取研究数值输入（含SHA）。** 后续先实现新主入口、访问/字段/域/回调/原生调用/预算guard、独立U/S审计，经完整18头合成及最大预算包络、执行代码绑定后再运行。此方案是source-fit局部可行性diagnostic，不是正式SPT/SPI、Knowledge Evolution或Module qualification。

## 问题与不变项

[上一轮](libero-task7-reference-joint-reachability-results-2026-11-11.md)四参照190槽有限网格P/U均0/18见证，所有内部点未过原M门；这不能证明连线之间、连线之外或整个表示不可达。本轮从固定连线扩展到**固定257维参数域中的受约束局部搜索**，不继续加密连线，不变更原T定义或放宽M上界。

原Policy保持500 updates，未来闭环稳定化5步、原λ1 loss/表示不变。全部task7的18头：A_original/B_fomaml×20261106/07/08×fold0/1/2，harm_protected_affine；P/U旧历史分别保留。source own20 fit、t≥1含neutral，每折2782/2817/2777帧；encoder已见全部30 source support，不能称未见泛化。query、validation、held-out、qualification、reserved、final、初态和其他环境数据继续封存。原主0/18、G停止/H未开展保持。

原Policy训练500、原辅助头500及其历史方向端点是不同计数。本轮未来辅助SLSQP参数提议不计为原Policy继续训练，不把“500+SLSQP迭代数”当作新Policy预算或正式适应结果。不会采纳/部署任何新头，不更新SPT/Knowledge、不注册Module。

## 固定目标、约束与可行见证

M为动作第3分量（索引2）专家MSE，P为相对原Policy逐帧平方误差的正超额均值/7，T=M/7+P。其余6维仍用缓存原Policy动作。给定固定标准化当前表示z：

```
raw3 = original3 + (sum_j z_j * theta_j + theta_bias)
prediction3 = clip(raw3,-1,1)
M = mean((prediction3-expert3)^2)
P = mean(max((prediction3-expert3)^2-(original3-expert3)^2,0))/7
T = M/7 + P
```

同原fit20 mean/std64、scale≤1e−6置1、normalize后Float32再Float64；不训练encoder、替换观测或读HDF5。四参照control/margin/zero/bisector的**原缓存**M/P/T不替换，分别取原M/T最小值并保留全部活跃参照。每角色重演四参照，原/重演M/P/T超原绝对1e−10即技术停止。

对本角色定义M_search_bound=min(原M_cap,本角色重演M_cap)。求解`min T(theta(u))`，仅一个显式不等式`c(u)=M_search_bound-M(theta(u))>=0`，加固定u边界；不另调λ、加惩罚loss或给P设上界。重演M约束仅增加数值确认，不能放松或替代原四参照。T不引入人为改善裕度，见证仍须**同一参数**严格小于原及重演的全部四参照T，并不高于它们的M。

原始门零容差；SLSQP的ftol不进入M/T通过门。候选必须在域内，由另一实现同参数复核M/P/T及两解析梯度（绝对1e−10/相对0），两边分别通过原缓存与各自重演的原始门才有数值primal见证。即使差在容差内，门分歧仍记录，不强制同判。上轮1 ULP自参照伪改善的防护继续保留。

## 明确搜索范围

每头、每历史独立构造一个域，center=c=该历史原bisector端点257参数。每坐标：

```
gap_rj = nextafter(abs(reference_rj-center_j), +inf)    # 四原参照
radius_j = max(2^-20, gap_control_j, gap_margin_j, gap_zero_j, gap_bisector_j)
lower_j = center_j - radius_j
upper_j = center_j + radius_j
u_j in [-1,1]
theta_j = center_j + (radius_j*u_j)
```

Float64、固定操作次序、不FMA/重排。差值向外nextafter是预定的边界舍入规则，非结果驱动扩大；半径下限2^-20是单个预先固定尺度，避免四参照相同的坐标被永久锁住，不是扫描参数。要求半径/边界有限、lower<center<upper、四参照每坐标被包含，失败则域不可表示并技术停止，不增大下限或换域。

实际范围的数值须在未来绑定后、任何求解前由冻结引用计算并完整记录。本阶段不读参数来预先测量半径。范围由同一固定公式和旧输入SHA唯一确定，不能看搜索结果后调整。相同公式分别作用P/U历史，不强迫两历史数值域相同。

这比六条边提供更多参数组合，**不保证覆盖整个盒、更不覆盖整个257维空间**。权重与偏置全部参与，表示和训练scaler保持不变；u只是求解坐标变换。

每问题仅从`u0=0`启动一次（即bisector）。不另启其他参照、不随机重启、不缩放/扩大域或重用成功头的预算。起点可能因原缓存舍入而不满足M_search_bound，必须记录实际余量，不将其强制改为可行。

## 固定求解器与解析梯度

采用已安装SciPy1.10.1 `optimize.minimize(method='SLSQP')`；NumPy1.22.4，包装源码和原生扩展SHA写入配置/绑定。固定maxiter=100、ftol=1e−12、disp=False、iprint=0，目标jac和约束jac均显式提供，无有限差分、额外eps探测或超参数候选。只有本一个算法/起点/尺度，不按结果选solver。

解析约定：`qM=2*(prediction-y)*1[-1<=raw<=1]`，`qT=(qM/7)*(1+1[current_error-original_error>0])`；分别计算mean(q·[z,1])得到theta梯度，再逐坐标乘radius得到u梯度。约束jac为`-gradient_M_u`。clip边界采用inclusive1，hinge相等采用0，沿用既有约定。**clip/hinge并非处处光滑，不能把所选解析约定、SLSQP success或残差当作凸性/KKT/全局最优证书。** 原生BFGS/QP工作单独计原生调用，不伪装成新的fit梯度或Policy训练。

安装源码表明SciPy目标包装会clip极少越界的原生u，但不等式回调直接接收原生u。因此本协议预先规定：**全部回调**（T、gradient_T、c、gradient_c、major_monitor）先以同一精确component clip规范化到[-1,1]，记录原/规范u摘要、改变坐标数和最大改变量；不使用epsilon或扩大域。该投影只定义搜索坐标，不修正M/T结果；只验证实际被求值的规范u和重建theta。

所有数值回调通过同一个按规范u的binary64 bytes键控缓存。每个新键一次性取得M/P/T和gradient_M_u/gradient_T_u；返回已存标量/梯度不新增前向，缓存键必须精确，不能近邻合并。major_monitor只记录事件和现有状态映射，不产生额外fit求值；终点若未缓存，不在停止后补读，不以未经打分的终点作见证。找到见证不提前结束头或丢弃其他头；保留所有访问状态，不选checkpoint。

缓存必须保存规范u、theta及梯度的不可变副本，不能持有Fortran原生u的可变视图；交付给solver的jac数组不得反向改写缓存。此类交付复制计入回调交付坐标上限，不新增一次梯度求值。

## 预算与停止

每个问题硬上限：100主迭代、1024原生reverse-communication调用、4096总回调、256个不同规范u求值（含u0）。先检查预算，再进入计算；不分配给其他头/历史。标准SLSQP结束状态保留；状态4“inequality incompatible”、0“success”、9“iteration limit”或预算耗尽均不自动证明可行/不可行。非法字段/身份/域/非有限值、绑定/数值审计不符、时间/RSS超限属于技术失败，无重试，保留实际部分消耗；不能称科学阴性。

SciPy包装器首次传入majiter的是maxiter−1=99上限槽位，不是已完成99次；必须区分该初值与原生返回的实际主迭代数及major callback事件。不能据初始99错误计账或跳过原生调用监测。纯协议的can_admit仅检查已完成工作的计数，缓存命中可按increment=0，不能把待执行工作先当作完成。

| 项目 | 冻结上限 |
| --- | ---: |
| P/U问题 | 各18，共36；S不启动solver |
| 主迭代 / 原生调用 / 总回调 | ≤3600 / 36864 / 147456 |
| P/U新状态 / 同参数S状态 | 各合≤9216，四角色合≤18432状态读出 |
| 四参照校准 | 四角色合288前向，另计入下行总量 |
| 前向 / fit帧 | 每角色≤4680 / 13066560；合≤18720 / 52266240 |
| 两解析梯度向量 / 转置矩阵形式 | 合≤36864 / 36864，梯度fit行≤102924288 |
| 全部注册线性形式 / 特征乘法 | ≤55584 / 39728775168 |
| 参数构造（含校准复制）/逻辑坐标写入 | ≤18720 / 4811040 |
| u梯度缩放坐标写入 | ≤9474048 |
| 回调规范化/梯度交付坐标访问 | 各≤37896192 |
| 域构造 / center-radius-bounds向量坐标 | 72 / 74016，P/U/S各自独立核对 |
| 原源JSON / SHA | 6解析 / 12；只投影4212标量+37008参照参数坐标 |
| 原NPZ / 成员解码 / SHA | 50 / 250 / 100，沿用每阶段25归档/125成员 |
| 新主JSON审计 / SHA | 1解析 / 2 |
| 新主参照+域 / 状态向量坐标投影 | 37008 / ≤4737024，合≤4774032 |
| 新主state/callback/native标量投影 | ≤1709568，字段逐项见配置 |
| 新主头/参照/校准/caps标量投影 | 1062；与上行合≤1710630 |
| 原Policy/CNN/环境、autograd/backward/有限差分 | 全部0 |

每角色的fit行数上限按各头真实n之和50256乘相应最大状态数，不能用均值n偷换实际消耗。所有回调和原生调用尝试/完成、投影/缓存命中、参数和M/T/梯度读出分别入账。主要预算为上限，提前正常停止时记录真实值，不能报告成固定全额。

CPU/Python3.10.22、SciPy1.10.1、NumPy1.22.4、Torch2.4.1+cu124；CUDA隐藏/hash seed0、OMP与BLAS各1、审计Torch intra/inter-op各1，禁用autograd。主≤900秒/1GiB，审计≤1800秒/2GiB，总≤3000秒；原JSON≤192MiB，新主输入/新主和审计JSON各≤256MiB，每CSV≤64MiB。达到输出/内存门时技术停止，不能截断轨迹冒充完整审计。新NPZ/checkpoint/可部署Policy写入0，所有产物独占新/tmp目录。

## 输入、轨迹与独立审计

冻结原3JSON/历史（latest、stepwise、zero_T）取得四参照与权威M/P/T锚点，共6次解析/12SHA。每阶段shared/current/原head归档1+6+18，投影/归一化契约沿用上轮：只解码5 shared、12 current、108 head成员。head生产者全schema核验、只读6项fit/scaler/预算，不读weights/bias/trace；归档参数以旧端点JSON为准。物理JSON解码及整文件SHA含未投影/未解码字节，不宣称物理字段隔离。

P阶段用NumPy解析目标/梯度，18个SLSQP问题；审计阶段U用Torch独立公式、同版本SciPy求解自己的18个问题，同时S_P用Torch核对P全部访问点、S_U用NumPy核对U全部点。P/U不保证走同路径，不能强迫路径/停止标记相同。S不运行求解器；复核实际同一theta、全部M/P/T与两u梯度、域、branch、缓存映射/返回值和原始门。两历史共用SLSQP包，不能称独立全局算法认证。

每访问点保存4个257向量：u、theta、gradient_M_u、gradient_T_u，以及配置白名单标量/分支/上下文与参数摘要。每头保存4参照参数和center/radius/lower/upper各257向量。回调/原生事件只存标量和摘要，不另存原始u全向量；S只输出同参数数值差/门/域摘要，不复制全梯度向量。完整返回值核对可从状态向量及负M梯度精确重建。所有回调/native/候选都保留，原输入CSV读取0，新主JSON审计只一遍。

## 接受、反证与下一步

唯一阳性为同域/同参数、两实现数值审计和原+重演四参照门共同通过的`verified_primal_witness_in_bounded_domain`。每头/每历史完整保留。它是数值可行见证，仍不证明原500端点经旧50步方向/Armijo/M500/逐步M门能到达，更不证明闭环成功或Module资格。

未找到为`not_found_within_frozen_domain_and_budget`；**域内全局不可行和全空间不可行都未解决**。本版不提供负证书，不将solver status4、status0、零梯度或预算耗尽升级为不可达结论。单一起点的局部搜索即使能找到部分见证，也不能推断失败头表示不足或统一增加训练有用。不做p值/因果结论，不以正式评价调solver。

本轮交付协议/配置/人造测试/科学绑定与dated冻结记录，实际研究搜索、梯度、原数值输入读取/SHA均0。下一步实现入口、field/domain/cache/native/budget guard和独立U/S审计，完成全18头合成及最大预算包络、绑定执行代码后再运行；本科学绑定不授权payload访问。结果之前不改原Policy训练预算/loss，也不打开任何正式评价数据。
