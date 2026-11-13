# task7 终点梯度与分支：只读几何核对冻结（2026-11-11）

**本轮只冻结元数据和人造验证，不读取研究数值JSON或运行新分析。** [M优先方向结果](libero-task7-m-priority-direction-results-2026-11-11.md)中18头均因所选方向使T导数为正停止；其中6个零更新起点已有缓存控制方向通过原selected/exact证书。需要区分“负M最陡方向失败”和“缓存梯度下是否存在联合线性方向”，并保留不可微与有限步解释边界。

## 对象与字段

保留task7全部18头：A_original/B_fomaml × seed20261106/07/08 × 三折，不选择表现较好的头。P和独立U历史各自保留。候选终点18个；再读取固定zero6对应的saved500控制起点6个，每源共24点。zero6为A/20261106/f2、A/20261108/f2，B/20261106/f1、f2，B/20261107/f1，B/20261108/f2，是已知outcome分层，不是新泛化集；另12个更新后终点单列。

未来只解析本次M优先运行的`primary-result.json`、`independent-audit.json`，每阶段各一遍、前后各一次SHA。head36行和endpoint54行只查冻结路由字段；head元数据全36行核对控制900/7691、候选33/266及停止；point body只投影18个`M_priority_final`和zero6的saved500。

每点投影38个白名单标量（身份、condition、更新数、n、M/P/T、缓存范数、selected/exact斜率、原M界/ρ、证书/状态、分支摘要及clip/hinge计数），以及G、Gb、Gp和已存方向四个257维向量。不得投影parameters、features、targets、其他states/trials或其他端点body。JSON物理解码仍包含未投影内容，语义访问由字段guard限定；不另读NPZ/CSV/HDF5/checkpoint、原始输入或环境状态。

沿用原λ1定义T=M/7+P，G=∇T、Gb=∇(M/7)、Gp=∇P。检查G≈Gb+Gp、缓存范数和已存方向的梯度点积与selected斜率，绝对1e−10、相对0，仅用于数值核对；不以此重写G或放宽符号。selected与exact斜率的差额分别记录，clip/hinge原始计数和身份精确。ρ仍=.01，缓存B必须与原表达式匹配；不扫描裕度或重新定义T。

## 联合线性可行性的解析判据

以**缓存binary64坐标按精确实数解释**，使用缓存半径r=direction_total_reference_norm与缓存负界B=required_M_slope_bound，不将B替换成理想化ρ计算值、不二次归一化。只判断下列线性模型的方向存在性，不生成方向：

\[
\|d\|_2\le r,\quad 7G_b^\top d\le B<0,\quad G^\top d<0.
\]

令A=GᵀG、B₂=GbᵀGb、z=GᵀGb、m=B/7，定义C=r²B₂−m²、D=r²(AB₂−z²)−m²A。M半空间在球内非空当且仅当C≥0；若A>0，则严格T下降方向存在当且仅当C≥0且（z>0或D>0）。A=0没有严格T下降，C<0为M约束在缓存半径内不可行；z≤0时D=0对应最小线性T为0，D<0为正，不能当作严格下降。

推导：给定Gb·d≤m，最小G·d在球/半空间上取得。z>0时沿−Gb方向可给严格T下降；z≤0时边界最小值为mz/B₂−sqrt(r²−m²/B₂)×sqrt(A−z²/B₂)，其严格为负等价于D>0。零向量/空半空间单列；判据不需要对夹角开平方、epsilon、cosine夹紧、近零投影阈值或方向搜索。

主参考用Fraction.from_float精确有理数，独立审计用各向量共同2幂分母的整数坐标，重算A/B₂/z、Gram及C/D符号。输出精确符号、线性可行分类和原因，不把缓存浮点梯度当作未知真实函数的精确导数。描述性夹角另用四norm、六dot计算G/Gb、G/Gp、Gb/Gp余弦，原始越界单列，不夹紧或据浮点余弦替代精确分类；零/不可表示分母为undefined，不加epsilon。

这是以原缓存界和半径建立的线性模型；原程序的浮点归约在边界可能与精确模型不同，保留已存selected决定作为独立事实，不回写旧门或结果。

## 分支与结论边界

clip五类计数和hinge三类计数分别总和为n；不存在clip两侧tie且hinge_zero_n=0时标为`smooth_by_cached_counts`，这是局部光滑的充分计数线索。出现tie仅标替代方向exact门未知；计数并不能完整决定任意新方向的clip/hinge单侧导数，不能从所选M方向的exact值反推出另一方向。

不论分支标签如何，`alternative_direction_exact_gate_checked=false`、`finite_trial_checked=false`始终保留。线性可行不代表通过新方向exact证书、固定20级网格Armijo、闭环、正式qualification或原Policy改善；线性不可行也只针对缓存梯度/半径/界，不宣称环境或全局任务不可行。不会产生新方向向量或试α。

zero6将候选零更新终点与saved500的三个梯度按原始浮点位、M/P/T、n和分支对齐，独立读取缓存控制selected/exact证书。相同参数的依据继承父运行已通过的原500起点/零更新不变性与独立参数trace审计，本轮不另投影参数。缓存方向确曾通过父运行证书，不等于本轮新做有限步检验。另12终点不套用这六个起点的控制方向；实际可行数本轮未知。

## 未来访问和计算预算

每源头路由36×6、端点路由54×6，共540字段值；head body36×10、point标量24×38，共1272；四向量24×4×257=24672坐标。每源合计26484投影值，每阶段两源52968，主/审计合计105936。字段/行/shape/身份必须全部齐全，拒绝额外投影或重复源解析。

| 未来上限 | 每阶段 | 两阶段合计 |
| --- | ---: | ---: |
| 源JSON解析 / SHA | 2 / 4 | 4 / 8 |
| 读取点（18终点+6起点/源） | 48 | 96 |
| norm / float dot | 192 / 288 | 384 / 576 |
| float dot坐标乘积 | 74016 | 148032 |
| G≈Gb+Gp坐标核对 | 12336 | 24672 |
| zero三个梯度坐标位核对 | 9252 | 18504 |
| 精确A/B₂/z归约 | 144 | 288 |
| 精确坐标乘积 | 37008 | 74016 |

每阶段输出36终点几何、12zero控制对照、12来源×seed、4来源、18双实现比较，共82行；不写研究CSV。审计另读本轮新主JSON一次/前后2SHA。0原始数组/模型/输入/私有/封存读取，0新前向/梯度/autograd/优化/trial/环境调用，0新方向向量。

CPU、Python3.10.22，无NumPy/Torch的独立新进程，CUDA隐藏、hash seed0、OMP/BLAS各1线程。每阶段≤120秒/512MiB，总≤300秒；每源JSON≤192MiB，每输出JSON≤4MiB。technical retries=0，0覆盖，全部新产物留/tmp。时间/RSS或字段/身份/provenance、非有限、分支计数/数值守恒或独立审计不匹配均fail closed保留partial；真实科学分类不是技术失败，P/U不同历史路径的分类差异单列，不强求相同。

## 汇总与后续门控

全部来源×seed×折保留，按来源×seed三折等权、来源九头等权报告分类与夹角；6zero/12updated单列。不把折当独立seed，不做p值，不选checkpoint、方向或新超参。当前0/18及四参照判据、G停止/H未开展保持，解析可行性不是新的科学晋级规则。

原Policy500 updates、未来稳定化5步、现有表示/λ1 loss、训练与停止规则不变。source仅own20 support fit，encoder见过全部30示教，不称未见泛化。query、validation、held-out、qualification、reserved、final及其他环境数据继续封存，不更新SPT/Knowledge、不注册Module。

本轮只创建协议、配置、测试、科学绑定及冻结记录，工厂只读完成元数据与仓库文件；不读取或SHA历史P/U数值JSON，不计算真实终点几何。下一步实现字段/坐标/预算guard、主入口和独立整数审计，通过完整合成、绑定执行哈希后按两阶段预算运行；科学冻结绑定不能代替执行绑定。
