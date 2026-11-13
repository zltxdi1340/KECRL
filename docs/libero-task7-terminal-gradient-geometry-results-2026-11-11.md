# task7 终点梯度/分支只读几何结果（2026-11-11）

**主分析及独立整数审计均完成；P/U历史两套缓存各18/18候选终点在线性模型内联合可行，均无clip tie或hinge zero记录。** 纯M方向的停止反映所选方向与T冲突，不能由此认定缓存半径和M负界内不存在联合方向。本轮没有生成或执行新方向，未验证替代方向exact门、固定网格有限步或闭环；原科学主判据仍0/18，G停止/H未开展。

## 做了什么、为什么做

按[冻结方案](libero-task7-terminal-gradient-geometry-plan-2026-11-11.md)和[执行契约](libero-task7-terminal-gradient-geometry-execution-contract-2026-11-11.md)新增字段/身份/坐标/预算guard、主入口和独立整数审计。完整人造流程通过、902项执行依赖绑定后，各运行一次；不改旧协议、原Policy500、未来稳定化5步、表示/λ1 loss或原接受/停止规则。

本轮只读上一轮P/U两份JSON中18个候选终点及固定zero6的6个saved500点/源。先核对全部36头、54端点路由和控制900/7691、候选33/266元数据，再投影指定点的38标量及四个257维缓存向量。保留A_original/B_fomaml × 三seed × 三折全部18头，区分6zero和12updated；这不是重新抽样的泛化集。

主参考对binary64缓存坐标用Fraction精确判定，独立审计用共同2幂分母整数重算内积及容量/可行差符号。模型为||d||≤缓存r、7Gb·d≤缓存负界B、G·d<0；G、Gb、Gp分别为已存∇T、∇(M/7)、∇P，T=M/7+P。不改缓存G或r/B、不加epsilon、不以余弦代替精确分类、不生成替代方向。P/U是上一轮不同数值实现的缓存历史；本轮另有主有理数和独立整数两套只读算术，不能把它们计成新增训练seed。

## 得到什么

| 来源 / seed | P终点线性可行 | U终点线性可行 | P/U各自zero / updated | P的cos(G,Gb)均值 / 中位数 |
| --- | ---: | ---: | ---: | ---: |
| A_original / 20261106 | 3/3 | 3/3 | 1 / 2 | −0.701587 / −0.787751 |
| A_original / 20261107 | 3/3 | 3/3 | 0 / 3 | −0.288718 / −0.346823 |
| A_original / 20261108 | 3/3 | 3/3 | 1 / 2 | −0.439706 / −0.325531 |
| B_fomaml / 20261106 | 3/3 | 3/3 | 2 / 1 | −0.471207 / −0.327595 |
| B_fomaml / 20261107 | 3/3 | 3/3 | 1 / 2 | −0.347392 / −0.173240 |
| B_fomaml / 20261108 | 3/3 | 3/3 | 1 / 2 | −0.308073 / −0.275013 |

每套历史均A9/9、B9/9；zero6为6/6、updated12为12/12线性可行。全部终点精确z=G·Gb<0、Gram>0、C>0、D>0，分类均为`strict_joint_linear_feasible`，没有空M半空间、零G或严格零边界。P/U在18个身份上的精确证书、分支标签、分支摘要全部一致；不要求两套历史参数逐位相同，也没有将折当作独立seed或做显著性检验。

P中cos(G,Gb)范围−0.923315至−0.001710，cos(G,Gp)为0.481142至0.993950，cos(Gb,Gp)为−0.995494至−0.208970；U相同至所示精度。所有余弦均有定义、无越界，未夹紧。**减小M的最陡方向与减小T的最陡方向均存在冲突，但还保留了足以满足冻结M负界的横向联合下降空间。** 这是缓存梯度模型的几何解释，不是惩罚权重应改变或真实动作会改善的证明。

上一轮所选纯M方向在全部18终点仍selected/exact T导数为正、M证书通过。P的selected T范围3.82310e−6至4.29436e−3；该停止条件被保留。描述性数值核对中，G−(Gb+Gp)最大绝对坐标差P为5.62e−17、U为5.38e−17；缓存范数/selected点积最大偏差均1.39e−17，均低于原绝对1e−10核对门，符号未用该容差修补。

全部终点clip两侧tie计数和hinge_zero_n均0，具有冻结的充分局部光滑计数线索。故这些终点未观察到所查clip/hinge边界tie，但本轮没有读取原始样本来检验任何替代方向的实际精确右导数。`alternative_direction_exact_gate_checked=false`、`finite_trial_checked=false`在全部行保持；线性可行与无tie仍不能代替新方向selected/exact证书或原20级网格/Armijo。

## zero6的同点控制证据

| 固定反例 | P与U各自候选/控制三梯度位一致 | 缓存控制selected / exact联合证书 |
| --- | --- | --- |
| A / 20261106 / fold2 | 是 | 通过 / 通过 |
| A / 20261108 / fold2 | 是 | 通过 / 通过 |
| B / 20261106 / fold1 | 是 | 通过 / 通过 |
| B / 20261106 / fold2 | 是 | 通过 / 通过 |
| B / 20261107 / fold1 | 是 | 通过 / 通过 |
| B / 20261108 / fold2 | 是 | 通过 / 通过 |

12个同点对照均核对三梯度float.hex、n/M/P/T、分支和零更新相同，缓存saved500控制也均在线性模型中可行。参数同点依据继承父实验已通过的同500起点、零更新不变性及参数trace审计，本轮未读参数。这些控制selected/exact记录是既有证据，本轮没有新试有限步，也不以其浮点方向范数替代精确半径模型。12个更新后终点的结论来自各自缓存梯度，未借用起点控制方向。

## 运行、预算与验证

两阶段均退出0，独立整数审计6235项核对通过；总墙钟1.883秒，其中主数值阶段0.357秒、独立审计0.231秒。各进程峰值RSS分别86913024、86138880 bytes，低于512MiB。时间短是因为本轮仅做缓存几何，没有训练或环境交互。

| 实际实账 | 每阶段 | 两阶段 |
| --- | ---: | ---: |
| 源JSON解析 / SHA | 2 / 4 | 4 / 8 |
| 点 / 投影字段与坐标值 | 48 / 52968 | 96 / 105936 |
| norm / float dot | 192 / 288 | 384 / 576 |
| float dot坐标乘积 | 74016 | 148032 |
| G和坐标核对 / zero位核对 | 12336 / 9252 | 24672 / 18504 |
| 精确内积 / 精确坐标乘积 | 144 / 37008 | 288 / 74016 |
| 输出行 | 82 | 164 |

审计另读新主结果1次、前后2SHA；两输出265466/265855 bytes，均小于4MiB。CPU、Python3.10.22新进程，无NumPy/Torch数值导入；CUDA隐藏、hash seed0、OMP/BLAS各1。命令、UTC、PID、环境、配置和实际账均保存。原始数组/CSV/HDF5/checkpoint/特征/target/参数/私有/封存读取（含SHA）、新前向/梯度/autograd/优化/trial/环境及新方向向量均0；无覆盖、技术重试或访问拒绝。两源JSON物理解码包含未投影字段，此限制在记录中明确，语义上仅用冻结白名单。

绑定前新增70项和相关165项共235通过，排除2项父实验历史“产物不存在”断言，旧测试未修改、旧产物未删除。覆盖精确边界/nextafter/极端尺度、字段/坐标/身份/分支和来源篡改、只读IO与禁止模型导入、预算/资源上限、zero逐位差异、完整18头主分析/新进程独立审计及驱动器日志/失败后不重试。py_compile、环境检查和git diff --check通过。本轮不改共享src或训练实现，未运行全量pytest。

初轮65通过/1失败源于测试错误要求描述性汇总逐位相同（fsum与statistics.mean最后一位不同），主分析/独立审计本身通过。修正测试为冻结的浮点规则、保留精确分类/符号零容差后70通过；随后相关测试首轮235通过/2历史产物不存在断言失败，排除这两项后235通过。所有初始日志留证。这些均发生在绑定和研究payload访问前，真实运行各阶段只执行一次。

## 证据在哪里

- [执行绑定](../configs/libero-task7-terminal-gradient-geometry-execution-binding_v1.json)：902项依赖；SHA-256 `cc547d0cdef606c8712dca9183d9c46b37ab6db330456c2256977a552233418e`，记录验证日志/XML、环境及初始工作区快照哈希。
- [主入口](../experiments/analyze_libero_task7_terminal_gradient_geometry.py)、[访问与计算guard](../experiments/libero_task7_terminal_gradient_geometry_access.py)、[独立整数审计](../experiments/audit_libero_task7_terminal_gradient_geometry.py)、[运行测试](../tests/test_libero_task7_terminal_gradient_geometry.py)。
- `/tmp/libero-task7-terminal-gradient-geometry-20261111-v1/analysis.json`：`b1cf02f1f8b698f5144159d3b50662ef52063b5f7e8276558a56dd5fa0cebff1`。
- 同目录`independent-audit.json`：`aca4e4584dea3ab9452c775e7cdfc909ece56dd51593051d368fe0bf923bb22a`。
- 同目录`completion.json`：`ca83a93c1575bb0517ab95c0c5bf3b49b1488bcd7b045da9db77970fd213ab38`。
- `/tmp/libero-task7-terminal-gradient-geometry-20261111-v1-process.json`：`65ac86fce94fea579db218bf3445d84c2d91b38c9f619099d6ae12b1a488deae`；同前缀launch与primary/audit日志保留。
- `/tmp/libero-task7-terminal-gradient-geometry-execution-tests-final-20261111-v2.log`及XML：235通过/2排除；初始失败/复核日志均已纳入绑定验证元数据。

原有1613文件在绑定前哈希未变；最终仅允许README、roadmap、500计划追加本次索引，历史正文、旧科学文件、AGENTS.md、.vscode/及不相关修改保留。所有新数值结果留/tmp，不提交checkpoint或生成数据。

## 下一步

建议另冻**满足现有M负界的联合T下降方向对照**，先明确解析构造、浮点selected/exact核验、失败停止及完整预算，再考虑执行。保持同18个原辅助500起点、同表示/loss和原T/Armijo、M500/逐步M门/四参照，防止只与纯M弱控制比较。需要实际验证所构造方向是否通过原始证书、固定网格内是否存在合法有限步；本轮存在性判据不能替它通过。

当前证据不支持增加原Policy预算或修改loss，也不能直接解释机器人闭环失败或证明SPT/SPI主实验完成。本轮是task7 source-fit辅助头的只读机制诊断，非Knowledge/SPT更新、正式SPI结果或Module qualification；正式评价角色继续封存。下一实验尚未冻结、未启动，不自动追加训练。
