# KECRL 项目交接（2026-11-12）

本稿用于新会话继续维护 `/home/zl202621/project/KECRL`。请先读完整交接，再核对工作区；所有汇报用中文，重要阶段说明做了什么、为什么、结果、证据和下一步。仓库指导以 `AGENTS.md`、`PROJECT_BIBLE.md` 及当前 dated 文档为准。

## 0. 本次交接核对发现的新阻塞，必须先处理

**当前不能直接实现绑定后启动真实搜索。** 2026-11-12交接核对时，`validate_freeze()` 已因父实验文件缺失而fail closed。首次报错：

```
FileNotFoundError:
/tmp/libero-task7-stepwise-m-nonincrease-acceptance-20261111-v1.json
```

路径存在性清单显示：

- 最新科学绑定声明953项依赖，其中30项路径当前缺失，详见文末清单。
- 新诊断需要的6份原源JSON全部缺失。
- 新诊断需要的25份源NPZ全部缺失（1 shared、6 current features、18原辅助头）。
- 最新冻结验证所引用的7份测试/环境日志或XML、初始工作区快照也都缺失。
- 上一轮有限网格的primary-result.json、independent-audit.json、completion.json及process.json当前均缺失。
- 缺失原因未确认；不能推断由哪个操作或系统清理造成。

本次仅检查这些研究文件的存在性，**未读取内容、未计算其SHA、未重训或运行搜索**。最新科学配置、科学绑定及本地协议/测试/方案仍在；本地三文件与绑定记录的SHA相符，但未完成其余现存依赖的全量哈希复核。

此前“151项通过”“953项依赖绑定完成”和上轮“独立审计通过”都是历史运行/冻结记录，不能写成本次当前重验通过。历史报告保留；当前可恢复性/执行前置校验被缺失产物阻断。

第一优先级：定位原实验主机或可信备份，恢复原始产物及父链元数据。保留原SHA与fail-closed契约；不能补空JSON、伪造completion、重算新SHA改旧绑定、移除父依赖或关闭校验。来源数值文件的读取/SHA仍遵守执行访问门及预算，不能因恢复工作无约束地读取封存数据。未经重新冻结与授权，不重跑历史训练“补齐”产物。

## 1. 当前用户目标与实际阶段

最近用户明确要求：

> 另冻有明确搜索范围与预算的约束可行性诊断。继续保持500 updates、稳定化5步、现loss和正式数据封存。

该科学冻结已完成，状态为 `scientific_frozen_execution_not_bound`。本轮新约束搜索尚未实现/执行，无新可行头数。此前建议的下一步是实现主入口、访问与预算guard、独立U/S审计，完整合成验证与执行绑定后运行；**现在新增了先恢复缺失证据/输入的前置事项**。

本次用户只要求输出交接，本次没有恢复、修改协议、重训或运行搜索。除本交接文档及/tmp中的只读清单外，没有改写旧代码、配置、报告或实验产物。

## 2. 项目术语与实验边界

- Knowledge：由证据支持的环境机制、因果关系和可复用知识，不能由Policy/Module成功率直接反推。
- SPT：技能族生成器；SPI：含任务上下文、数据角色、预算、观测/动作接口与训练边界的任务学习规格。
- Policy：执行控制器；Module：通过完整资格契约的可执行技能。未qualification不能注册。
- BC、action MSE、辅助头可纠正性、闭环pilot或本次可行性搜索均不能直接写成SPT更新、Knowledge Evolution、正式SPI结果或Module qualification。
- LIBERO、Crafter、Continual World、RoboCasa365的结果分开。Crafter主要用于长期记忆/探索/因果诊断；后两者不能由LIBERO诊断代替。
- 当前active工作是LIBERO task7固定表示的source-fit机制诊断。原主0/18、G停止/H未开展保持；不能把A_original/B_fomaml这些来源标签当作新的正式主结果。

LIBERO角色：source tasks为0/1/3/4/7；validation为2/6；held-out为5/8/9。source demonstration meta_support=0–29，meta_query=30–49。candidate_nonregression初态0–19已多次使用，不能称未见测试；qualification20–39与reserved40–49封存。当前新诊断不读取任何初态、query、validation、held-out或final evaluation。

当前保持：原Policy500 updates、未来闭环稳定化5步、原Policy训练loss不改，当前辅助λ1目标也不改。不选checkpoint、不部署新头、不更新SPT/Knowledge、不注册Module。原Policy训练步数、原辅助头500及其历史方向端点、新SLSQP迭代是不同计数。

## 3. git与工作区（本次实查）

- 分支：main，状态行 `main...origin/main`。
- HEAD：`2beb6271ee66dfba423d517762ff92acfd83444f`。
- 最近提交：`2beb627 归档LIBERO诊断与G阶段实现，核对几何记录边界`；其前为e525d19、fc79cd6、a8f1a86、3780739。
- 已跟踪修改：README.md、docs/research-roadmap-2026-10-20.md、docs/libero-500-update-next-step-plan-2026-11-01.md、docs/libero-phase-gripper-loss-diagnostic-2026-10-25.md。
- 交接文档创建前有433个未跟踪条目；大量最近研究代码/配置/文档尚未提交，**不能git clean、删除“无用文件”或用checkout/reset覆盖**。
- AGENTS.md、.vscode/为用户文件，不自动修改、删除或提交。
- phase-gripper报告是不相关既有修改，保持原样。前三个状态文件已有连续研究状态，保留其正文与历史。
- 本次没有commit/push；不要把/tmp checkpoint或研究数值归档复制入仓库。

## 4. 最近已完成实验（历史记录，当前原产物缺失）

规范报告：

- [原M负界联合T方向](libero-task7-original-m-bound-joint-t-direction-results-2026-11-11.md)。
- [M/P/T取舍与剩余梯度只读分析](libero-task7-joint-t-tradeoff-gradient-results-2026-11-11.md)。
- [原四参照有限见证结果](libero-task7-reference-joint-reachability-results-2026-11-11.md)。

关键结论：18个联合T候选相对角平分均M更高/P更低，14个T降低，但原四参照主线索0/18；不能据剩余梯度断言增加训练预算能解决。

随后每头四参照端点+6条两两连线各31个内部点，共190配方。P_numpy/P_torch、U_torch/U_numpy四角色共13680前向，208967项独立核对通过；历史驱动时间10.703秒，同参数最大差约1.39e−17。P/U均0/18经复核见证，A/B各0/9；三个审计角色各3348个内部点均未过原M门，部分T较低仍未共同通过。

原M最小参照全18头是bisector；原T最小参照control11、margin4、bisector3。P历史A/20261108/fold1的ref:bisector有1 ULP舍入伪改善，被自身参照重演严格T门拦截。**网格未找到不等于全空间不可达，端点见证也不等于原训练路径可达。**

这套有限网格已有runner/guard/审计/测试：

- experiments/run_libero_task7_reference_joint_reachability_diagnostic.py
- experiments/libero_task7_reference_joint_reachability_access.py
- experiments/audit_libero_task7_reference_joint_reachability_diagnostic.py
- tests/test_libero_task7_reference_joint_reachability_diagnostic.py

历史验证204项通过；相关科学/执行绑定仍在configs，但运行文件目前缺失，不自动重跑旧实验。

## 5. 最新已冻结方案与文件

优先阅读：

1. AGENTS.md、README.md、PROJECT_BIBLE.md相关术语。
2. [最新方案](libero-task7-bounded-constrained-feasibility-plan-2026-11-12.md)。
3. [最新科学冻结记录](libero-task7-bounded-constrained-feasibility-freeze-2026-11-12.md)。
4. [上一轮有限见证结果](libero-task7-reference-joint-reachability-results-2026-11-11.md)。

本阶段已有文件：

- experiments/libero_task7_bounded_constrained_feasibility_protocol.py
- tests/test_libero_task7_bounded_constrained_feasibility_protocol.py
- configs/libero-task7-bounded-constrained-feasibility-diagnostic_v1.json
- configs/libero-task7-bounded-constrained-feasibility-freeze-binding_v1.json
- docs/libero-task7-bounded-constrained-feasibility-plan-2026-11-12.md
- docs/libero-task7-bounded-constrained-feasibility-freeze-2026-11-12.md

科学配置SHA-256：`e345f813a492edf823e7a39cd518e511081d2541659d1d097f57c80fa41064c1`。
科学绑定SHA-256：`26ea46b5324112c3583d13f3e923566f5039cc38d4a8c0dbfe83ac12581bdc1a`。
本次这两个本地文件哈希与历史记录相符。953项是声明闭包，当前有30项缺失，不能称当前全闭包可验证。

科学冻结历史验证新增72项+父协议79项=151通过，0排除；compile、环境和diff检查历史通过，未跑全量pytest。本次未重新跑这些测试，因为canonical工厂会沿缺失父链失败。当前runtime包元数据实查：Python3.10.22、NumPy1.22.4、SciPy1.10.1、Torch2.4.1+cu124；本诊断CPU、隐藏CUDA。完整pytest历史曾受可选Crafter依赖影响，本次未复核该可选包，不把它称为LIBERO代码失败。

## 6. 冻结的问题、域与求解规则

18头：task7，A_original/B_fomaml×seeds20261106/07/08×fold0/1/2，harm_protected_affine；P/U历史分别搜索。只用own20 source-support fit、t≥1含neutral；encoder已见全部30 support，非未见泛化。

M=动作第3分量（索引2）专家MSE；P=相对原Policy逐帧平方误差的正超额均值/7；T=M/7+P，λ1不变，其余6维动作固定。原control/margin/zero/bisector四参照全部保留。

每头、每历史：

```
center = own_history_bisector_parameters
radius_j = max(2^-20, nextafter(abs(reference_rj-center_j),+inf) over all4 refs)
u in [-1,1]^257
theta_j = center_j + (radius_j*u_j)
```

固定Float64操作顺序；半径/边界必须有限可表示，包含四参照坐标，失败不修补/扩域。单一起点u0=0，无随机启动、其他起点或半径扫描。真实radius尚未计算，未来由冻结输入与公式在求解前记录。

使用已安装SciPy1.10.1 SLSQP：min原T，硬约束M≤min(原缓存四参照M_cap、本角色重演M_cap)。options固定maxiter100、ftol1e−12、dispFalse、iprint0；目标和约束解析jac均显式提供，禁有限差分/autograd。ftol不能放宽M/T门。

四角色：P_numpy_search、U_torch_search，以及只核对同参数的S_P_torch/S_U_numpy。P/U各18个solver问题，S不运行solver；共用SciPy包不等于两个独立全局算法。

实现时必须遵守：

- clip边界inclusive1、hinge相等取0的解析约定；非处处光滑，不默认全局凸或给KKT/全局证书。
- 全部回调先统一精确clip原生u到[-1,1]并记录投影；SciPy本身目标包装会clip，约束回调不一定clip，不能混用。
- 按规范u binary64 bytes精确缓存，一次新键同时取得M/P/T及两u梯度；禁近邻合并、隐藏额外前向。
- 缓存保存不可变副本，不能持有Fortran原生u的可变视图；返回jac不得改写缓存。
- 监测原生_slsqp调用与实际major迭代；包装器初始majiter=99是maxiter−1槽位，非已经完成99次。
- major_monitor不增加fit求值；停止后不补未缓存终点；找到见证不提前丢弃其他头或状态。
- 同参数两实现必须各自通过原缓存及新重演四参照M≤/T严格<，绝对1e−10仅用于数值核对，门零容差。保留原始分歧，不用epsilon救结果。
- 阳性为verified_primal_witness_in_bounded_domain；阴性仅not_found_within_frozen_domain_and_budget。本版没有域内/全空间负证书，solver status0/4/9、零梯度或预算耗尽均不证明不可达。

## 7. 冻结预算（全部为未来上限，不是已执行量）

每问题：100主迭代、256不同规范u求值（含u0）、1024原生调用、4096总回调；一域一起点，预算不跨头/历史转移。

合计36个P/U问题：≤3600主迭代、36864原生调用、147456回调。四角色合≤18720前向（含288参照校准）、52266240 fit帧、36864解析梯度向量、55584注册线性形式。原Policy/CNN训练和环境调用0。

源访问：6原JSON/12SHA，50NPZ/250成员解码/100SHA；4212源标量+37008源参照参数坐标。审计另读新主JSON一次/2SHA，向量坐标≤4774032，state/callback/native及头标量≤1710630。原checkpoint/CSV/HDF5/RGB/初态/私有/封存读取0，新NPZ/checkpoint/可部署Policy写入0。

主≤900秒/1GiB，审计≤1800秒/2GiB，总≤3000秒；BLAS/OMP及Torch线程各1，hash seed0、CUDA隐藏。完整字段、操作上限、输出字节/RSS门以科学配置和方案为准。

## 8. 尚未实现的文件、下一步顺序

以下当前都不存在，执行绑定和新artifact目录也不存在：

- experiments/run_libero_task7_bounded_constrained_feasibility.py
- experiments/libero_task7_bounded_constrained_feasibility_access.py
- experiments/audit_libero_task7_bounded_constrained_feasibility.py
- tests/test_libero_task7_bounded_constrained_feasibility.py
- docs/libero-task7-bounded-constrained-feasibility-execution-contract-2026-11-12.md
- configs/libero-task7-bounded-constrained-feasibility-execution-binding_v1.json

推荐顺序：

1. 先定位并恢复原父链/输入产物；不能改旧SHA或绕过fail closed。必要时取得原实验主机/备份信息。
2. 恢复后重新核对953项闭包；科学绑定仍不能授权研究payload访问或求解。
3. 实现主入口、访问/字段/domain/cache/native/budget guard、独立U/S审计及执行契约。只用人造数据做开发，不能把mock绕过后的结果记作真实preflight通过。
4. 跑完整18头合成流程与最大预算包络，核对真实SciPy回调、缓存、投影、major/native计数、原始门和失败停止。
5. 完成新执行绑定，才按冻结预算主/审计各一次运行；记录实际优化状态、资源、IO、哈希、所有头与反例。不自行追加训练或预算。
6. 更新dated结果报告与README/roadmap/500计划；继续封存正式评价，不作Module/SPT/Knowledge晋级。

恢复后建议验证命令（当前第一条已知因缺失父链失败）：

```bash
cd /home/zl202621/project/KECRL
/home/zl202621/.conda/envs/libero-kecrl/bin/python -c 'from experiments.libero_task7_bounded_constrained_feasibility_protocol import validate_freeze; validate_freeze()'
/home/zl202621/.conda/envs/libero-kecrl/bin/python -m py_compile experiments/libero_task7_bounded_constrained_feasibility_protocol.py
/home/zl202621/.conda/envs/libero-kecrl/bin/python -m pytest -q tests/test_libero_task7_bounded_constrained_feasibility_protocol.py tests/test_libero_task7_reference_joint_reachability_protocol.py
git diff --check
git status --short --branch
```

## 9. 缺失产物清单（本次只读存在性检查）

以下30项来自科学绑定的依赖列表。原SHA必须保留；路径缺失不授权生成替代内容。其余现存项未在本次作完整哈希复核。

| 缺失路径 | 冻结SHA-256 |
| --- | --- |
| `/tmp/libero-task7-direction-penalty-tradeoff-driver-20261110-v1.py` | `4db21205b9c86bbb8b9825c0c6b4181e1737c9d7f60a0f8c8f4e424bce9e2fd6` |
| `/tmp/libero-task7-endpoint-gradient-20261109-v1-process.json` | `ba994551f3b9d0fa081af163d805a0d0652819bf6a6d0d830d45b58b5f63033e` |
| `/tmp/libero-task7-endpoint-gradient-20261109-v1.json` | `f56da26798fc9d43f718f37af75d50f0c1c9db7d2d6c1b734e941d7ae7d39e54` |
| `/tmp/libero-task7-endpoint-gradient-20261109-v2.json` | `43ace1221276fdfd68aaaaed9fda6366e0120d608856762d7f956dbee2886601` |
| `/tmp/libero-task7-endpoint-gradient-artifacts-20261109-v1/completion.json` | `531c9f8428a13ef045986c948fe2bb3df82f9b719581773b8dd5ee5e80b0d5f1` |
| `/tmp/libero-task7-endpoint-gradient-artifacts-20261109-v1/independent-audit.json` | `caddfc87b11374225fe486b285613b99fde15ec7cbb7ace8cf41299986fb8a3e` |
| `/tmp/libero-task7-endpoint-gradient-artifacts-20261109-v2/completion.json` | `f5c2ee4ed8b51e545c92d40043ca881774d8d475c63d873049bf550f074a6b29` |
| `/tmp/libero-task7-endpoint-gradient-artifacts-20261109-v2/independent-audit.json` | `698fe898fc7525416556e1175a368831bcf827a7261f2cd9d304383289c1908e` |
| `/tmp/libero-task7-endpoint-gradient-v1-budget-deviation-20261109-v1.json` | `daced34af4a4bbd67794b670383e643dbf21727b890fb3b262293ec2eb12d991` |
| `/tmp/libero-task7-fixed-representation-optimization-20261109-v1.json` | `33021ef69f0040396d33923b5606c62248815c55d0c3655d76cfcc94515b1132` |
| `/tmp/libero-task7-fixed-representation-optimization-artifacts-20261109-v1/completion.json` | `4c79178ede41348124177ccd6c53caf6e1e567f3b4fcf6a0c7fc1e2c63fc1832` |
| `/tmp/libero-task7-fixed-representation-optimization-artifacts-20261109-v1/independent-audit.json` | `88acec5c4276bec92778237448abdf7da71eab2fb189c5c14509e500458ad6d2` |
| `/tmp/libero-task7-joint-t-tradeoff-gradient-20261111-v1-process.json` | `7cff0ca7892c4876357bc782527d8daa1e24879d5cd63e2b5ea68d58f38c42e2` |
| `/tmp/libero-task7-joint-t-tradeoff-gradient-20261111-v1/completion.json` | `fd5f972a4092ea74eb026aed4ba374ada4886b81e9488381b0f7cd4622eebf48` |
| `/tmp/libero-task7-m-priority-direction-20261111-v1-process.json` | `6bedaabb41f0bcaf3437080127e51023079acf7a42faecd3bba1be28c4607aab` |
| `/tmp/libero-task7-m-priority-direction-20261111-v1.json` | `8d869c68ed5543ce6794f5eb0e5bf6024a948aa18781f9be8cbd4c3202d930fc` |
| `/tmp/libero-task7-m-priority-direction-artifacts-20261111-v1/completion.json` | `9e2180fb5b2755e2f1541b4da5710a8df944c233d52bcfea04893459a1cc965b` |
| `/tmp/libero-task7-needed-direction-reporting-binding-20261109-v1.json` | `bd7df78e4583de90142ba68d7f6bdd72c7ff62f7574031c4abebf1fa05db3fe3` |
| `/tmp/libero-task7-original-m-bound-joint-t-direction-20261111-v1-process.json` | `5c432f70c000700eaf1356d695fab9c51b400805f8b8cb90882b0a72e82053bb` |
| `/tmp/libero-task7-original-m-bound-joint-t-direction-20261111-v1.json` | `35be3fa0440ea272cf8b8a95b7aa5a6ffbe3a6b4515a226afde1fbebded32ae9` |
| `/tmp/libero-task7-original-m-bound-joint-t-direction-artifacts-20261111-v1/completion.json` | `ea45029c864275de2368abde9bf7539af77354116dcfd5d6db0824454817ad33` |
| `/tmp/libero-task7-reference-joint-reachability-20261111-v1-process.json` | `e437b0ea25ee6e55557562af2d2cdc28262fd3ea66266bff8a297db6ec55db8b` |
| `/tmp/libero-task7-reference-joint-reachability-20261111-v1/completion.json` | `c6f24db92786c8dc3583ae4b32d5624e64e3bdd7137cca1e874cc856c1c0a0ee` |
| `/tmp/libero-task7-stepwise-m-nonincrease-acceptance-20261111-v1-process.json` | `5b614ce3ba2c9ebe45b81795fc9ebec44316e8e2f8c869997a6dfe92d232cf16` |
| `/tmp/libero-task7-stepwise-m-nonincrease-acceptance-20261111-v1.json` | `e3bc2ec68d94566cad5777695cba3d084b33c6ff526c2229734008a574f7ea50` |
| `/tmp/libero-task7-stepwise-m-nonincrease-acceptance-artifacts-20261111-v1/completion.json` | `733c90695c44475dd946c60d5dbf0f63f1b1ccaf0f3bc912d6469ac71dfaae87` |
| `/tmp/libero-task7-stepwise-m-reference-tradeoff-20261111-v1-process.json` | `617c3cd948cf10f5c563c2c7c74f51827146df4219a0171aa6a43339bc0f5106` |
| `/tmp/libero-task7-stepwise-m-reference-tradeoff-20261111-v1/completion.json` | `290fa88ed90d95d65276ec403b4771f918ad0a2e2ae5b95a87b5425f52909d9e` |
| `/tmp/libero-task7-terminal-gradient-geometry-20261111-v1-process.json` | `65ac86fce94fea579db218bf3445d84c2d91b38c9f619099d6ae12b1a488deae` |
| `/tmp/libero-task7-terminal-gradient-geometry-20261111-v1/completion.json` | `ca83a93c1575bb0517ab95c0c5bf3b49b1488bcd7b045da9db77970fd213ab38` |

另有6份源JSON当前全部缺失，准确SHA已在科学配置source_JSON中保留：

- latest/P：`/tmp/libero-task7-original-m-bound-joint-t-direction-artifacts-20261111-v1/primary-result.json`。
- latest/U：`/tmp/libero-task7-original-m-bound-joint-t-direction-artifacts-20261111-v1/independent-audit.json`。
- stepwise/P：`/tmp/libero-task7-stepwise-m-nonincrease-acceptance-artifacts-20261111-v1/primary-result.json`。
- stepwise/U：`/tmp/libero-task7-stepwise-m-nonincrease-acceptance-artifacts-20261111-v1/independent-audit.json`。
- zero_T/P：`/tmp/libero-task7-t-priority-direction-artifacts-20261111-v1/primary-result.json`。
- zero_T/U：`/tmp/libero-task7-t-priority-direction-artifacts-20261111-v1/independent-audit.json`。

25份源NPZ也全部缺失。完整路径/预期SHA见科学配置inputs.shared、inputs.models[*].current_archive和inputs.heads[*].archive；来源分别是：

- `/tmp/libero-source-expert-component3-observability-artifacts-20261107-v1/`：1 shared+6 task7 features。
- `/tmp/libero-task7-residual-harm-protection-artifacts-20261108-v1/`：18 harm_protected_affine heads。

旧151项验证日志/XML/环境日志和初始快照的路径/原SHA保留在科学绑定validation.evidence及initial_workspace中；当前对应文件均缺失。不要把缺失的旧路径当作现成可点击证据，也不要把记录中的历史计数当作本次重跑结果。

本次辅助清单：`/tmp/libero-handoff-readonly-audit-20261112-v1.json`及`/tmp/libero-handoff-git-status-20261112-v1.log`；/tmp不应作为唯一持久交接载体，本稿已保存关键路径、原SHA、现状及下一步。
