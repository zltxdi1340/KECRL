# task7 原四参照M/T有限见证：执行与独立审计结果（2026-11-11）

**执行完成，独立数值审计通过；P/U两套历史各0/18头找到经复核的共同见证。** A_original/B_fomaml各0/9，三个seed各来源均0/3。它只说明冻结的每头190配方未命中，不能证明固定表示的整个257维参数空间不可达、表示不足或预算已经足够。本轮没有训练修改，原Policy500、未来稳定化5步、λ1 loss与原主0/18保持。

## 做了什么、为什么做

按[科学方案](libero-task7-reference-joint-reachability-plan-2026-11-11.md)和[执行契约](libero-task7-reference-joint-reachability-execution-contract-2026-11-11.md)，完成主入口、访问/字段/归一化/参数与预算guard、独立Torch/NumPy审计，通过完整18头人造流程和最大预算包络，943项依赖执行绑定后主/审计各运行一次。真实运行无异常、无重试或覆盖。

上一轮T较低仍可能M较高，本轮检验同一个固定表示辅助头参数能否同时满足原四参照：M≤min(M_control,M_margin,M_zero,M_bisector)，T<min(T_control,T_margin,T_zero,T_bisector)。M为动作第3分量（索引2）专家MSE，P为相对原Policy逐帧误差的正超额均值/7，T=M/7+P；其余6维原Policy动作固定，不新增P上界。

保留task7全部18头：A_original/B_fomaml×20261106/07/08×fold0/1/2、harm_protected_affine。只计算各自own20 source-support fit的t≥1全部帧，含neutral；encoder已见全部30示教，不能称未见泛化。每头四原参照端点+六条两两连线各31个内部点，固定α=j/32，合190槽；P/U历史分别保留，未使用上轮新联合T候选做第五顶点，不优化、不去重、不提前成功停止或加密网格。

原缓存caps不替换。每角色先重演四参照；同参数两实现M/P/T须满足原绝对1e−10/相对0核对，且各自同时通过原缓存与新重演四参照的原始门。M可等于、T严格更低，不用数值容差放宽门。

## 得到什么

| 来源 | seed | P经两实现复核 | U经两实现复核 | 双历史都有见证 |
| --- | ---: | ---: | ---: | ---: |
| A_original | 20261106 | 0/3 | 0/3 | 0/3 |
| A_original | 20261107 | 0/3 | 0/3 | 0/3 |
| A_original | 20261108 | 0/3 | 0/3 | 0/3 |
| B_fomaml | 20261106 | 0/3 | 0/3 | 0/3 |
| B_fomaml | 20261107 | 0/3 | 0/3 | 0/3 |
| B_fomaml | 20261108 | 0/3 | 0/3 | 0/3 |
| 合计 | — | **0/18** | **0/18** | **0/18** |

全部头分类not_found_on_frozen_grid。P_numpy/P_torch使用同一实际P参数，U_torch/U_numpy复用同一实际U参数；每角色3420前向，全部13680槽保留。

### 网格的直接阻塞项

原M最小参照在P/U全部18头都是bisector；原T最小参照分别为control 11头、margin 4头、bisector 3头，zero无活跃T最小值。以下从新审计输出按协议原有reference/edge配方类型只读计数，没有追加源访问或读出；列出独立审计的三个角色，不将它们当作额外独立样本。

| 角色 / 配方 | 槽位 | 原M门通过 | 原T门通过 | 原M/T联合 | 重演确认联合 |
| --- | ---: | ---: | ---: | ---: | ---: |
| P_torch / 4参照端点 | 72 | 13 | 3 | 1 | 0 |
| P_torch / 内部插值 | 3348 | **0** | 386 | 0 | 0 |
| U_torch / 4参照端点 | 72 | 18 | 0 | 0 | 0 |
| U_torch / 内部插值 | 3348 | **0** | 390 | 0 | 0 |
| U_numpy / 4参照端点 | 72 | 17 | 2 | 0 | 0 |
| U_numpy / 内部插值 | 3348 | **0** | 390 | 0 | 0 |

因此，这些冻结内部点中确有T更低的点，但它们均未满足原M上界；不能把T降低写成专家动作更准确或共同改善。端点M门计数的微小实现差异保留，未用epsilon强迫同判。**“3348个内部点都不满足M”不是“连续边上或全空间不存在满足M的点”**；本轮不是连续约束求解，也未建立全局证书。

### 保留并拦截了一处舍入伪见证

P历史A_original/20261108/fold1的`ref:bisector`，Torch重演M/T各比原缓存小1个binary64 ULP：

| 值 | 原缓存上界 | 同一参照Torch重演 |
| --- | ---: | ---: |
| M | 0.03672178804609304 | 0.03672178804609303 |
| T | 0.005699476998471354 | 0.0056994769984713535 |

仅比较旧缓存会产生联合原始门通过；但它与新重演的自身参照T完全相等，严格T门不通过。对应P_numpy/P_torch联合原始门有1处不同，同参数最大M/P/T差6.938893903907228e−18；重演确认联合门无分歧，两边都未记见证。该例是原协议预设的数值风险，不是新改善，也未修改容差。全局同参数最大数值差1.3877787807814457e−17，208967项审计核对通过。

## 实际预算、时间与provenance

全部24项联合逻辑/IO包络逐项满足冻结预算：

- 四角色共13680核心前向、38194560 fit帧、9777807360特征乘法；10260参数配方向量/2636820逻辑坐标写入，U_numpy复用参数不另构造配方。
- 17100组原/重演门（含独立重算P保存门）、342000次逻辑比较，360次cap归约/1440输入值。
- 原6JSON各解析一次、合12SHA；324 headers、144 selected bodies、36 pairs、4212标量+37008参数坐标=41220源投影值。审计另读新主JSON1次/2SHA及18504参照参数坐标，总55512参数坐标投影。
- 50NPZ加载、250成员解码、100源归档SHA。source head只解码6项fit/scaler/预算成员；未解码weights/bias、额外trace/predictions或current logits/shared proprio。物理JSON解码及整文件SHA仍含未投影/未解码字节，未声称物理字段隔离。
- 四份新probe CSV各3420行、共13680行，无额外参数CSV。主JSON7266211 bytes、审计JSON22064299 bytes，均低于64MiB。
- 实际新优化器/优化更新/梯度/autograd/方向/trial、原Policy/CNN和环境调用均0；不读原checkpoint/CSV/HDF5/RGB/初态、私有或封存数据。环境seed=null、初态=[]、交互0。

CPU/Python3.10.22、NumPy1.22.4；审计Torch2.4.1+cu124、intra/inter-op各1、inference_mode，CUDA隐藏/hash seed0/OMP与BLAS各1。驱动阶段总10.703135秒，两子进程退出均0；从驱动器UTC算主进程3.520518秒、审计进程7.177828秒（包含启动、绑定检查/导入）。审计内部工作4.355075秒、实际峰值RSS585478144 bytes；主512MiB与审计1GiB资源门均通过。短耗时来自固定缓存线性头读出，本轮没有训练原Policy/CNN或运行环境，不能用该时间反推500-update训练是否充足。

## 代码、验证与错误修正

[主入口/驱动器](../experiments/run_libero_task7_reference_joint_reachability_diagnostic.py)、[访问与预算guard](../experiments/libero_task7_reference_joint_reachability_access.py)、[独立审计](../experiments/audit_libero_task7_reference_joint_reachability_diagnostic.py)、[执行测试](../tests/test_libero_task7_reference_joint_reachability_diagnostic.py)。独立审计有自己的插值/loss/caps/原始门公式，不调用主程序计算实现；共享guard负责字段/角色和调用计数。

新增执行测试56、冻结协议79、相邻只读分析69，共204通过，0排除；py_compile、环境检查与git diff --check通过。完整人造流程在两个新进程中经生产驱动器完成实际13680前向/208967核对/24项最大包络，并保留12阳性与6重复参数阴性头；这是合成验证，不能冒充上述真实0/18。未改共享src/训练代码，未运行全量pytest。

绑定前检查生产者源码纠正了把8个必要核心成员误当整个归档schema的实现：按来源生产者全schema核验，仍只解码冻结6成员，不改科学字段/预算。首轮人造测试38通过、1失败/4 setup错误，原因是`/proc/self/status` resolve后的PID路径白名单遗漏，以及测试错锚点同时先触发守恒门；均在真实数据访问前修正。第二轮43通过，扩展及最终各204通过，全部日志保留。真实运行没有技术失败、追加预算或重试。

## 证据在哪里

- [科学配置](../configs/libero-task7-reference-joint-reachability-diagnostic_v1.json)：SHA-256 `f66a06736891529a45fa2cd5b791ca6841163ab99760e1e273c11f1f5880d145`。
- [科学绑定](../configs/libero-task7-reference-joint-reachability-freeze-binding_v1.json)：`613dbf8a471f600ce30433aa4598be0666fee9848b5e4957c3fac79fe994136d`，937项旧闭包保持。
- [执行绑定](../configs/libero-task7-reference-joint-reachability-execution-binding_v1.json)：`251869eabe25c43b06a73c537fe76bb0816bba8e3479112038492195eed37550`，943项依赖在原payload读取/SHA前锁定，含测试日志/XML、人造完整包络与环境证据；绑定前1648旧仓库文件哈希未变。
- `/tmp/libero-task7-reference-joint-reachability-20261111-v1/primary-result.json`：`e65fabc965671d8ecef23acbd4345c299f2ddb5b1cfe333ac9f5a1818dc5a875`。
- 同目录`independent-audit.json`：`71cdb6a873d4b636ef18e65914ca1477b4ad955ca955d9e12bfd07cb3f8a5b33`；包含全部新审计probe、6840同参数比较、18头/来源/seed摘要及实际计数。
- 同目录`completion.json`：`c6f24db92786c8dc3583ae4b32d5624e64e3bdd7137cca1e874cc856c1c0a0ee`；同前缀`-process.json`：`e437b0ea25ee6e55557562af2d2cdc28262fd3ea66266bff8a297db6ec55db8b`。launch、主/审计及run-console日志保留命令/时间/退出状态。
- `/tmp/libero-task7-reference-joint-reachability-execution-tests-bound-20261111-v1.log`及XML：204通过；同前缀initial/v2/final日志/XML保留早期修正过程。`execution-human-envelope-20261111-v1.json`保留人造全预算证据哈希。
- `/tmp/libero-task7-reference-joint-reachability-execution-initial-20261111-v1.json`保留1648文件初始工作区；仅README/roadmap/500计划追加本轮结果，原正文、AGENTS.md/.vscode/及无关phase-gripper修改保留。新数值产物全部留/tmp，未提交/推送、未删除旧文件或改写checkpoint。

## 下一步

收束这版有限插值扫描，不用其阴性结果加密网格、调loss或追加原Policy训练。它未提供共同可达见证，也未提供不可达证书；若继续回答全空间共同可达问题，下一步应**另冻有明确参数搜索域、预算及数值证书边界的约束可行性诊断**，再区分局部搜索未找到与可证明的否定，不能默认clip目标全局凸或原路径一定可达。本轮不自动展开该新诊断。

原Policy500、未来稳定化5步、现loss、原主0/18及G停止/H未开展保持；query、validation、held-out、qualification、reserved、final及其他环境数据继续封存。不更新SPT/Knowledge、不注册Module，不把source-fit有限网格结果写成正式SPT/SPI或完整KECRL验证。
