# task7 同loss/固定MSE上界方向对照：执行与审计失败记录（2026-11-10）

**入口、guard和独立审计已实现、哈希绑定并实际运行；主计算完成，独立审计未通过，因此v1为技术失败，没有可接受的新实验结论。** 主执行1298次新增辅助更新（控制398、候选900），审计在第24条路径的候选第35步停止。`direction_unit_sum_norm`跨实现差异超过冻结绝对1e-10；保留所有pending/partial/log/process，未重试、未放宽容差或改科学配置，未生成completion/独立通过凭证。

这是fit辅助优化机制诊断，原Policy500、未来稳定化5步、loss、表示和正式封存保持。未训练原Policy/CNN，未执行环境/闭环，未访问probe/source-meta-query/validation/held-out/qualification/reserved/final，没有SPT/Knowledge更新或Module注册。不能把主输出中看似有利的拟合指标当作通过审计的结果，旧cue/G停止/H未运行保持。

## 设置与实现

[科学方案](libero-task7-joint-direction-plan-2026-11-10.md)、[执行契约](libero-task7-joint-direction-execution-contract-2026-11-10.md)、[科学配置](../configs/libero-task7-joint-direction-diagnostic_v1.json)、[执行绑定](../configs/libero-task7-joint-direction-execution-binding_v1.json)。科学SHA `3a586c3270b9834889848e3dd175aa54ff8fd4d693f83ba7005fa55f7ec09251`保持，785项代码/契约依赖在研究归档SHA/解码前绑定。

task7，A_original/B_fomaml，seed20261106/07/08，三整示教折、18个λ1辅助500头，每头两臂从原保存的同一500副本开始。own20 fit示教来自support0..29，t≥1，三折2782/2817/2777行；encoder已见全部30个support示教。控制−G_T，候选以G_T和Gb的单位向量角平分方向匹配自身当前G_T范数，T=MSE3/7+P、λ1保持，其余六维动作固定为缓存p0。所有新点标辅助500+k，不称Policy550。

两臂20级α=2^-j(j0..19)、每轮从1开始、首接受即停；共同条件为T严格下降、精确右导数Armijo、MSE≤本头固定辅助500上界，无epsilon。候选另在试步前检查所构造方向的精确sT<0/sM≤0；逐步MSE不增是实测科学线索，不是新增接受门。每头每臂最多50辅助更新；无预算转移/早停选优。旧6零/5短/7完整分层固定，候选完整头可以变化，只有起点/fit相同，不要求后续共同前缀。

主NumPy解析梯度及稳定范数构造；审计独立Torch Float64 autograd和方向构造，不调用主数值实现或science.direction参考函数。DirectionWork记录norm/dot/vector真实调用及257维坐标量，每状态最多4/2/4，Jobs检查方向记录、精确导数、当前状态/试步/接受更新和固定上界。原归档只读准入和同fit表示复用、新三点NPZ保存门控、不可变性沿用既有契约。

Python3.10.22、NumPy1.22.4、Torch2.4.1+cu124、CPU，CUDA隐藏，PYTHONHASHSEED0、OMP/OPENBLAS/MKL各1线程；无新随机采样，环境seed null、初态为空。

## 失败位置与已经核对的范围

| 字段 | 记录 |
| --- | --- |
| 路径 | task7 / B_fomaml / seed20261107 / fold2 / norm_matched_joint_bisector |
| 新增辅助更新k | 35 |
| 主states零基索引 | 880 |
| 已完成审计路径 | 23 |
| 已测审计梯度状态 | 881 |
| 已成功标量检查 | 319010 |
| 失败字段 | direction_unit_sum_norm |
| 主字段值 | 1.01468206567 |
| 审计字段值/实际绝对差 | 异常日志未保存 |
| 已知失败条件 | 两个有限数值的绝对差>1e-10 |
| 主G_T范数 | 0.0192915972358 |
| 主Gb范数 | 0.0212497951086 |
| 主方向status | joint_bisector |
| 主clip/hinge精确tie | 0 / 0 |
| 主分支摘要 | d20d1fc336829bed8fbd824c79a3a100a6efc973d334096428acaa936e70b32e |

该字段是两个单位梯度和的范数q。当前主G_T范数约0.0193、Gb约0.0212、q约1.015，没有证据将此处直接解释为接近1e-12角度阈值或零梯度。报错发生在CheckedTable.row逐字段核对：按照冻结表列顺序，此状态的身份/旧标量/梯度范数/clip-hinge计数和branch_digest，以及direction_status、两个参考范数已通过，随后q失败；因此不能把“分支已改变”认定为原因。

独立重演可能累积细小参数/梯度偏差，归一化也可能放大输入偏差，但本次未保存审计端q和该中间状态的参数/梯度坐标，现有记录不足以区分这些机制。不能在没有新证据时宣称算法根因、归咎于CNN/数据覆盖，或直接把审计容差改大。此处是数值审计未通过，不是可选Crafter依赖或环境终止问题。

Checks.equal仅在比较成功后增加n，所以319010为已成功核对项；失败项未计入。第12个来源头两臂已准入，但只完成控制及11个候选末点，当前第12个候选到k35；完整独立验证尚未覆盖全部18对。

主程序自己的History已复现控制18头398更新、416状态/3509试步/36端点/18摘要，共88313项标量核对。独立审计只完成至失败位置，不能把主History全通过表述为控制398路径已获完整独立认证。

## 实际预算与运行provenance

| 项目 | 主计算完整记录 | 审计失败partial | 冻结上限/说明 |
| --- | --- | --- | --- |
| 已完成路径 | 36 | 23，另1条中断 | 36 |
| 新增辅助更新 | 1298 | 857 | 每阶段≤1800 |
| 其中控制/候选更新 | 398 / 900 | 272 / 585（含当前35） | 各18×50 |
| 梯度状态/三梯度向量 | 1334 / 4002 | 881 / 2643 | 1836 / 5508 |
| autograd | 0 | 2643 | 主0/审计≤5508 |
| 试步 | 10920 | 7199 | 各≤36000 |
| 线性形式 | 13588 | 8961 | 各≤39672 |
| 状态fit行 | 3724073 | 2457282 | 5126112 |
| 试步fit行 | 30485720 | 20083693 | 100512000 |
| 标准化/复用 | 18 / 18 | 12 / 12 | 18 / 18 |
| NPZ读取/成员解码 | 25 / 161 | 28 / 186 | 全程≤68 / 448 |
| 归档SHA | 68 | 28 | 全程≤154，实际96 |
| 新NPZ写/成员写 | 18 / 126 | 0 / 0 | 18 / 126 |
| 方向构造/范数/点积 | 1334 / 4920 / 2668 | 未保存 | ≤1836 / 7344 / 3672 |
| 方向向量形式/坐标 | 4088 / 1050616 | 未保存 | ≤7344 / 1887408 |
| 活跃秒 | 7.43838802539 | 未保存 | 300 / 600 |
| 峰值RSS MiB | 139.83984375 | 未保存 | 512 / 768 |
| 原Policy/CNN/环境/封存调用 | 0 | 0 | 0 |

总watchdog墙时48.591429秒（≤960），主进程exit0，审计exit1。主阶段墙时25.072秒、审计23.519秒，含元数据/绑定检查与启动，不能将其全部当优化计算。UTC起止、命令、PID与退出码完整记录于process.json，主PID1094537、审计PID1094753。

实际总NPZ读取53次、成员解码347次、归档SHA96次；主新写18头/126成员。审计只读取原17归档及新11头，尚未执行完整末尾SHA核对；不能报告计划68/448/154已全数完成。审计的direction_work与Limits最终cost没有写入当前partial，缺失按未保存列示，不从别的运行或主轨迹推造；这是后续失败日志应补充的记录缺口。每个实际调用仍受运行guard的局部预算和资源检查。

主七表实际26258行，18个NPZ只存源500/控制末点/候选末点。新表与头未通过完整独立终检；不因主输出完整而生成completion。原训练/数据和本轮pending/partial全部保留，技术重试0；没有原NPZ/CSV额外读取或SHA来排查失败。

## 未审计主输出：仅供排错

主文件记录候选18/18走满50、路径逐步MSE不增线索17/18、主配对线索14/18；两个来源全9头条件均false。这些计数尚未通过完整独立审计，**不作为方向假设的新结论、接受条件达成或扩展训练依据**。不能只报告“14/18改善”而省略本轮技术失败。末点/逐状态/全部试步仍保存在pending JSON和表中，供后续另冻数值核对。

## 绑定前验证与修正记录

57项新增执行检查通过（55局部+1完整合成+1最大包络），CPU回归3661 passed/27 deselected，69个Crafter关联文件显式排除；已冻结元数据/规则和后运行产物不变检查在该选择内。既有只读测试的“默认结果应不存在”断言保留原hash，显式排除，由新增postrun不变检查覆盖；不删历史结果来让旧断言通过。

完整合成v1因零目标数据的等价两臂出现严格末点cue跨实现翻转，正确失败；v2改为非共线梯度数据后，全路径数值通过，但最终IO汇总有metadata.NEW_HEAD_MEMBERS命名空间引用错误，已修正为finer.NEW_HEAD_MEMBERS；v3通过1032401项核对。两次失败均在真实payload前，未改变科学规则，原日志/XML保留。与本次实际q字段失败不同，不能混为同一已解释原因。

最大1836状态/36000试步资源包络通过：主12.431秒/57.133MiB、审计10.402秒/472.555MiB，实际5508次autograd；方向1836构造/6426范数/3672点积/4590向量形式/1179630坐标，全部在上限内。合成不是研究结果，也没有覆盖所有真实轨迹上的数值敏感性。3661测试通过不替代研究级独立审计。

Python3.10编译、CUDA隐藏环境检查、git diff --check通过。绑定前1431个原有非results文件全部未变；本轮最终只新增7个执行/契约/绑定/结果文件，并在README/roadmap/500计划前置失败状态。科学方案/配置、已绑定源码和旧结果在研究payload后保持不变；AGENTS/.vscode/原phase文档保持。

## 文件、哈希与状态

| 项目 | 路径 | SHA-256 |
| --- | --- | --- |
| 科学配置 | /home/zl202621/project/KECRL/configs/libero-task7-joint-direction-diagnostic_v1.json | 3a586c3270b9834889848e3dd175aa54ff8fd4d693f83ba7005fa55f7ec09251 |
| 执行绑定 | /home/zl202621/project/KECRL/configs/libero-task7-joint-direction-execution-binding_v1.json | 4061a0134a5f6000459977b987ffcbd23baaeb308b7fac2b5f06e1c1d12d8400 |
| pending主结果 | /tmp/libero-task7-joint-direction-20261110-v1.json | a920d27a4f0fb61840c9adc33c96262d97e28e7d47ef75c3ea55f8d004c0ddbe |
| audit失败partial | /tmp/libero-task7-joint-direction-artifacts-20261110-v1/audit-failed-partial.json | bcefd39f8a4b5af6b6a47a81976ab88019c441ecb8e2ae0d6e64d88a70b5eba9 |
| process失败记录 | /tmp/libero-task7-joint-direction-20261110-v1-process.json | ceb94cbb2b521de5eeeac812f70aedc136ce1ef541c5291d88c2c13f6d121bb6 |

实现：[主入口](../experiments/run_libero_task7_joint_direction_diagnostic.py)、[guard/方向计账](../experiments/libero_task7_joint_direction_access.py)、[独立审计](../experiments/audit_libero_task7_joint_direction_diagnostic.py)、[执行测试](../tests/test_libero_task7_joint_direction_diagnostic.py)。测试/包络/XML/合成失败/绑定前证明均在binding.validation钉住。研究日志为 `/tmp/libero-task7-joint-direction-20261110-v1-{primary,audit}.log`；没有independent-audit.json或completion.json。

## 下一步

保留v1技术失败，不以剩余预算重试、不改v1绑定，也不把容差直接放宽。建议另冻失败头前35步的有限数值诊断：明确区分同一参数/同一梯度输入上的归一化差异与两条独立重演路径的累积差异，完整记录q的主/审计值、绝对差、参数/梯度偏差和分支摘要；再决定是否需要修正数值实现或审计口径。现有v1未保存该中间状态的全参数/审计梯度，复核若需重新读源缓存/计算必须另冻独立预算和代码绑定，不能称零成本只读现成轨迹。

当前已完成本次实现、绑定和一次执行尝试；没有通过审计的新方向结果。原Policy500、未来5步稳定化、现有loss和正式封存维持，既有正式G停止决定不变。
