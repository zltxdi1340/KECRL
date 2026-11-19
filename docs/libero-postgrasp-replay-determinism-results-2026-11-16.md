# task7确定性短程对照：两臂完成、只读审计修正及v2建议（2026-11-16）

**实际366物理步、0训练/模型前向；两个新进程首环境的153步前缀均与旧RGB/公开状态精确一致，新A/B记录的高精度机器人轨迹和目标端点逐位一致，查询前后记录字段未变。没有重现此前+1像素，不能认定纯EGL或连续上下文根因。** 独立审计初版有一个跨采样通道的错误额外断言；保持原失败和绑定，另建只读v2修正后4512项核对通过，未补跑任何物理步。

## 目的、配置与预算内实际执行

用户批准原366步提案，并额外建议A连续先跑前6条。单连续组即需2123物理步，与366硬上限冲突；本轮已提出可选澄清，未新增授权，按原获批提案执行两次新进程首环境前缀。**没有把两次首环境叫成“已测过连续前6条”的组A。**

| 阶段 | 目的/配置 | 结果 |
| --- | --- | --- |
| 准备 | task7初态0..5一次选择性解码、原153步数值前缀、不读原RGB | 6行/1次原bank；6数值成员、2916项、12126成员未压缩字节；不可变派生缓存 |
| A_no_geometry | 新进程首环境，顺序reset0..5，各5零动作，再原动作1..153；无逐步目标几何查询 | 183 native，153步全通过原公开门；终点图像生成后才查目标端点/接触 |
| B_geometry | 另一个新进程首环境，相同步骤，150..153增加GeometryReader | 183 native，153步全通过原公开门；4份几何、查询前后记录状态未变 |
| 原独立审计v1 | 加了raw qpos转float32应等于公开缓存的额外断言 | 技术失败；原driver失败/console/代码原样保留，不补写原completion |
| 只读审计v2 | 区分两个采样通道，保留旧/新公开零差和新A/B原生逐位门 | 4512项核对通过，新增环境/物理/初态bank/模型/训练均0 |

两臂各1环境/6reset/30零动作/153保存动作=183 native；全366，无自动重试。GPU1/EGL1、seed20261022、horizon300、稳定化5、原版本及单线程。153是技术前缀终点，不伪造truncated/完整episode，也不估计成功率。原驱动22.650187秒（含失败审计），只读修正另行保存。

全318公开观察、636相机帧、31260672公开RGB原始字节。每臂159份×18维robot原生float64，共2862项跨臂比较；B四次几何查询后另存72项robot数据；每臂107项端点前/后/接触值。A在回放期间没有GeometryReader，末图像产生后的端点接触查询之后不再有任何物理步；不会反向影响其已生成153帧。

## 逐项观测结论

| 检查 | 实测 | 能支持的结论 |
| --- | --- | --- |
| 旧/新完整公开摘要（含153腕部） | 两臂全相等 | 本次首环境短前缀没有复现旧+1 |
| 原保存动作/reward/terminated/truncated | 全相等，153仍非terminal | 未改动作或停止语义 |
| 新A/B robot joint/gripper qpos/qvel | dtype/shape/字节全相等，含正负零 | 这两次新重放的已记录机器人轨迹一致 |
| 153步碗/盘body pose、free-joint qpos/qvel、EEF与接触 | 新A/B端点逐位一致 | 只覆盖本次端点，非旧历史全程真值 |
| B几何查询前后robot字段、两臂端点查询前后所存字段 | 全部逐位一致 | 未观察到这些字段的查询副作用，未覆盖所有内部缓存 |
| 连续前6条后的第7环境 | 未执行（预算不容纳） | 不能归因OpenGL/EGL上下文累加 |
| 纯EGL边缘抖动 | 未证实 | 不能据此说物理旧真值已排除所有误差 |

旧档只有float32缓存关节/夹爪，没有raw float64 qvel/目标真值。新A/B相等不证明与旧未保存的高精度状态相等；目标端点相等也不是全程每步目标高精度相等。不能把此处范围偷换成“所有后续评估的物理已完全确定”。

## 原审计错误与只读修正

原审计实施了`raw_qpos.astype(float32)==cached_public_qpos`的额外假设，真实从第0步起两臂均有相同的跨通道差异。robot与gripper在154帧中均非同值，描述性最大差分别0.0010223388671875和0.0002498999238014221（raw先转float32后与公开缓存比较）；这些是**不同采样通道的差，绝不是A/B差**。

robosuite源码显示：每个控制步含多次内部sim.step和observable更新，joint sensor按control_freq采样，`_get_observations(force_update=False)`取observable缓存；直接复制步后sim.data.qpos是另一通道。未记录完整sensor时间，不能声称已经测得固定几毫秒延迟。原87项测试未覆盖缓存时序，是本次审计实现缺口。

新独立文件`audit_libero_postgrasp_determinism_v2.py`只将该错误跨通道等价断言变为描述读出。原RGB/公开state/action/reward/flags、新A/B float64轨迹/端点、查询前后状态等零误差门全部保留。独立上下界公式的1e−12不用于任何物理或RGB准入。新增3项反例测试通过：缓存/raw可以不同且各自A/B相同；原生1 ULP差或RGB1字节差仍拒绝。

原执行896项依赖绑定、87项相关回归通过；只读v2另绑899项依赖、3项测试通过。编译、环境、差异检查通过，未跑全量pytest。原执行审计失败与新只读审计通过分别记录，未回写原结果或旧SHA。

## 几何补录v2修订建议

**不建议仅凭本次结果就把旧v1的RGB门改为告警并继续剩余29条。** 未证实纯EGL、未测连续上下文，旧高精度物理参照也不存在。

可直接推进[另定义的新配对动作重放几何v2建议](libero-postgrasp-geometry-v2-revision-recommendation-2026-11-16.md)：原36轨迹全保留，旧可比公开字段/动作/奖励/原生flags零差；每条新增物理参照R和测量G，raw物理逐位一致。新设计才考虑逐相机“最大通道差≤1且差异像素≤16/16384”的预注册warning候选，超出仍停止；它是测量一致性规则，不宣称已证明EGL或可用于Policy闭环正式门。

要求高精度可比参照意味着72次新重放、25940物理步，0训练/模型；初态0..18一次选择性解码。若只做12970步单次36条，就不能声称raw qvel/目标物理已与旧档零差核验。该v2仅建议、未获新增预算、未绑定、未运行；当前没有结构/模型v2选择。

## 证据

[事前执行协议](libero-postgrasp-replay-determinism-protocol-2026-11-16.md)、[科学配置](../configs/libero-postgrasp-replay-determinism_v1.json)、[绑定](../configs/libero-postgrasp-replay-determinism-binding_v1.json)、[只读修正说明](libero-postgrasp-determinism-readonly-reaudit-2026-11-16.md)、[紧凑证据](evidence/libero-postgrasp-replay-determinism-results-2026-11-16.json)。

真实A/B、prepare、原驱动失败位于`results/libero-postgrasp-replay-determinism-20261116-v1/`和同名`logs/`。只读v2审计4512项及completion位于`results/libero-postgrasp-replay-determinism-reaudit-20261116-v2/`；v2建议机器清单在`results/libero-postgrasp-geometry-v2-revision-proposal-20261116/`。

Policy500、稳定化5、现MSE和全部正式封存保持；不涉及Continual World、不更新Knowledge/SPT、不注册Module。原36条几何补录v1技术中止仍保持；本次366没有与旧2123步混成一个成功运行。
