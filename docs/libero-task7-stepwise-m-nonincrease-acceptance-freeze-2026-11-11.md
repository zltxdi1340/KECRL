# task7 仅增加逐步M不升门：科学冻结记录（2026-11-11）

**已完成科学冻结和855依赖绑定，尚无新优化或执行绑定。** [完整方案](libero-task7-stepwise-m-nonincrease-acceptance-plan-2026-11-11.md)、[配置](../configs/libero-task7-stepwise-m-nonincrease-acceptance-diagnostic_v1.json)、[科学绑定](../configs/libero-task7-stepwise-m-nonincrease-acceptance-freeze-binding_v1.json)、[协议/人造参考](../experiments/libero_task7_stepwise_m_nonincrease_acceptance_protocol.py)、[测试](../tests/test_libero_task7_stepwise_m_nonincrease_acceptance_protocol.py)。

依据[407步只读分解](libero-task7-m-margin-finite-step-decomposition-results-2026-11-11.md)，检验单一接受条件：两臂使用相同负裕度方向ρ=.01/κ=.02、原严格T/Armijo/M500门、同loss和20级α网格；候选仅加`M_trial ≤ M_current`。M_current来自本臂当前实测目标，不取控制值或原M500，不重新定义惩罚P。M相等可接受，不加P单调门，无epsilon。

task7、A_original/B_fomaml、seed20261106/07/08、三折，共18个原λ1辅助500头，两臂从同一起点副本重跑，各≤50辅助更新。原Policy500、未来稳定化5步、表示/λ1/ρ/κ、数据角色和最后接受点规则不变。仅source own20 fit、t≥1，encoder已见全部30示教，不称未见泛化；source-meta-query和全部正式角色继续封存，无SPT/Knowledge更新或Module注册。

负裕度控制须在P/U各自实现中复现900更新/918状态/7017试步/18头/407个M上升步，不是旧零裕度81/74或角平分7411试步。旧saved500方向字段不作本次方向参照，使用负裕度k0；只比较旧saved500方向无关字段。历史P/U JSON未来各一解析/前后各一次SHA，同次读取提取零裕度和角平分各18个固定末点，无新训练臂或旧文件额外访问。

同实现同参数前缀在首次控制接受M上升之前必须一致，候选在该试步拒绝后可试剩余网格；分岔后不预测路径。不能把旧407条全当新拒绝数，也不预设候选仍900更新。20次全拒绝允许科学停止；零更新不算进展。候选trial过门但自身下一实测状态M上升则技术停止，不以1e−10容差修补。P/S各用自己的当前M和M500，原始gate/接受/停止分歧分别留痕。

逐步M不升是强制不变量，不算独立改善证据。主线索仍需有更新、自身T进展，且相对本次负裕度控制、保留的旧零裕度及角平分三个参照均T严格更低、M不高；每来源9/9、两来源与P/U及局部离散稳定条件均需满足。原1%T/10%梯度线索另报，不能替代主线索。无自动原Policy或正式评价推进。

未来P/U合计辅助更新≤3600（控制复现各900、候选各≤900），S0；这不增加原Policy500训练预算。总梯度状态≤5508、autograd≤11016、trial目标≤75051、线性形式≤87903、fit线性行≤245951451。每引擎、S和两臂预算独立约束；历史控制剩余空间不能给候选。新门实际评估/标量记录再核验与前缀预算在配置逐项固定，不增加模型前向。

原68次NPZ读取/574成员/172 SHA、36文件/630成员写保持，P/C CSV总≤67710行；trace只作审计，产物留/tmp。CPU、CUDA隐藏、单线程，P300秒/512MiB、C900秒/1GiB、总1500秒，0技术重试/覆盖。未来执行前需完整18头合成复现/新门/前缀检查及新P/U/S最大包络，通过后另行绑定执行代码。

验证：首轮44项通过（1.07秒）；增加两臂证书/停止同构与零方向检查后，48项新增与相邻协议共144 passed、1 deselected（1.99秒）。排除已完成父实验要求“产物不存在”的旧断言，旧测试未改；本轮未绑定执行拒绝仍通过。覆盖原控制flags/reason、原始nextafter边界、M相等可行、三方参照保留、前缀首次veto/更弱T可能性、20级耗尽、零更新、方向保持、预算守恒及伪造协议拒绝。Python编译、CUDA隐藏环境检查和git diff --check通过；未改共享训练逻辑，未重复全量CPU/Crafter回归。

| 凭证 | SHA-256 |
| --- | --- |
| 科学配置 | a37d3433274f0d76a4db22d283a0059878a343b648f2c1256071fdf7a5afee18 |
| 855依赖科学绑定 | 6609cfe8ea7b8d5412ab9b48fccbd1399c4e1c8b5b15a94e901b352bfe152243 |

验证日志/XML、编译和环境SHA见binding.validation；初始工作区及最终元数据凭证留`/tmp/libero-task7-stepwise-m-gate-freeze-initial-20261111-v1.json`、`/tmp/libero-task7-stepwise-m-gate-freeze-20261111-v1.json`。本轮研究归档、历史P/U数值JSON及前次派生分析JSON的读取/SHA、真实梯度/试步/更新/环境调用均0，只有元数据和人造标量/向量参考。原有文件除README/roadmap/500计划增加状态外保持；AGENTS/.vscode、phase文档与旧结果未动。

**下一步：实现新主入口、字段/角色/逐步门/预算与历史前缀guard、独立U/S审计和运行测试，验证并完成执行绑定后各运行一次。** 当前科学绑定不授权执行，也没有候选进展或闭环改善的新结果；原0/18、G/H及正式封存保持。
