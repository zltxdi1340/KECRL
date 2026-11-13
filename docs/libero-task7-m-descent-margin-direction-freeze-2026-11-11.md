# task7 负M下降裕度：科学冻结记录（2026-11-11）

**已完成科学冻结与831依赖哈希绑定；尚无执行绑定或新研究结果。** [完整方案](libero-task7-m-descent-margin-direction-plan-2026-11-11.md)、[科学配置](../configs/libero-task7-m-descent-margin-direction-diagnostic_v1.json)、[科学绑定](../configs/libero-task7-m-descent-margin-direction-freeze-binding_v1.json)、[协议/合成参考](../experiments/libero_task7_m_descent_margin_direction_protocol.py)、[测试](../tests/test_libero_task7_m_descent_margin_direction_protocol.py)。本轮不重训原Policy，不访问研究归档或正式评价。

根据[零裕度诊断](libero-task7-t-priority-direction-results-2026-11-11.md)，唯一新变量为M方向裕度：控制沿用零裕度T优先投影，候选固定要求ρ=0.01，并以κ=2ρ=0.02构造方向。G=∇T、Gb=∇(M/7)，候选在Gb·d≤−κ||Gb||||G||、||d||≤||G||下局部优先降低T；原始选定M斜率和精确右导数均须≤−7ρ||Gb||||G||。构造与核验留有固定负余量，不对原始正值加epsilon，不分别调整两个比例，不扫描ρ，也不声称1%是最优值。

1%指方向导数尺度，不是每步MSE降低1%。有限步仍可能M上升、M500拒绝、T不下降或停止；已冻结零/近零、裕度下溢、Gb=0、精确边界导数和无共同下降的处理，不回退更小裕度或改loss继续。

固定task7、A_original/B_fomaml、三个seed20261106/07/08、三折，共18个原辅助500头；两臂从同一原500起点开始，各臂≤50辅助更新/20级试步。原Policy500 updates、未来稳定化5步、表示/λ1 loss、M500有限步接受门和最后接受点规则保持。仅source own20 fit、t≥1；encoder已见全部30示教，probe不评分，source-meta-query/validation/held-out/qualification/reserved/final及环境初态继续封存。不作SPT/Knowledge更新或Module注册。

为避免放宽接受标准，必须同时满足相对零裕度控制和既有角平分端点的T严格更低、M不高及候选逐步M不增/自身500有进展。每来源仍需9/9，两个来源及两实现与同参数离散稳定条件均需满足；只改善弱控制不能替代原参照。角平分18个端点仅从同一次历史JSON投影，不是第三训练臂，不选checkpoint。

控制按引擎分别复现：P81更新/99状态/673试步，U74更新/92状态/624试步；不得强迫两实现历史相同。前次saved500是在角平分下测得，方向字段改按零裕度k0核对。旧zero6/short5/full7及prior_control_updates只作早期−G_T分层注释，不用作当前控制长度。

冻结总预算进一步收紧：未来P≤981辅助更新、U≤974、S0；候选各实现≤900，全部≤1955更新。梯度状态≤3044、autograd≤6081、trial目标≤55970、线性形式≤63075，方向dot≤9132。原归档/新trace成员角色保持，总68次NPZ读取/574成员/172归档SHA、新写36文件/630成员。保留同参数绝对1e−10/相对0、raw分支精确及严格无epsilon决定；不对P/U不同参数路径套局部门。P300秒/512MiB，C900秒/1024MiB，总1500秒，0技术重试/覆盖；不能把控制的未用预算转给候选或新读出。

验证：首轮42通过；补充裕度字段伪造、零梯度和停止顺序后，47新增与相邻检查共128 passed、1 deselected（0.95秒）。排除的是已完成父实验中“输出必须不存在”的旧断言，旧测试/结果均保留，新协议的未绑定拒绝测试仍通过。最后预算汇总字段补齐后47项新增再通过。独立二维闭式解、稠密可行圆扫描、原控制一致、负界严格核验、有限M500门和双参照条件均由人造向量/标量验证。Python编译、CUDA隐藏环境检查及git diff --check通过；未改共享训练实现，未重复全量CPU/Crafter回归。

| 凭证 | SHA-256 |
| --- | --- |
| 科学配置 | 0101f655813b94f0036048eefc2f3e2c922fb4db20da06e7ccbb637cfacc6600 |
| 831依赖科学绑定 | 5c6cc522f1abfbefce430b35211c930877842d78be5953f81357fad034d1f007 |

元数据凭证：`/tmp/libero-task7-m-margin-freeze-20261111-v1.json`；三轮日志/XML及运行环境的SHA列在binding.validation。研究方向/梯度/试步/更新/环境调用、source归档和前次P/U数值JSON的SHA/读取均0。旧冻结文件、AGENTS/.vscode、phase文档及历史产物保持，README/roadmap/500计划只增加当前状态。

**下一步：实现主入口、角色/字段/裕度/引擎预算guard、独立U/S与双参照审计；先做完整18头合成和新上限包络，执行哈希绑定后按冻结预算P/C各运行一次。** 本次科学绑定不替代该执行绑定，当前不声称裕度已有效，也不启动正式评价。
