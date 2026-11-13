# task7 T优先方向对照：科学冻结记录（2026-11-11）

**方案已冻结，819依赖科学绑定通过；执行入口尚未实现/绑定，研究计算0，暂无新实验结果。** 本轮完成上一份[惩罚取舍分析](libero-task7-direction-penalty-tradeoff-results-2026-11-10.md)提出的下一步，不把数学公式或合成测试当作4反例已解决的证据。

[完整方案](libero-task7-t-priority-direction-plan-2026-11-11.md)、[科学配置](../configs/libero-task7-t-priority-direction-diagnostic_v1.json)、[科学绑定](../configs/libero-task7-t-priority-direction-freeze-binding_v1.json)、[参考实现与门控](../experiments/libero_task7_t_priority_direction_protocol.py)、[合成及元数据测试](../tests/test_libero_task7_t_priority_direction_protocol.py)。

唯一变量是下降方向：控制为现有角平分，候选在Gb·d≤0、||d||≤||G||约束内最小化G·d，G=∇T、Gb=∇(M/7)。冲突时将−G的单位方向投影到Gb的正交子空间并匹配范数。它可能以较少M下降换更多T下降；若实际M高于控制，原严格配对线索仍不通过。线性方向不升不保证有限步M不升，更不保证闭环成功。

固定task7、A_original/B_fomaml、seed20261106/07/08、3折，共18个原辅助500头，两臂均从原500开始。原Policy500 updates、未来稳定化5步、表示/λ1 loss/数据角色/20级网格与M500接受门不变。每臂≤50辅助更新、每轮≤20试步，0技术重试、0覆盖；不选checkpoint，不修改SPT/Knowledge或注册Module。probe不评分，source-meta-query/validation/held-out/qualification/reserved/final与环境初态继续封存。

冻结了精确公式、c/q原始符号、1e−12近零停止、无epsilon/夹紧/修补/回退、精确右导数检查及停止优先级。候选投影落在M一阶边界，可能因有限步M500门停滞，须作为反证保留。主配对线索与来源各9/9、总体双来源门沿用；不改为只看T。

同时明确两个历史参照问题：本轮角平分控制应复现v2的900更新/918状态/7411试步，旧398更新的−G_T只用于原zero6/short5/full7分层；v2 saved500末点是在−G_T下测得，仅比较方向无关字段，起点方向及导数改用角平分k0状态。避免更换控制后误用旧字段。

未来P/U各≤1800辅助更新，S0更新；合计≤3600更新、5508梯度状态、11016 autograd、108000 trial目标、120852线性形式。投影多一次单位向量点积已显式加账，总方向dot上限16524，其他前向/梯度上限不变。68次NPZ读取/574成员/172归档SHA、新NPZ写36次/630成员。P/U历史JSON各一次、前后SHA2次；归档/字段投影及同参数P/S审计规则见方案。这里只冻结未来预算，实际消耗均0。

验证：首轮45项合成检查通过；补充严格来源/线索/停止测试后49项新增与相邻检查合计145 passed（133.77秒）。最后统一NPZ字节别名和补齐科学绑定到执行绑定的链条后，82项定向再通过（0.64秒），其中包含全部49新增；无失败、无排除。包括二维独立闭式解、稠密可行圆扫描、零/近零投影、控制公式一致、有限M反弹、T优但M劣仍失败、角色/预算突变与未绑定拒绝。测试使用人造向量/标量，没有研究梯度或试步。Python编译、CUDA隐藏环境检查与git diff --check通过；本轮未改共享训练实现，未重复全量CPU/Crafter测试。

科学绑定钉住父813闭包和当前方案/代码/测试/配置，共819依赖；执行绑定还必须包含该科学绑定及其闭包、未来主入口/guard/独立审计/运行测试/执行契约。当前科学绑定不能充当执行绑定。首次元数据写入命令的conda stdin未转交、未创建文件，已改用libero-kecrl的直接Python完成；无研究执行或重试，说明留在绑定内。

| 凭证 | SHA-256 |
| --- | --- |
| 科学配置 | 335d7f2e310af5b97ad2e5fbaf406f74beecfe1fd7b0a14ed96c9e5f450b0c97 |
| 819依赖科学绑定 | e1ebd2e271864c91eee6861c69496f6037499284e569f7447a92b73c7ead4acb |

元数据记录：`/tmp/libero-task7-t-priority-freeze-20261111-v1.json`；三轮测试日志/XML的SHA在科学绑定validation。源归档SHA/解码、v2历史P/U数值JSON读取、研究方向/梯度/试步/更新和环境调用均0。仅更新README/roadmap/500计划的前置状态，保留旧正文、AGENTS/.vscode、phase文档和全部历史定义/产物。

**下一步：实现主执行入口、角色/字段/方向/预算guard及独立U/S审计，验证完整18头合成和最大包络，完成执行哈希绑定后，按冻结预算P/C各执行一遍。** 在实际配对结果及审计完成前，不决定原Policy训练修改，也不开放正式评价。
