# 切换近邻只读分解：执行修正v2（2026-11-04）

科学定义、样本和预算完全沿用[原协议](libero-phase-history-transition-neighbors-protocol-2026-11-04.md)。[v2配置](../configs/libero-phase-history-transition-neighbors-readonly_v2.json)的scope与analysis逐项等于v1；只增加失败与旧绑定的provenance，并使用新的输出文件。

v1入口在NPZ数值读取之前失败：它要求父结果status为通用`complete`，实际父结果为`complete_phase_history_observability_diagnostic`。原失败记录、配置、绑定、代码和测试保留；实际事件分类、距离计算以及新增研究调用均为0。不能据此把父结果或其独立审计判为失败，也不将失败覆盖为成功。

[v2入口](../experiments/analyze_libero_phase_history_transition_neighbors_v2.py)通过独立小函数严格要求已有专用完成状态、formal_result=false、父结果/科学配置/执行绑定的精确hash以及独立审计的完成状态；通用complete、运行中、失败或任何hash失配仍拒绝。[v2测试](../tests/test_libero_phase_history_transition_neighbors_v2.py)增加这些状态与provenance的回归检查。科学算子、分层、容差、样本、输出统计及停止规则不变。

代码、测试和本文通过v2 binding绑定后才执行。新结果使用`/tmp/libero-phase-history-transition-neighbors-20261104-v2.json`；不重新训练、提取表示、搜索全参照库或读取封存角色。最终报告同时列出v1失败和v2实际成本，不增加科学预算。
