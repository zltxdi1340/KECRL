# task7 v6 科学冻结（2026-11-19）

已冻结[固定后半程3:1采样方案](libero-task7-window-focus-plan-v6-2026-11-19.md)，状态scientific_frozen_execution_not_bound。

两臂复用完整相同的共享3A，均1,499,144可训练参数，500更新及现七维masked MSE不变。只改变432窗口的有放回抽样概率，后半程起点209个、前半223个，候选整数权重总量850。参数/初态、采样CDF边界、局部RNG、旧uniform序列、实际有效帧口径、非法元数据和阶段执行拒绝均有人造测试覆盖。

123项定向合成/父模块测试及环境检查通过；1017项代码/元数据依赖核验通过。科学配置SHA256 `634bd74abf966a53b2dd08627686b754f729daf4936f8e482adc1f9d1f613cda`，科学绑定SHA256 `d24e82827659cae278ff5991f6e6ef79d6feb5d1781c57432a837f68473f72f9`。证据位于logs/libero-task7-window-focus-freeze-20261119-v6/的science-final、environment、metadata-check、freeze-integrity及initial。绑定前一次计数断言误填122，实际为123；未建立绑定或读载荷，纠正记录保留于freeze-count-correction.json。

本阶段研究HDF5/权重/checkpoint/NPZ/初态读取及SHA均0，训练与物理步0。后续须完整人造执行验证及独立审计、新阶段绑定后才读载荷。44条经审计并满足净增≥4及逐seed不退化后才建立条件120条绑定；不是科学冻结本身授予执行权。

只按时间索引倾斜，不把后半程直接当作放置/近盘标签，不宣称已证明训练稀释是根因。保留旧v5编码报告与SHA，仅新增已绑定来源的可读副本。
