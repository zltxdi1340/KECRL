# LIBERO source 公共 t0 的专家首动作误差控制（2026-11-02）

覆盖诊断的距离不能代表动作是否正确，因此补充 source 示教的有标签首步控制。候选初态没有专家正确动作，本项只评价 source-support demos `0..29`，不把 source 标签移植到 candidate。

## 在读取目标前冻结的设置

[配置](../configs/libero-task4-task7-source-t0-target-error-diagnostic_v1.json) SHA-256 `e5842d59d1a86a990039f5282ac3e580d6967f1dc3063a7468f2fbb51bc0682f`；[脚本](../experiments/analyze_libero_task4_task7_source_t0_target_error.py) SHA-256 `cfcdc74f6a6193383566b1636e204905e2c35bae5765e0995f41b6194fb95ccd`。

六个 baseline500 在 source 公共 t0 上的 180 行预测已经由覆盖诊断保存，不再执行模型。新增 HDF5 数值读取预算仅为两个任务各 30 条 source 示教的 `actions[0]`，共 60 行；先调用原角色授权器，只读取动作，不读 observation、attrs、init_state、states、robot_states 或 official state 文件。每个目标均与同 demo 的已核验公共 runtime t0 预测配对。

对照来自 H2 的同一 baseline：source `t=1` 各 30 行，以及全部 `t>=1`。按 translation/rotation/gripper 自身维数和原 uniform 七维分别报告 MSE/MAE 等，保存首步目标与预测。t0、t1 的观察和目标都不同，这不是同一目标下的输入替换因果对照。

只将 t0 translation MSE / all-t>=1 translation MSE 在同任务三个 seed 均 >=1.5 时记作一致首步拟合缺口；分母为 0 时记 null。该指标本身不许可训练、不指定闭环根因，也不评价 candidate 首动作对错。role/query/封存边界、500-update 权重及全部数据保持；新 forward/backward、optimizer/environment steps 均为 0。

结果和两份小型目标/预测 NPZ 另存 `/tmp/libero-task4-task7-source-t0-target-error-20261102-v1.json` 与对应 artifacts 目录，拒绝覆盖历史输出。此项增加的是 60 条已授权 source 首动作标签读取，不增加新观察采集或学习预算。

## 执行结果

以下均为 source-support 平移三维自身归一化的 MSE。各首步列分母为 30 条 demo，all 列为 task 4/7 的 4,442/4,188 条 `t>=1`。

| task | seed 后两位 | t0 平移 MSE | t1 平移 MSE | all-t>=1 平移 MSE | t0 / all 比值 |
|---|---:|---:|---:|---:|---:|
| 4 | 22 | 0.046107 | 0.033076 | 0.038715 | 1.191 |
| 4 | 23 | 0.087918 | 0.066514 | 0.033710 | 2.608 |
| 4 | 24 | 0.094906 | 0.069850 | 0.037132 | 2.556 |
| 7 | 22 | 0.060009 | 0.075744 | 0.037785 | 1.588 |
| 7 | 23 | 0.025054 | 0.030622 | 0.034419 | 0.728 |
| 7 | 24 | 0.030083 | 0.041259 | 0.037530 | 0.802 |

两个任务均未达到“三 seed 比值全部 >=1.5”的预设条件。task 4 两 seed 的首步拟合较差，task 7 则有两个 seed 首步优于各自全时序平均，故不能说首动作拟合统一失效。t0 夹爪 MSE 在 task 4 约 `5.23e-7–1.58e-5`、task 7 约 `0.00154–0.00229`，也不能由首步特征距离推导“夹爪首步已错”。这些是 source 拟合结果，仍不能评价 candidate 的首动作对错。

实际只新增读取 60 个 `actions[0]`，复用 180 行缓存预测，elapsed `0.7420s`，模型/优化器/环境调用均为 0。结果 SHA-256 `2e39434deb173af9af8ab8e9e8c5cbe820ba2d334aa11ca44bba237d367b57cc`，两份 target/prediction NPZ 摘要在主结果内。

独立复核见 `/tmp/libero-night-last-stages-independent-audit-20261102-v1.json`，SHA-256 `1b98445e7dd3445af32b3e2d7ac125068534cd6171c90999616f51ef7c2b00cc`。在原角色约束下重新读取同一 60 个 source 标签，核对 t0 预测来源，重算三个时段×六个模型的指标及判断；与时序/反馈诊断合计 126 个误差单元重算差为 0，没有读取新角色或执行模型。

本结果不许可首步重加权训练。下一步先控制稳定化和参照时相的混杂，具体范围见[晨间汇总](libero-overnight-diagnostic-progress-2026-11-02.md)。
