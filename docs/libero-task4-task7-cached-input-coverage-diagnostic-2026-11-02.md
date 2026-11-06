# LIBERO 固定 Policy 的初始与闭环观察覆盖诊断（2026-11-02）

H2 未发现预设的 projection 邻域区分一致下降，错误 tanh 饱和也未达到训练条件。按[夜间计划 v2](../configs/libero-task4-task7-overnight-diagnostic-plan_20261102_v2.json)继续检验 H3：当前闭环观察与固定 source bank 的特征覆盖差异，是首次动作前已经存在，还是主要随运行增长？本项只复用已有公开缓存，不采集新环境数据。

## 执行前冻结

- [配置](../configs/libero-task4-task7-cached-input-coverage-diagnostic_v1.json) SHA-256：`43a5b0e1efd1e1786e700a67ff8ad8e068ea74eb125764fe90d960d69b28e8c8`。
- [脚本](../experiments/analyze_libero_task4_task7_cached_input_coverage.py) SHA-256：`b09d1ca768f98aaf56690b3345fd02dbd73caf79b7f8e8b44e6fcc664aa96109`。
- 固定 task 4/7 × 三 seed 的六个 baseline500，H2/前缀/公共 t0/权重及代码共 17 个输入 pin。原 BCE 分支不恢复，不改 checkpoint、Policy、数据角色、环境或训练。

候选帧来自已有 `candidate_nonregression 0..19` 的 120 个 20 步 archive，共 2,400 帧。新增前置核对已证明 `120/120` reset、action、joint/gripper 前缀与原 300 步 baseline 记录精确一致，文件为 `/tmp/libero-task4-task7-existing-prefix-preflight-20261102-v1.json`，SHA-256 `d3ad809d79fe0a456e532c40099a8c80e4613b7e488549bb9c09e25a0be7cef6`。本 runner 再核验每个 archive，不用容差放宽来源前缀。旧短窗口不构成新成功率，初态不是 unseen。

source bank 特征来自已独立核验的 H2 source-support cache。另只读取旧训练保存的 task 4/7 meta-support 公共 t0 各 30 帧，逐 demo 核对公开观察摘要；没有新的初态重建，没有读取 HDF5、init_state、private simulator state 或 official state 文件。

## 固定 bank 与校准规则

四个空间为视觉拼接、fusion、逐帧 projection、公开 proprio。每个空间都报告以下三组：

| bank | bank 数据 | reference 数据 |
|---|---|---|
| historical_all | support demos 0..19 全部 t>=1 | support demos 20..29 全部 t>=1 |
| historical_prefix20 | 同 bank demos 的 action indices 1..20 | 同 reference demos 的 indices 1..20 |
| runtime_t0 | 同 bank demos 的公共 runtime t0 | 同 reference demos 的公共 runtime t0 |

这里的 `20..29` 是 source-support **示教编号**，不是 qualification 初态编号。编码器已训练过全部 support；这不是独立编码器测试集。每个 bank 仅用自己的数据拟合 float64 均值/总体标准差，std<=1e-6 用 1，不用 reference 或 candidate 拟合。固定 5NN，取五个平方欧氏距离的均值，平局按 bank 原行序稳定排序。reference 95% 分位数（默认线性插值）仅为相同 bank 的描述性校准线。

主空间/参照在结果之前固定为 fusion/historical_prefix20。报告每个 seed 在动作前 index `0,4,9,19` 的距离均值/中位数及严格超过 reference95 的比例。初始比例三个 seed 均 >=50% 才记一致初始覆盖差；index19 减 index0 的超线比例三个 seed 均 >=25 个百分点才记一致增长。初始覆盖差成立且 runtime_t0 bank 的初始比例三个 seed 均 <=25%，才记参照对比线索。全部 bank/space 矩阵保留，不根据最好看的组改主指标。

距离不是动作误差或失败率，95% 不是校准过的 OOD/资格阈值。若初始超线比例已经接近 100%，增长比例受上限影响，不能由增长条件未通过断言没有分布偏移。runtime_t0 只有 20 bank / 10 reference 帧；与历史 bank 的规模、时相和稳定化设置均不同：source t0 为 stabilization=0，candidate 为原 post-stabilization=5。因此对比不是单一 runtime 因果效应。

## 精确计算与输出预算

candidate 主前向 78 次、2,400 模型行；source 公共 t0 6 次、180 模型行；六个首批重复 6 次、192 行，总新增前向 **90 次/2,772 行**。每模型仍是其 500-update 既有权重，所有输入经原 float32 Policy 正常前向提取；72 组 bank 校准、28,800 行 candidate 邻域统计。CPU 单线程，新增 backward/optimizer/environment steps 均为 0。

CPU 输出与旧 CUDA action 的差异单列描述，不用于放宽原前缀匹配；模型权重、原输入 archive 和输出来源仍必须精确匹配。任何 hash、角色、矩阵、前缀、重复性、非有限值或预算不符立即停止。结果和六个 feature/distance NPZ 位于 `/tmp/libero-task4-task7-cached-input-coverage-20261102-v1.json` 及对应 artifacts 目录，拒绝覆盖旧输出。

不做显著性、真实物理阶段或闭环因果结论，不因距离结果训练 Policy、选择 checkpoint、更新 SPT/Knowledge 或注册 Module。qualification、validation、held-out、reserved 与 final evaluation 继续关闭。

## 执行结果：初始差异依赖参照库

下表为 fusion 距离超过各自 source reference95 的候选比例。index 0 是首个 Policy 动作前，index 19 是第 20 个动作前；每格分母为该模型的 20 条既有轨迹。各 bank 有自己的 scaler/校准线，不能把比例相减当作相同尺度的因果效应。

| task | seed 后两位 | prefix20：index 0 | prefix20：index 19 | historical_all：index 0 | runtime_t0：index 0 |
|---|---:|---:|---:|---:|---:|
| 4 | 22 | 1.00 | 0.00 | 0.00 | 1.00 |
| 4 | 23 | 1.00 | 0.20 | 0.05 | 1.00 |
| 4 | 24 | 0.85 | 0.00 | 0.00 | 0.95 |
| 7 | 22 | 1.00 | 0.45 | 1.00 | 1.00 |
| 7 | 23 | 1.00 | 0.35 | 1.00 | 1.00 |
| 7 | 24 | 1.00 | 0.35 | 1.00 | 1.00 |

两个任务均通过预设的 **prefix20 初始覆盖差** 条件，均未通过超线比例随时间一致增长的条件，也未通过 runtime_t0 消除初始差异的对照条件。原输入在第一个动作之前已与“source 前 20 步”的参照不同，不能全部归因于 Policy 运行后才产生的漂移。

重要反证是 task 4 的 historical_all 初始比例只有 `0/0.05/0`，三个 seed 的 index19 也均为 0。因此不能把 task 4 描述成相对全部 source-support 的全局 OOD，更不能把覆盖距离直接当首动作错误或失败率。task 7 的 historical_all 初始比例均为 1，但 index19 降至 `0.05/0/0.10`。当前结果更具体地支持**时相和参照库相关的初始覆盖差异**，不足以定位环境版本、稳定化、视觉或数据几何中的单一原因。初始比例接近上限时，未通过增长条件也不排除其他形式的 drift。

CPU 与原 CUDA action 的最大绝对差在 task 4 约为 `1.5–1.95e-5`、task 7 约为 `1.12–1.24e-4`，仅作设备数值记录；120 个源前缀的精确匹配规则没有放宽。

## 实际成本、证据与复核

完成 84 次主前向/2,580 行、6 次重复/192 行，72 组校准、28,800 行候选近邻统计，elapsed `14.1171s`。新增学习与交互为 0。

结果 SHA-256：`0009843ce72f6fa802da4c9538e7ede206074b3201f5cd32d614861c8d056488`。六份 task×seed 特征/预测/距离 NPZ、120 个候选文件及 source t0 的逐 demo hash、六个权重 hash 和运行时间均在主结果内。

独立复核 `/tmp/libero-task4-task7-cached-input-coverage-independent-audit-20261102-v1.json`，SHA-256 `373ef8b6576ea611595a490692cbd3447789d3edb12a7c3513818cdc1c1aabe1`。核验 17 个输入 pin、120/120 源前缀、六个 checkpoint 文件与权重、72 组 bank/scaler 和全部保存距离/汇总，另完整重搜 1,440 个首步近邻。距离运算重算最大差 `1.12e-8`，全部判断一致，elapsed `2.8049s`，无新增模型或环境调用。

为检查“距离大是否意味着动作差”，随后执行了[有专家标签的 source 首步控制](libero-task4-task7-source-t0-target-error-diagnostic-2026-11-02.md)。下一项设计应匹配 source 的稳定化设置与 bank 规模，避免再用混杂的 runtime_t0 对照指定根因；详见[晨间汇总](libero-overnight-diagnostic-progress-2026-11-02.md)。
