# LIBERO task 4/7 固定表示的跨示教条件诊断（2026-11-02）

承接 H1 后续分支与[夜间总计划](libero-overnight-diagnostic-progress-2026-11-02.md)。本项在执行前冻结[配置](../configs/libero-task4-task7-condition-separability-diagnostic_v1.json)，用[独立脚本](../experiments/analyze_libero_task4_task7_condition_separability.py)回答：既有视觉/fusion/逐帧 projection 表示中的相邻专家观察，其 open/close 命令是否一致？投影层邻域的区分是否比 fusion 更差？固定 Policy 是否比同表示的跨示教近邻更易发生命令错误？

## 角色、预算和隔离

固定 task `4,7` × seeds `20261022/20261023/20261024` 六个 baseline500，不加载其他 budget、jitter 或已停止的局部候选。只读取 source-support demos `0..29` 的 `t>=1` causal 输入；不读 query dataset、示教初态属性、private simulator state 或 official evaluation states。`action0` 只用于命令切换 mask。

8,630 个独立观察、25,890 个模型—观察配对，32 帧一批且不跨 demo；task 4/7 分别 154/146 批，共 900 个主前向。每模型第一批重复一次，共 6 次/192 行。新增 optimizer、backward、环境步骤均为 0。CPU、torch/OMP/OpenBLAS/MKL 单线程，原 float32 Policy，输入与权重摘要不变；预测必须逐元素匹配上轮 source-support 预测 archive。

钩子只观察正常 `forward_step`，捕获两视角拼接视觉特征 256 维、fusion 256 维、逐帧 projection 256 维及原 head logits；另以 9 维公开 proprio 作对照。钩子必须在 finally 中移除；`tanh(logits)` 精确重建实际输出。没有新增 probe optimizer 或部署近邻控制器。

## 在结果之前固定的近邻方法

每条 demo 作为一次邻域查询，其全部帧从近邻 bank 和标准化统计中排除。对其余 29 demo 拟合 float64 均值/总体标准差，标准差 `<=1e-6` 的维度用 1；固定 k=5、平方欧氏距离、64 行 query chunk。距离平局按原 demo/action 索引顺序稳定排序，投票平局按最近的并列类别解决。两任务、三 seed、四个空间、30 folds，共 720 folds / 103,560 行邻域读出。

**编码器此前训练过全部 support 示教。** 留一只针对近邻库和标准化，不是重新训练 30 个编码器，也不构成未见示教泛化或正式测试。邻域 label agreement 只是固定距离规则下的表示结构，不能直接证明真实欠观测、信息丢失或闭环根因；不同空间的距离绝对值不能作统一 OOD 阈值。

报告 all、专家 open/close、open/close 切换 ±5 步、open 窗口外的邻域同标签比例、5NN 符号误差和原 Policy 目标动作误差。主判断是 open 窗口内 fusion agreement 减 projection agreement 是否在同任务三个 seed 都 `>=0.05`；只有满足才记该邻域规则下的一致 projection 区分下降。另记 Policy sign error 减 projection-5NN error，不以单 seed 或最低 error 选择 checkpoint。

## 同前向的饱和机制检查与夜间条件分支

以原 float32 夹爪 action `a` 的 `1-a²` 描述 tanh squash 因子，特别统计错误符号的 open 窗口。原 gripper MSE 对 logit 的导数为 `2(a-y)(1-a²)`；固定 `2×BCEWithLogits(2z,(y+1)/2)` 的导数为 `2(a-y)`，在 z=0 与原 MSE 梯度匹配，错误饱和时不会额外乘 squash 因子。本项只做解析读出，不更新参数。

后续 logit-loss 候选必须在该任务三个 seed 全部满足：support 目标全为 ±1；open 窗口错误符号行数各 `>=20`；其中 squash 因子 `<=0.1` 的比例各 `>=20%`；projection-5NN 的 open 符号误差均不高于原 Policy；同时没有达到上述一致 projection agreement 下降条件。这只是允许另行冻结一次 500-update 新损失形式诊断，不能直接训练、闭环或 qualification。任何任务不满足就不跑该任务训练，不找新的阈值使其通过。

## 来源与停止

配置 SHA-256：`85769119f0b10bc5c92049a621952577d1380abf29bb5998d1894d0a97b4cd00`；脚本 SHA-256：`58229e2684ad76bd9111b0c6099542b840380d12212d40eaa1c69943826929c7`。13 个输入 pin，包括原 Policy/角色读取、source metadata、固定 500 权重来源、旧预测 archive 与总计划。任何角色/hash/预测 parity、重复性、数据预算、hook 次数或非有限值不符即停止，保留失败，不放宽门槛。

结果另存 `/tmp/libero-task4-task7-condition-separability-20261102-v1.json`；同 stem 的 `artifacts` 目录保留 source batch 摘要、两个 task 的特征/logits/目标/索引/mask/近邻 NPZ。已有输出存在时拒绝覆盖，不复制 checkpoint 或 RGB 入仓库。

## 执行结果与停止决定

六个模型均完成，旧 source-support 预测逐元素一致。以下只统计 open 切换窗口；task 4/7 的窗口行数分别为 333/349。agreement 越高表示固定近邻中同命令标签比例越高，error 越低越好；它们不是闭环成功率。

| task | seed 后两位 | fusion agreement | projection agreement | projection-5NN sign error | Policy sign error | 错误符号中强 squash 抑制比例 |
|---|---:|---:|---:|---:|---:|---:|
| 4 | 22 | 0.6961 | 0.6691 | 0.2913 | 0.2492 | 2/83 = 2.41% |
| 4 | 23 | 0.7045 | 0.6847 | 0.2943 | 0.3904 | 2/130 = 1.54% |
| 4 | 24 | 0.6583 | 0.6757 | 0.2883 | 0.3664 | 2/122 = 1.64% |
| 7 | 22 | 0.5977 | 0.5817 | 0.4126 | 0.3782 | 11/132 = 8.33% |
| 7 | 23 | 0.6097 | 0.6040 | 0.4011 | 0.3467 | 13/121 = 10.74% |
| 7 | 24 | 0.6074 | 0.6097 | 0.3926 | 0.3696 | 16/129 = 12.40% |

两个任务均未达到预设的三 seed 一致 agreement 下降条件。这限制了“逐帧 projection 在此邻域规则下统一破坏命令区分”的解释，不表示表示已充分、没有信息混叠或不存在其他优化问题。

六个模型的强 squash 抑制比例都未达到 20%；projection-5NN 也没有在任一任务全部三 seed 优于 Policy。**两个任务的后续 BCE 训练资格均失败**，因此停止该条件训练和闭环分支，不降低门槛、不改变损失系数。该结果不证明所有 BCE 设计无效，只说明本次预设的错误饱和解释不够充分。

实际主计算为 900 次正常前向/25,890 行，加 6 次重复/192 行，CPU elapsed `95.0921s`；720 folds、103,560 行近邻读出。新增 backward、optimizer 和环境步骤均为 0。

## 证据与独立复核

- 主结果 SHA-256：`66682c55850acde665ca09c84ffcffe9ba5f0a0e9ae6c743b2390394ddb2a51e`。
- task 4/7 特征与近邻 NPZ 的 SHA-256 分别为 `1d0e9cce3c8d087a2cfd57ccc3f2f430ecf8375ea2ef65ebbff486ad1791c2c5`、`7c297fb04833c7588c098a0d5fb8d3bf71ca632593ec581788878c1565df8efb`；完整路径及 300 个 source batch 摘要在主结果中。
- `/tmp/libero-task4-task7-condition-separability-independent-audit-20261102-v1.json`，SHA-256 `a698a1b6423a11d4f9b24130c3310929cd7966708b64067d67ed15bb0d397894`。独立重读同一 8,630 个 source 输入、核验 300 批摘要/720 scaler/103,560 行近邻指标及全部停止判断；每 fold 另做一次完整 bank 近邻核查，共 720 次，指标重算差为 0，elapsed `6.4882s`，无新增模型计算。

下一步已按独立配置完成[命令时序分解](libero-task4-task7-expert-command-timing-diagnostic-2026-11-02.md)和[已有闭环输入覆盖诊断](libero-task4-task7-cached-input-coverage-diagnostic-2026-11-02.md)，未恢复训练分支。统一设置、测试与晨间结论见[夜间汇总](libero-overnight-diagnostic-progress-2026-11-02.md)。
