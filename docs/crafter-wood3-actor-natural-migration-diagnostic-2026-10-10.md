# Crafter wood>=3 actor 自然迁移诊断（2026-10-10）

本轮检查冻结的原始 CNN-only actor 与上一轮得到的 standardized linear actor，
能否从原生 Crafter 的自然 reset 开始执行完整的 `wood >= 3` 子任务。没有进行
PPO 更新、监督重新拟合、模型选择、Knowledge 更新、SPT 更新或 Module 注册。
因此这是一项可行性诊断，不是正式训练结果，也不是 Module 资格结论。

## 协议

- 使用原始 CNN-only 100k checkpoint，source seed 为 `0/1/2`。
- 每个 seed 比较 `baseline` 和 `standardized_linear`，每个 variant 30 个 episode，
  每个 episode 最多 256 个交互步，共 180 个主 episode。
- 两个 variant 使用相同的环境 seed/action seed manifest；这保证配对的初始
  随机角色，不保证两个不同策略的完整轨迹相同。
- 环境为原生 Crafter，稳定对象排序版本为
  `balance-object-insertion-order-v1`，策略输入只有 `64x64x3` RGB。
- 初始 reset snapshot 只用于记录起始 wood 阶段；逐步 collection oracle 在动作选择
  后读取，用于标记邻接树、朝向、可采机会、阻挡移动、敌害和终止原因。它们都不
  进入 observation，也不参与动作选择。
- 每个 policy 的首个 episode 做一次带/不带 collection oracle 的 observer 重放；控制
  episode 同时关闭 adapter diagnostics，只保留 RGB、动作、奖励、公开 inventory
  和终止轨迹字段；seed 0
  再做独立进程完整重复。

配置和 runner 固定了 `formal_result=false`、`evaluation_only=true`、零训练更新、
零新拟合/选择，并在运行前后核对 source、checkpoint、head 和已安装 Crafter 文件
哈希。

## 结果

| variant | 成功 `wood>=3` | wood=1 | wood=2 | wood=3 | death | external truncation | 交互步 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 30/90 = 33.3% | 68/90 | 51/90 | 30/90 | 52 | 8 | 12,690 |
| standardized linear | 0/90 = 0.0% | 0/90 | 0/90 | 0/90 | 88 | 2 | 15,026 |

按 source seed 的成功次数：baseline 为 `10/30、13/30、7/30`；standardized
linear 为 `0/30、0/30、0/30`。同一 environment/action seed 配对后，90 对中
baseline 独有成功 30 对，两个都成功 0 对，standardized 独有成功 0 对，两个都
失败 60 对。

## 失败归因

standardized linear 的 90 个 episode 全部停留在 wood=0：

- 观察到可采机会 3,214 步，但选择 `do` 为 0，平均 `p(do)` 很低；
- 只选了 143 次 `do`，其中 140 次发生在没有邻接树的状态；
- 移动动作被位置阻挡 10,700 次，说明大量动作没有形成有效朝向/接近推进；
- 最终失败阶段中，16 个 episode 从未朝向 ready tree，38 个没有观察到无阻挡
  邻接树，36 个曾有 ready 机会但没有完成采集；
- 死亡提示为 hostile/projectile 56、depletion 17、mixed/unknown 15。

baseline 在同一类自然 reset 上已经能采到 30 次 wood=3。它的 408 个 ready 步
中执行 `do` 149 次，并产生 149 次木材增量；阶段完成数为 wood0: `68/90`、
wood1: `51/68`、wood2: `30/51`。这说明当前环境的收集规则和 adapter 不是阻塞点，
但 baseline 仍有明显导航、生存和信用分配缺口。

## 审计状态

- observer 审计：6/6 通过。
- seed 0 跨进程 baseline/standardized 完整轨迹 digest：均一致。
- policy、optimizer 和输入文件前后状态：未改变。
- 木材增量规则核对：每次 `action=do` 且处于 ready 状态才产生 1 wood，未发现规则错配。
- 分析脚本从逐 episode JSONL 重算 digest、阶段漏斗和配对结果；配对输出为
  `both_success=0`、`baseline_only_success=30`、`standardized_only_success=0`、
  `neither_success=60`。
- 新增协议测试：`2 passed`。本轮 runner 修改后的仓库全量回归为 `226 passed`。

## 判断

上一轮 standardized head 在构造的局部 fixture 上达到约 77% 的新背景采集率，
但它没有迁移到自然 reset。现在可以排除几种解释：不是自然环境收集 API 本身失效，
不是 oracle observer 改变轨迹，也不是跨进程随机不稳定。更直接的现象是：fixture
监督标签学到的动作坐标在自然 RGB 状态上没有形成可执行的 `do` 行为，并伴随大量
阻挡移动。局部 teacher-assisted head 不能直接充当 wood Module executor。

这一步对 SPI 的作用是建立“候选执行器能否通过自然任务契约”的前置证据。只有当
一个 actor 从自然 reset 能稳定完成 wood 子任务，才有理由把它包装成 SPI 的
`Module/Pipeline` 候选；本轮结果尚未达到该边界，因此没有注册 Module，也没有开始
stone、coal 或 pickaxe 的正式链路。

## 下一步

下一项先做小规模、冻结 policy 的**自然 RGB 状态迁移对照**，不再扩大 PPO 预算：

1. 在相同 encoder 下比较 CNN-only actor 与 standardized actor 的自然状态 feature
   分布、动作 logits 和 `do` 行为，确认是不是输入分布/动作坐标错配。
2. 固定自然世界中的短窗口，分别构造“已朝向树”和“邻接但需转向”两类状态，
   只测转向、接近和 `do` 三段执行，不把 oracle 标签提供给 policy。
3. 再决定是否值得做单因素 PPO 优化条件对照；若继续训练，必须固定环境 wrapper、
   交互步数、奖励和 action allowlist，单独改变表示/优化条件。
4. 只有 wood>=3 的自然资格、Module 完整契约和连续任务链稳定后，才扩展到 stone、
   coal、pickaxe，并进入 Knowledge Evolution、SPT/SPI 的正式持续学习实验。

## 产物

- [配置](../configs/crafter_wood3_actor_natural_migration_cuda_v1.yaml)
- [自然迁移 runner](../experiments/run_crafter_wood3_actor_natural_migration.py)
- [结果摘要](../results/crafter_wood3_actor_natural_migration_cuda_20261010_v5/summary.json)
- [逐轨迹分析](../results/crafter_wood3_actor_natural_migration_cuda_20261010_v5/analysis/verification.json)
- [配对结果](../results/crafter_wood3_actor_natural_migration_cuda_20261010_v5/analysis/paired_outcomes.csv)

复现命令：

```bash
PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/zl202621/.conda/envs/kecrl/bin/python -u -m \
experiments.run_crafter_wood3_actor_natural_migration \
--config configs/crafter_wood3_actor_natural_migration_cuda_v1.yaml \
--output results/crafter_wood3_actor_natural_migration_cuda_NEW
```
