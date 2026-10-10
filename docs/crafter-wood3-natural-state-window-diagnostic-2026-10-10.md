# Crafter 自然状态配对短窗口诊断（2026-10-10）

## 本轮结论

冻结的 standardized linear actor 对训练 fixture 的场景分布依赖很强。在从自然
轨迹取出的 74 个可采状态中，greedy 模式 8 步内只采到指定树 `3/74`；完整还原
fixture 条件，并替换为一个已见训练背景后，恢复到 `74/74`。同一还原条件下，
70 个转向状态恢复到 `70/70`，但 84 个接近状态仍为 `0/84`。

这说明局部采集规则和 actor 的转向/采集能力可以在训练支持范围内工作，而自然
RGB 迁移、接近执行仍有缺口。当前证据支持训练场景覆盖不足；不能唯一归因于
sapling、生命值、夜间光照、背景或标准化中的某一个因素。完整组合恢复也不能
代替自然 reset 的 `wood>=3` 资格评估，尚未形成可注册的 SPI Module。

本轮没有 PPO 更新、监督拟合或 head 选择，没有 Knowledge/SPT 更新或 Module
注册，没有开始正式训练，也没有扩大到 stone、coal、pickaxe。

## 代码和输入

开始本轮前，上一项自然 reset 诊断已提交并推送至 GitHub：
`3cb0c9b03318a7eb5165321d4a575538090aeabb`，中文备注为
“增加Crafter自然迁移诊断与SPI前置验证”。

- 配置：[crafter_wood3_natural_state_window_cuda_v1.yaml](../configs/crafter_wood3_natural_state_window_cuda_v1.yaml)
- 状态复制与干预：[crafter_natural_state_window.py](../experiments/crafter_natural_state_window.py)
- CUDA runner：[run_crafter_wood3_natural_state_window.py](../experiments/run_crafter_wood3_natural_state_window.py)
- 独立分析：[analyze_crafter_wood3_natural_state_window.py](../experiments/analyze_crafter_wood3_natural_state_window.py)
- 输入：上一项 `crafter_wood3_actor_natural_migration_cuda_20261010_v5` 的 baseline 轨迹；
  CNN-only 100k checkpoint 与已保存的 standardized linear head，source seeds `0/1/2`。

结果目录为 `results/crafter_wood3_natural_state_window_cuda_20261010_v1`。
最终分析目录为其中的 `analysis_verified/`。初次分析在完成数值核验后，因环境未
安装 matplotlib 而未生成图；安装绘图依赖后完整重跑分析，未重跑或修改 CUDA 评估。

## 配对协议

每个 source seed 使用先前已经查看过的 30 条 baseline 自然轨迹，共 90 条。
按固定动作重放，每条轨迹最多选择三个状态：首次出现的 ready、turn、approach。
选择只依赖几何与可采规则，没有使用 actor 成败、logits 或新 head 选择。

- **ready**：清醒且已朝向无遮挡邻接树，诊断目标首步动作为 `do`。
- **turn**：有无遮挡邻接树，但未处于 ready；固定顺序选择指定树，目标为先转向。
- **approach**：没有可用邻接树，选择曼哈顿距离为 2 的无遮挡可见树，且存在可通行
  的首步能把距离降到 1。目标动作可以有多个。这是局部几何标签，不是全局最优动作。

实际选出 228 个状态：ready 74、turn 70、approach 84。各 source seed 为：

| seed | ready | turn | approach | 合计 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 23 | 22 | 26 | 71 |
| 1 | 26 | 27 | 28 | 81 |
| 2 | 25 | 21 | 30 | 76 |

每个状态复制成 9 个条件，指定树、player 位置、初始朝向和 wood 数保持一致：

| 条件 | 干预 |
| --- | --- |
| natural | 保留完整自然状态 |
| full_daylight | daylight=1，并在窗口中保持白天 |
| full_vitals | health/food/drink/energy=9，清醒，重置相应内部生存计数器 |
| reset_extra_items | 除 wood 和四项生命资源之外，其余库存恢复为原生初始值 |
| no_other_entities | 移除其余实体，并关闭窗口中的实体补充 |
| clear_local_ground | 中心 3×3 除指定树外改成 grass |
| fixture_combined | 同时应用 daylight、vitals、extra items、entities、local ground 干预；外围地形仍自然 |
| fixture_background_only | 可见 9×7 范围中，中心 3×3 外的地形换为固定训练背景；保留指定树 |
| fixture_combined_anchor | fixture_combined 加固定训练背景 |

背景 anchor 是场景文件中按 manifest 顺序保留的首个训练背景，environment seed
`41000002`，没有按评估表现选择。它是**已见训练阳性对照**，不属于未见背景泛化评估。
ready/turn 的最后一个条件共 144 帧，均逐像素匹配原训练 fixture；approach 的指定
树在距离 2 处，保留该树，不要求与旧的邻接树训练帧相同。

预检查发现：未重置 extra items 时，还原画面与旧 fixture 的地形区域相同，但
sapling 库存导致 HUD 的少量像素不同。因此在完整评估启动前加入独立
`reset_extra_items` 条件并补齐组合还原。这是顺序诊断中通过预检查修订的协议，
没有事前预注册的含义。

两个 actor 使用同一冻结 CNN encoder，输入始终为 `64×64×3` RGB，动作 allowlist
为 `0..6`。每个状态/条件/actor 运行一次 greedy、两次 sampled 窗口。action seed
在配对条件和 actor 之间一致；不同 actor 的后续轨迹允许不同。诊断 labels 不进入
policy，也不决定其动作。每个窗口运行完整 8 步，采到指定树后继续运行，只有
native done 会提前结束；分别记录指定树成功与其他树的 wood 增量。

## 执行结果

下表是 **greedy，8 步内采到指定树**，列的分母是固定状态数。

| 条件 | 原 actor ready /74 | 标准化 actor ready /74 | 标准化 actor turn /70 | 标准化 actor approach /84 |
| --- | ---: | ---: | ---: | ---: |
| natural | 49 | 3 | 2 | 1 |
| full_daylight | 46 | 0 | 2 | 0 |
| full_vitals | 47 | 4 | 2 | 1 |
| reset_extra_items | 51 | 6 | 2 | 1 |
| no_other_entities | 48 | 3 | 2 | 1 |
| clear_local_ground | 49 | 4 | 2 | 1 |
| fixture_combined | 47 | 8 | 6 | 0 |
| fixture_background_only | 33 | 5 | 6 | 1 |
| fixture_combined_anchor | 23 | 74 | 70 | 0 |

标准化 actor 自然 ready 的首步 greedy `do` 只有 `1/74`；组合条件为 `8/74`，
背景单独替换为 `3/74`，完整组合与背景为 `74/74`。自然 turn 首步选对指定方向
为 `16/70`，完整还原为 `70/70`。完整还原下 approach 首步选对接近方向为 `0/84`。

自然 ready 的标准化 actor greedy 成功按 seed 为 `0/23、3/26、0/25`；完整还原
分别为 `23/23、26/26、25/25`。原 actor 在自然 ready 上为 `23/23、6/26、20/25`，
反映较大的 source seed 差异。

sampled 模式的结果如下。每个状态有两次重复，因此分母翻倍；重复不是独立 episode。

| actor / 条件 | ready /148 | turn /140 | approach /168 |
| --- | ---: | ---: | ---: |
| 原 actor / natural | 109 | 26 | 44 |
| 标准化 actor / natural | 4 | 5 | 2 |
| 标准化 actor / fixture_combined | 17 | 14 | 0 |
| 标准化 actor / fixture_background_only | 13 | 16 | 2 |
| 标准化 actor / fixture_combined_anchor | 145 | 138 | 5 |

原 actor 在自然状态下的 sampled 接近与转向优于 greedy，但还远未达到稳定执行。
标准化 actor 的完整还原改善集中在邻接转向和采集，接近仍然失败。这与先前训练
fixture 主要覆盖邻接树、未覆盖距离 2 接近动作的范围一致。

## 特征与动作坐标

用原 head 训练 split 的 mean/std（std 下限 `1e-4`）重新计算特征诊断。
“超训练范围比例”指 128 维 embedding 中超出原训练逐维 min/max 的维度比例，
下表取 ready 状态的均值。它是分布描述，不能单独证明因果。

| 条件 | 平均 feature z RMS | 平均超训练范围比例 | 标准化 actor 平均 p(do) |
| --- | ---: | ---: | ---: |
| natural | 8.79 | 60.6% | 0.0144 |
| fixture_combined | 6.93 | 48.3% | 0.0987 |
| fixture_background_only | 3.83 | 34.3% | 0.0483 |
| fixture_combined_anchor | 1.21 | 0.0% | 0.9211 |

完整还原把 ready 输入带回已有训练支持范围，同时恢复动作读出；单独还原背景
或 fixture 的其他因素均不足以恢复。这支持多个输入因素共同影响 head 的解释，
并没有分离标准化、拟合目标和训练样本覆盖各自的贡献。

74 个自然 ready 状态中，40 个有非零 sapling、34 个为零；清零额外库存使 greedy
采集从 `3/74` 增至 `6/74`。即使 sapling 本来为零，仍有 `31/34` 个自然 ready
窗口没有采到指定树。因此 sapling 是一个真实的 HUD 覆盖缺口，但目前不能将它
作为主要失败原因。

图表为 `analysis_verified/ready_interventions.png`（另有 SVG）和
`analysis_verified/paired_ready_rgb.png`。原始配对画面未经过重新绘制或插值修改。

## 核验与成本

- 主评估：12,312 个短窗口，98,364 个原生交互步。
- 90 条自然轨迹固定动作回放：12,690 步，RGB、reward、wood、terminal 全部一致。
- 54 个关闭 adapter diagnostics 的 observer 控制：432 步，完整公开轨迹一致。
- seed 0 独立进程重复：回放 4,452 步、短窗口 30,672 步、18 个 observer 控制 144 步。
- 本轮总原生步数：146,754，包括回放、重复和 observer 控制；policy 训练步数为 0。
- 主评估重新核对 39 个输入文件哈希；重复进程为 31 个。包括 source、checkpoint、
  head、参考场景及已安装 Crafter 文件，均未改变；安装包代码未修改。
- policy、optimizer、requires_grad 和 grad 冻结边界均通过；真实 CUDA tensor 已确认。
- 独立分析重新计算 RGB/array/JSON digest、首次几何状态选择、预测覆盖、训练
  standardizer、概率与 logits、逐步采集规则、指定树标签、终止及汇总统计。
- seed 0 的窗口、图片、records、features、predictions 五个 digest 跨进程一致。
- 批量特征预测与单帧窗口推理首步概率最大差为 `3.34e-4`；greedy 首步动作
  分歧为 0。差异保留在报告中，未用批量概率替代窗口实际动作。
- 新增状态复制/协议测试 13 项、分析数据一致性测试 4 项。全量回归：
  `243 passed in 59.59s`。

## 对 SPI 的意义和下一步

当前工作验证的是 SPI 候选执行器的具体能力边界。完整还原条件下的局部成功
说明执行通路可用，但自然迁移和接近动作不足，尚不能保证自然世界中的
`wood>=3` 契约。候选 0.8 门槛仍待验证，本轮短窗口不用于判定资格。

下一项应做**冻结 encoder 的自然状态动作可读性与覆盖诊断**：

1. 分别覆盖 ready、turn、approach 的自然 RGB，保留生命资源、库存、光照、实体
   与背景的变化，不按 actor 成败挑选样本。
2. 以完整 environment seed/episode 隔离 train、validation 和新 heldout 状态，
   按三段报告动作读出和真实短窗口执行；本轮已查看的轨迹只用作开发证据。
3. 若使用几何 teacher 标签拟合读出，明确标记 teacher-assisted 诊断；不将它混入
   teacher-free PPO 正式结果，不让私有 oracle 进入执行策略。用它检验现有 CNN
   能否表达自然动作目标，再决定是补训练覆盖、修改表示，还是校准优化条件。
4. 只有新自然状态执行改善后，再进行完整自然 reset 的 `wood>=3` 资格和 Module
   契约检查，然后验证 SPI 管线与连续任务链。之后再扩展 stone、coal、pickaxe。

本轮是顺序探索性诊断。同一 episode 的三类状态、两次采样和各干预窗口彼此依赖，
汇总是状态暴露加权率，不是独立样本成功率或确认性统计推断。背景 anchor 仅一个，
短窗口干预也改变了动态（光照/RNG 消耗、生存计数器、实体补充与可通行地形）。
首帧对照描述同一 state 的输入变化，窗口结果则描述被干预环境中的执行能力。

## 复现

依赖包括现有 Crafter/PyTorch/NumPy；分析绘图另需 matplotlib。本轮安装的版本为
`matplotlib==3.10.9`，NumPy/PyTorch/Crafter 未升级。

```bash
PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/zl202621/.conda/envs/kecrl/bin/python -u -m \
experiments.run_crafter_wood3_natural_state_window \
--config configs/crafter_wood3_natural_state_window_cuda_v1.yaml \
--output results/crafter_wood3_natural_state_window_cuda_NEW

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/zl202621/.conda/envs/kecrl/bin/python -m \
experiments.analyze_crafter_wood3_natural_state_window \
--input results/crafter_wood3_natural_state_window_cuda_NEW \
--output results/crafter_wood3_natural_state_window_cuda_NEW/analysis_verified
```
