# LIBERO 候选 BC Policy / SPT / SPI Pipeline 检查（2026-10-21）

本记录验证候选 `PickPlace(Bowl, Plate)` SPT 能否生成一个带明确 SPI 绑定的连续控制 Policy，并在不越过数据角色边界的前提下完成真实示教读取、时序行为克隆更新和在线环境动作。它是组件集成检查，不是学习 pilot、迁移比较、SPT 更新结果或 Module qualification。

## 数据访问边界

新增 [`src/environments/libero_dataset.py`](../src/environments/libero_dataset.py)：

- 先解析 task role 和 demonstration role，再打开 HDF5；
- 开发阶段默认拒绝 transfer protocol 中所有 `before_method_freeze` 锁定项；
- held-out 只能由未来冻结 runner 显式请求 `post_method_freeze`；
- 每次构造记录 task、role、索引、文件路径、哈希验证、序列长度、stride 和窗口数；
- HDF5 按序列窗口惰性读取，不把 6 GB 演示数据整体载入内存；
- 最后一个不足 10 步的窗口只在尾部补零，并用 `valid_mask` 排除 padding loss；窗口不会跨 demonstration。

本次只访问 source task 0 的 `meta_support` demonstration 0--29。文件大小和 SHA-256 均重新验证，得到 306 个长度 10、stride 10 的窗口。validation 和 held-out 数据均未读取。

## 候选 SPT/SPI

[`src/skills/libero_pick_place_spt.py`](../src/skills/libero_pick_place_spt.py) 将已审计的任务族实例化为候选 SPT 版本 `0.1.0-candidate`。SPI 保留完整的：

- `source_predicate`；
- `source_object`；
- `source_anchor`；
- `source_geometry`；
- `source_relation`；
- 初态能力、目标状态转移、Policy 接口和适应规格。

task 4 的 `Open(drawer)` 被写为启动能力，不是动作目标。程序规格明确 `phase_labels_available=false`；示教中的夹爪事件不能伪装成阶段监督。

SPT context 为 11 维结构化 multi-hot：

| 组成 | 维数 | 类别 |
|---|---:|---|
| source predicate | 2 | `on`, `in` |
| source geometry | 4 | table、movable surface、open container、fixture |
| source relation | 5 | between、next-to、center、on-support、inside-container |

每个 context 恰有 3 个非零位。精确 `source_anchor` 保留在 SPI 中，由 RGB Policy 视觉解析，不编码 task id 或 anchor one-hot。因而 task 1/8、3/5、7/9 分别共享相同 context，但保留不同原始 anchor；这为同类关系迁移建立了可检验条件，避免 held-out 依赖从未训练过的 one-hot 列。

可学习 SPT 保存共享 Policy 模板，并用一个 `11 → 1799` 的线性生成器产生完整动作头权重/偏置残差，共 19,789 个参数。生成器零初始化，因此初始 SPI 与共享模板一致；SPI 适应发生在克隆 Policy 上，不改变活动 SPT。后续 FOMAML 才能更新隔离的 candidate SPT。

## 候选 Policy

配置保存在 [`configs/libero_spatial_bc_policy_candidate_v1.json`](../configs/libero_spatial_bc_policy_candidate_v1.json)，实现位于 [`src/algorithms/libero_bc_policy.py`](../src/algorithms/libero_bc_policy.py)：

- agentview 与 eye-in-hand 两路 `128×128` RGB 共享 CNN；
- CNN 保留 `4×4` 空间网格，再分别映射到 128 维视觉 embedding；
- 拼接两路视觉特征、7 维关节和 2 维夹爪状态，映射为 256 维；
- 256 维 GRU 保存序列状态；
- 7 维 `tanh` 动作头满足 `[-1, 1]`；
- 长度 10 的序列使用 valid-step masked MSE。

Policy 有 821,703 个参数。它是 pre-pilot 候选，不是已接受的最终架构；当前没有 CNN-only、不同窗口或预训练视觉编码器对照。

## 有界集成结果

入口为 [`scripts/check_libero_bc_pipeline.py`](../scripts/check_libero_bc_pipeline.py)。固定 seed `20261021`，在 `cuda:1` 上取首个 batch（2 个窗口、20 个有效动作步），执行 2 次 Adam 更新，`lr=3e-4`：

| 指标 | 结果 |
|---|---:|
| masked BC loss | 0.253167 → 0.232143 |
| clip 前 gradient norm | 0.630285 → 0.603877 |
| Policy 参数 L2 变化量 | 0.394073 |
| SPT generator 是否被 SPI 适应改变 | 否 |
| 两次更新耗时 | 约 0.23 s |
| 微型检查 valid-step-updates/s | 约 170 |
| 峰值 CUDA allocation | 110,571,520 bytes |
| 在线动作范围 | `[-0.1077, 0.0656]` |
| 在线 adapter steps | 1 |

在线动作使用 source task 0 的 `candidate_nonregression` 初态 0；对应初态文件哈希已校验。reward 与 terminated 只被记录，没有用于 checkpoint、调参或结论。

两步 loss 下降证明数据、模型、mask、优化器和环境动作接口连通，也证明 SPI 适应不会直接改写 SPT。它不证明 Policy 学会抓取或放置，不证明 GRU 优于 CNN，不证明 SPT 更新有效，也不能产生 Module。

## 验证与下一步

CPU 环境中的定向测试为 `15 passed, 1 skipped`，其中跳过的是缺少 h5py 的真实 HDF5 fixture；联合 `libero-kecrl` 环境为 `16 passed`。更新访问状态后重新执行完整 transfer audit，10/10 任务通过，两个状态命名空间仍为零个完全相同 raw state。下一步应按以下顺序进行：

1. 实现确定性的 source 多任务 sampler、训练步数核算、validation query 评估和 checkpoint 选择日志；
2. 单 source SPI 固定预算、闭环归因和 causal pre-action 对照已完成：专家动作回放 6/6，因果臂闭环 0/9，未能解释闭环失败。当前聚焦 GRU 短窗口训练与连续推理的状态差异，详见 [`libero-bc-causal-alignment-diagnostic-2026-10-22.md`](libero-bc-causal-alignment-diagnostic-2026-10-22.md)；
3. 再做全部 source SPI 的 BC pretraining 与两个 validation SPI 的有限 support 适应曲线，校准学习率、窗口和预算；
4. 只有 BC 学习边界稳定后，才实现 candidate SPT 的 FOMAML 外循环和等数据、等更新预算的 A/B/C；
5. held-out query 在方法、预算和 checkpoint 规则冻结前继续保持锁定。

## 复现命令

```bash
MUJOCO_GL=egl LIBERO_CONFIG_PATH=$HOME/.libero PYTHONPATH=. \
  conda run --no-capture-output -n libero-kecrl \
  python scripts/check_libero_bc_pipeline.py \
  --dataset-root /home/zl202621/third_party/LIBERO/datasets \
  --device cuda:1
```
