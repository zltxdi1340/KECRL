# LIBERO BC 因果时序对齐诊断（2026-10-22）

**状态：协议已提交，实验未运行。** 本文记录即将执行的单 seed、有界诊断，不是学习结果、正式实验或 Module 资格结论。

## 要回答的问题

LIBERO HDF5 将 `obs[t]` 记录为执行 `action[t]` 后的观测。现有 BC loader 用 `obs[t] -> action[t]` 训练，与在线控制所需的 pre-action state/action 配对错位。本轮只检验这个数据时序错误是否足以解释“离线 MSE 下降、闭环执行不稳定”。

因果视图从每条授权 demonstration 的 `init_state` 重建首个 pre-action observation，再用 `[initial_observation, stored_obs[:-1]] -> actions[:]`。它与历史同索引视图保持序列长度、动作标签和窗口数一致；原始 HDF5 不修改，默认 loader 行为也不变。

## 固定对照

| 项目 | 固定设置 |
|---|---|
| 任务 | LIBERO-Spatial task 0，`PickUp(Bowl) -> Place(Plate)` |
| 训练 seed | `20261021` |
| 训练数据 | source `meta_support` demonstrations 0--29 |
| 对照臂 | `stored_same_index` 与 `causal_pre_action` |
| 每臂预算 | Adam 500 updates，batch 8，`lr=3e-4`，gradient clip 1.0，序列长 10、stride 10 |
| 配对约束 | 同一 Policy 初始化哈希、同一 shuffle generator seed/批次顺序；检查两臂有效动作步数相同 |
| 离线评估 | 两个训练后 Policy 均在 support 与 query demonstrations 上，用两种 observation/action 视图计算指标 |
| 闭环评估 | 连续 GRU；support 初态 0/1/2、query 初态 30/31/32、官方 candidate 初态 0/1/2；最多 300 步 |
| 稳定步 | demonstration 初态 0 步，便于与专家轨迹一致；官方初态 5 个零动作稳定步 |
| 前置门槛 | support/query 各 3 条专家动作重放必须全部成功，否则不训练 |

validation、held-out、qualification 初态均不访问；不选 checkpoint、不更新 SPT、不创建 Module、不更新 Knowledge，也不用于完整架构比较。该协议只允许诊断单 task、单 seed 下 observation/action 时序的影响。

## 文件与 provenance

- 代码 commit：`96d37f7bd8d694cd6792786d1267924cde9ddc83`，GitHub `main` 已推送。
- 配置：[因果对齐诊断配置](../configs/libero_spatial_bc_causal_alignment_diagnostic_v1.json)，SHA-256 `95c800594719063ed2a5b02e0e206584817b527b263fecbe3f50cc13b2a1af48`。
- 入口：[run_libero_bc_causal_alignment_diagnostic.py](../experiments/run_libero_bc_causal_alignment_diagnostic.py)。
- 基础单 SPI 配置 SHA-256：`bbc4b28c4d2d606f2595642ee0f2936bb3fbcded3e5cbd9449e78b337f134e53`。
- 回归：LIBERO 定向 `14 passed`；全量 `526 passed, 2 skipped`。
- 环境：`libero-kecrl`、Python 3.10.22、PyTorch 2.4.1+cu124、2 张 RTX 4090 可见。
- 结果文件尚未生成；因此没有结果哈希或策略性能数字。

## 运行状态和下一步

检查时两张 GPU 分别有持续运行的场景服务、梯度实验和路由实验，设备利用率约为 53% 与 75%。本轮没有启动训练，也没有中止其他进程。GPU 空闲后使用：

```bash
MUJOCO_GL=egl LIBERO_CONFIG_PATH=/home/zl202621/.libero PYTHONPATH=. \
  /opt/anaconda3/bin/conda run --no-capture-output -n libero-kecrl \
  python -u experiments/run_libero_bc_causal_alignment_diagnostic.py \
  --dataset-root /home/zl202621/third_party/LIBERO/datasets \
  --device cuda:1 \
  --output /tmp/libero-bc-causal-alignment-20261022-v1.json
```

运行后应记录输出 SHA-256、实际两臂有效动作步数、support/query 离线分项指标、9 个 Policy 闭环结果、专家回放结果和耗时，并更新本报告及研究路线。若因果对齐仍不能形成稳定闭环 SPI，后续再拆解动作头、训练窗口/隐藏态和 on-policy 状态覆盖；如果稳定改善，也只进入多 seed/multi-source 学习 pilot，不直接进入正式 SPT/FOMAML 比较。
