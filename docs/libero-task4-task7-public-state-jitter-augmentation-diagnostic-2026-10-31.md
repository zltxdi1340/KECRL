# LIBERO task 4/7 public state jitter augmentation 训练诊断（2026-10-31）

## 目的与范围

本次运行执行了用户明确授权的 public joint/gripper state jitter 训练诊断。它检验训练时对公开本体状态加入固定幅度扰动，是否改变同一 source SPI 的离线 `meta_support → meta_query` action MSE。它不是正式 SPI 结果，不访问 candidate、qualification、validation、held-out 或 final evaluation，不执行闭环 rollout，不更新 SPT/Knowledge，也不注册 Module。

协议沿用 causal pre-action HDF5 数据视图：task `4, 7` 的 `meta_support` demonstrations `0–29` 用于训练，独立的 `meta_query` demonstrations `30–49` 只用于离线评价。每个 task 的全部 support fit 完成后才创建 query dataset。训练为 CNN-only Full Policy、Adam、learning rate `0.0003`、batch size `8`、每模型 `500` updates、uniform 七维 action MSE；jitter 只作用于 training 输入，query 使用原始状态。

## 授权、实际运行和 provenance

- 授权配置：[libero-task4-task7-public-state-jitter-augmentation-diagnostic_v2.json](../configs/libero-task4-task7-public-state-jitter-augmentation-diagnostic_v2.json)，SHA-256 `1280ebc125e1d9afd01d37cbebdf106245fc247a56a18b56cc833a4e6f1c4d4f`。
- 授权记录：用户指令“增加预算、启动训练”；training budget 从 `0` 增至 `6` 个模型，episode budget 仍为 `0`，checkpoint selection 仍为 `false`。
- runner：[run_libero_task4_task7_public_state_jitter_augmentation_diagnostic.py](../experiments/run_libero_task4_task7_public_state_jitter_augmentation_diagnostic.py)，SHA-256 `db33ef40cc06f0b019b95da79c9143618c377b54d4a21c8cfa75c055a28f17d2`。
- validator：[validate_libero_task4_task7_public_state_jitter_augmentation_protocol.py](../experiments/validate_libero_task4_task7_public_state_jitter_augmentation_protocol.py)，SHA-256 `18964906620658b54d0b051074d890dd34939f7ab89d9b1468ae6c70c46763ae`；v2 只有显式 authorized-run 路径可执行，v1 仍保持 fail-closed。
- 结果 JSON：`/tmp/libero-task4-task7-public-state-jitter-augmentation-20261031-v1.json`，SHA-256 `257904ef89aaec64d465311de5bb1d0b2ac2fb6929c22f1977a2e48db132bc7a`。
- checkpoint 目录：`/tmp/libero-task4-task7-public-state-jitter-checkpoints-20261031-v1`，6 个最终训练 artifact；manifest SHA-256 `1f16b0636376569ae9d1a2ee669bd60f2def063cdd012fc3382466e8c2d2de79`。
- source query result：`/tmp/libero-source-meta-query-generalization-task4-task7-20261030-v1.json`，SHA-256 `091bbb49669d2f0c4196f856ec5f9618e51a08cbc08709dbc844fb747880ac27`。
- 运行 commit：`e525d19cb45a40cc6df421b6c33b96445500c2b0`；运行时工作区有未提交研究文档/脚本修改，结果 JSON 的 `tracked_worktree_clean=false` 已记录。
- 实际命令使用 `libero-kecrl`、`cuda:0`、NVIDIA RTX 4090、Python `3.10.22`、PyTorch `2.4.1+cu124`、CUDA `12.4`；开始时间 `2026-10-31T12:21:52+00:00`，结束时间 `2026-10-31T12:24:14+00:00`，runner elapsed `141.24s`。

训练前 GPU0 仍有外部进程使用约 `5.2GB`，但利用率约 `4%`、总显存 `49.1GB`；本运行新增峰值分配约 `286MB`，只使用 GPU0 的剩余容量，没有终止或迁移外部作业。GPU1 未用于本运行。

## 结果

当前运行只训练 `public_state_jitter` arm；source baseline 使用已固定的 source meta-query Full Policy 结果作逐 task/seed 参考，并检查 jitter 模型的 initial weight hash 与对应 baseline 一致。jitter 为每个 joint state 元素独立采样 `[-0.05, 0.05]`，每个 gripper state 元素独立采样 `[-0.005, 0.005]`，每个训练 seed 使用 `seed + 910041` 的 CPU generator。

| task | jitter query MSE（三 seed均值±sample SD） | source baseline query MSE（三 seed均值±sample SD） | jitter - baseline |
|---|---:|---:|---:|
| 4 | `0.037290 ± 0.000628` | `0.038517 ± 0.000965` | `-0.001227 ± 0.001296` |
| 7 | `0.042862 ± 0.000997` | `0.042630 ± 0.001165` | `+0.000231 ± 0.000659` |

六个模型合并的未加权描述性均值为 jitter `0.040076`、source baseline `0.040574`，差值 `-0.000498`。task 4 的三个 seed 差值均为负（`-0.002713/-0.000334/-0.000634`）；task 7 的三个 seed 为 `+0.000593/-0.000529/+0.000630`，方向不一致。jitter query 的 support-to-query MSE gap 均值为 task 4 `0.003147`、task 7 `0.004521`。

按动作子空间，task 4 的 query translation MSE 均值下降 `0.001607`，gripper MSE 下降 `0.003638`；task 7 translation MSE 下降 `0.000418`，但 gripper MSE 上升 `0.002773`。这些是固定 source-demo 分区上的描述性离线差异，不做显著性检验或因果解释。

## 为什么运行时间短，预算是否足够

这次运行的短耗时来自固定诊断规模，而不是 runner 跳过了训练。每个模型的 provenance 都记录了 `optimizer_updates=500`；六个模型合计执行 `3,000` 次 optimizer update。task 4 的 30 条 support demonstrations 共包含 `4,472` 个有效 action step、`461` 个长度为 10 的窗口；task 7 共 `4,218` 个有效 action step、`434` 个窗口。batch size 为 `8`，所以每个模型名义上消费 `500×8×10=40,000` 个窗口步，相当于约 `8.7–9.2` 遍 support window；mask 后实际 valid-step-updates 约 `38,343–38,580`。CNN-only Full Policy 的可训练参数量为 `821,704`，不包含闭环 rollout，RTX 4090 上六个模型耗时 `141.24s` 属于这一规模的正常结果。

因此，500 updates 对本次单变量离线机制诊断足以完成一次固定预算比较，但不能据此说 Policy 已收敛或预算足以支撑稳定闭环。当前 JSON 只保存首批、末批和均值 loss，没有 0/250/500 等中间曲线；末批 loss 已明显低于首批，但这不足以证明后段已经 plateau。仓库已有的 100→500 update 复核显示离线 MSE 从约 `0.090` 降到约 `0.037`，同时仍只有少量 candidate pilot 成功，说明 500 比 100 更合理，却不能排除更长预算的影响，也不能把闭环瓶颈归结为训练步数。

如果要判断“预算是否不够”，下一项应单独冻结 budget-sweep 协议，在相同 task、seed、support/query 角色和 jitter 变量下同时重训 baseline 与 jitter，例如比较 `500/1000/2000` updates，并记录每个预算点的 support/query 曲线和最终权重。不能把更长预算的 jitter 结果直接与当前 500-update baseline 比较，也不能用 query MSE 选择 checkpoint；在新协议冻结和明确授权前不启动这组训练。

该预算 sweep 已按单独冻结协议执行，完整曲线和结果边界见 [public-state jitter budget-sweep 诊断](libero-task4-task7-public-state-jitter-budget-sweep-diagnostic-2026-10-31.md)。

## 解释边界

结果支持：在当前两个 source SPI、当前 support/query demonstrations、当前 500-update budget 和当前 jitter 幅度下，task 4 的离线 query 拟合出现一致的小幅改善，而 task 7 没有同方向复现。它不能说明 public-state jitter 改善闭环控制，也不能说明 joint/gripper 状态漂移是唯一根因；query MSE 不是在线成功率，且 source baseline 来自已固定的前一份 query 诊断结果。

因此不据此更新 SPT 或 Knowledge，不注册 Module，不访问 qualification/held-out/final evaluation。若继续工作，下一步应先对本结果做固定口径的逐 task/seed provenance 复核，再决定是否需要新的冻结协议；不能把 task 4 的单方向离线变化扩展成 LIBERO 总体或 KECRL 正式结论。

## 验证证据

- v2 validator：`8 passed`，包括未显式授权时拒绝 v2、预算漂移拒绝和 candidate/qualification 边界检查。
- runner wrapper/diagnostic tests：`2 passed`。
- 相关回归集合：`42 passed`。
- `py_compile` 和 `git diff --check` 通过。
- 6 个 checkpoint 文件逐一重新计算 SHA-256，与结果 JSON manifest 一致；每个 checkpoint 的 weight digest、initial model digest、training loss 和 500-update provenance 均写入结果 JSON。
