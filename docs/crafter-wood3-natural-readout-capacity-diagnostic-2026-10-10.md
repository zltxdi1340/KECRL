# Crafter wood>=3 自然状态读出容量诊断（2026-10-10）

## 目的

本轮继续诊断冻结空间 CNN 后的 actor 读出能力。前一轮已经在自然 RGB 状态上完成原始线性 head 和标准化线性 head 的诊断；本轮加入一个固定宽度 64 的两层 MLP head，用来检查线性读出是否成为局部动作选择的瓶颈。

这仍是 SPI 可行性诊断，不是正式 Policy 训练，也不创建 Module、更新 Knowledge 或注册 SPT。

## 固定边界

- CNN checkpoint、critic、PPO optimizer 和 RGB 输入保持不变。
- 训练集和验证集复用上一轮已验证的 376/153 个自然状态；旧 heldout 166 个状态只用于历史审计，不能进入本轮拟合或选择。
- 新增 fresh 环境 seed 为 `93000000..93000023`，动作 seed 为 `94000000..94000023`；新 heldout 状态与旧自然状态做 exact RGB 排除。
- 监督目标是几何 teacher 给出的合法局部动作集合，损失为按类别平衡的 valid-action-set negative log likelihood。
- 标准化均值和方差只由训练集计算；MLP 的 epoch 和 learning rate 只由验证集宏准确率、再由集合 NLL 选择。
- 所有 head 在 fresh heldout 采集前锁定；采集和 8-step greedy/sample 窗口都不更新策略、encoder、critic 或 PPO optimizer。
- 每个 seed 使用 baseline、复用的标准化线性 head、标准化 MLP64 三个 arm；主运行包含跨进程 seed 0 重复。

## MLP 解释边界

MLP 使用固定初始化，线性 head 复用上一轮已经拟合并锁定的 artifact。因此该实验是容量诊断，不能宣称完全由架构单变量造成的因果差异。即使 MLP 在局部验证集上更好，也不能推出它能解决生存、探索、导航、挖矿或完整 wood>=3 任务；如果两者执行成功率接近，优先继续查执行链、状态覆盖和训练策略。

## 预验证 smoke

seed 0 smoke 已完成：fresh heldout 224 个状态，三种 head 共执行 2016 个 8-step 窗口；源策略和 optimizer 冻结检查通过。验证集宏准确率为：标准化线性 `0.436`、标准化 MLP64 `0.365`；fresh heldout 宏准确率为：标准化线性 `0.363`、标准化 MLP64 `0.366`。MLP 的 heldout 集合 NLL 为 `1.865`，线性为 `2.162`。局部 8-step greedy 成功没有出现同步提升，所以 smoke 不能支持“读出容量已经是主瓶颈”的判断。

smoke 使用 `--repeat-seed0`，没有生成独立进程重复目录；它只验证执行链，不是正式结果。

## 正式运行

```bash
PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=0 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/zl202621/.conda/envs/kecrl/bin/python -u -m experiments.run_crafter_wood3_natural_readout_capacity \
  --config configs/crafter_wood3_natural_readout_capacity_cuda_v1.yaml \
  --output results/crafter_wood3_natural_readout_capacity_cuda_20261010_v1
```

独立复核：

```bash
PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=0 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/zl202621/.conda/envs/kecrl/bin/python -u -m experiments.analyze_crafter_wood3_natural_readout_capacity \
  --input results/crafter_wood3_natural_readout_capacity_cuda_20261010_v1 \
  --output results/crafter_wood3_natural_readout_capacity_cuda_20261010_v1/analysis_verified
```

正式结果仍需按 seed、category、greedy/sample 和窗口依赖关系解释。无论 MLP 结果如何，本轮都不能把 wood 局部 teacher readout 当作完整 Module 资格；后续必须继续诊断 survival、导航执行以及 stone/coal/pickaxe 的任务链覆盖。
