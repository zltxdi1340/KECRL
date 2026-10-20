# wood >= 3：预算与表示 CUDA 诊断

日期：2026-10-08。所有运行 `formal_result=false`、无教师、无 Module
注册、无 Knowledge/SPT 更新，正式训练未启动。

## 结果与判断

任务：RGB 观测下 `wood >= 3`。五个 seed `0–4`，horizon `256`，每个
checkpoint 使用 20 个不参与训练的资格 episode，候选成功门槛 `0.8`。

| 设置 | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | 均值 ± 样本标准差 | 达到 0.8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| avgpool，80 episode | 0.25 | 0.35 | 0.25 | 0.30 | 0.30 | 0.29 ± 0.0418 | 0/5 |
| 同一 avgpool 继续至 160 episode | 0.15 | 0.45 | 0.30 | 0.15 | 0.25 | 0.26 ± 0.1245 | 0/5 |
| 空间 CNN + GRU，160 episode | 0.45 | 0.35 | 0.45 | 0.25 | 0.20 | 0.34 ± 0.1140 | 0/5 |

预算 checkpoint 差值均值 `−0.03`，样本标准差 `0.1037`。本次预算翻倍
没有稳定改善，也没有解决资格门槛问题；它不能排除更大预算的作用。

CNN + GRU 比同预算 avgpool final checkpoint 的描述性均值高 `0.08`，
逐 seed 差值为 `+0.30, −0.10, +0.15, +0.10, −0.05`，差值样本标准差
`0.1605`。这只是可行性信号，不代表统计显著改善；空间编码与记忆同时
改变，无法归因于 GRU。

实际平均训练步数：avgpool 在 80/160 episode 为 `11479` / `23378.4`，
CNN + GRU 在 160 episode 为 `24036.6`。episode 数相同并不意味着交互
步数相同。每 seed 平均 wall time 为 `182.7` / `144.2` 秒；avgpool
多一次资格评估，不能据此归因网络吞吐，也不能称为纯 GPU 时间。

训练时未成功且在 256 步之前终止的 episode 比例分别为 `67.5%` 和
`64.875%`。这提示优先检查生存、探索与奖励/信用分配；尚未记录完整的
终止原因分类，不能直接断言它们是性能差的唯一原因。

## 设备与追溯

- RD-4240，单卡 `CUDA_VISIBLE_DEVICES=0`；GPU1 未用于训练。
- Python `3.10.22`，PyTorch `2.14.1+cu130`，CUDA runtime `13.0`。
- 驱动 `580.173.02`；两个完整诊断的所有 seed 均记录 `device=cuda`
  和 `cuda_tensor_verified=true`。
- 预算结果：`results/crafter_wood3_budget_continuation_cuda_20261008_v1/`。
  运行启动 HEAD 为 `d76daa3`，带未提交诊断代码。源码快照与后续
  `3cdfbbe` 的相关脚本逐字节一致，详见目录中的 `provenance.md`。
- GRU 结果：`results/crafter_wood3_spatial_gru_cuda_3cdfbbe_v1/`；启动
  commit 为 `3cdfbbe`。日志为同名前缀的 `.run.log`。
- 汇总：预算目录下 `cuda_comparison.json`、`cuda_comparison.csv`。

旧目录 `results/crafter_wood3_budget_continuation_cuda_20261008/` 实际是
CPU 结果，其原始设备字段已保留；不得把目录名当作 CUDA 证据。

固定 Python/NumPy/Torch RNG 和 `PYTHONHASHSEED=0` 仍不能保证独立构造
Crafter 环境的完整轨迹一致。预算比较使用同一 Policy 的连续训练流，
在两个 checkpoint 上冻结策略评估；同资格 seed 不等于同完整轨迹。
资格样本从不参与策略更新，但本轮反复用于可行性诊断，后续正式验证必须
使用新冻结的数据。这里的目标成功率不能替代真实 Module 的完整契约检查。

## 验证与下一项

完整回归 `140 passed`；包含 CUDA 不可用时拒绝静默降级，以及 GRU
分步推理/序列回放一致性测试。环境检查通过，formal freeze 配置审计
`checks_passed=true`、`ready_for_formal_training=false`；候选数值未冻结。

下一项是记录采集进度、终止原因与动作分布，检查生存/探索/奖励信号；
如果要区分记忆作用，应增加 CNN-only 同预算对照。继续保留候选 `0.8`
门槛，先解决基础 Policy 可学习性，再推进真实 Module qualification。

## 复现命令

使用 `3cdfbbe` 的实现。每次使用新的输出目录，不覆盖历史文件；串行执行。

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/zl202621/.conda/envs/kecrl/bin/python -u \
-m experiments.run_crafter_wood3_paired_budget_pilot \
--config configs/crafter_wood3_budget_continuation_cuda_pilot_v1.yaml \
--output results/crafter_wood3_budget_continuation_cuda_reproduction_new

CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/zl202621/.conda/envs/kecrl/bin/python -u \
-m experiments.run_crafter_wood3_spatial_gru_pilot \
--config configs/crafter_wood3_spatial_gru_cuda_pilot_v1.yaml \
--output results/crafter_wood3_spatial_gru_cuda_reproduction_new
```
