# LIBERO CUDA 训练路径 Smoke（2026-10-21）

本记录验证 LIBERO 演示数据、CUDA 前向/反向、优化器和 KECRL adapter 能否在同一运行环境中连通。它不评价策略学习能力，不选择 checkpoint，不更新 SPT，不执行 SPI qualification，也不是正式实验。

## 环境变更

保留唯一联合环境 `libero-kecrl`，没有新建第二个 LIBERO 环境。将原 CPU wheel 原位替换为 PyTorch 官方同版本 CUDA 12.4 wheel：

```bash
conda run -n libero-kecrl python -m pip install --upgrade \
  --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.4.1 torchvision==0.19.1
```

安装后检查结果：

| 项目 | 结果 |
|---|---|
| Python | 3.10.22 |
| PyTorch | `2.4.1+cu124` |
| torchvision | `0.19.1+cu124` |
| CUDA runtime | 12.4 |
| cuDNN | 9.1.0 |
| GPU | 2 × NVIDIA GeForce RTX 4090 |
| `pip check` | 无损坏依赖 |

实际 smoke 使用 `cuda:1`。运行时两张 GPU 均已有其他进程，因此本次计时只用于确认链路可运行，不作为吞吐 benchmark。

## 数据与访问边界

入口 [`scripts/check_libero_cuda_training.py`](../scripts/check_libero_cuda_training.py) 在运行前校验候选 transfer split，并拒绝读取边界外数据。本次使用：

- task 0，角色为 `source_spi`；
- demonstration 0、1，均属于 `source_spi.meta_support`；
- 在线初态 0，属于 `source_spi.candidate_nonregression`；
- 读取两路 RGB、7 维关节状态、2 维夹爪状态和演示动作；
- 不读取隐藏对象位姿或完整模拟器状态；
- 对选中的 HDF5 文件重新检查固定大小和 SHA-256；
- adapter 只执行 1 个动作，观察到的 reward/terminated 不用于模型选择或研究结论。

脚本内的两层小型 CNN 只负责覆盖真实张量路径。它不是候选实验 Policy，其结构、学习率和三步 loss 都不能写入正式协议。

## 结果

使用 16 个样本、Adam `lr=1e-3`、3 次全 batch 更新：

| 指标 | 结果 |
|---|---:|
| BC MSE loss | 0.248377 → 0.243336 → 0.239145 |
| gradient norm | 0.258342 → 0.225421 → 0.195214 |
| 参数 L2 变化量 | 0.217321 |
| 3 次更新耗时 | 0.324 s |
| 本次微型 smoke sample-updates/s | 147.98 |
| 峰值 CUDA allocation | 39,213,056 bytes |
| 在线动作范围 | `[-0.1153, 0.1631]` |
| adapter policy steps | 1 |

所有 loss、梯度和动作均为有限值；动作满足 `[-1, 1]`，真实 LIBERO adapter 成功接收。更换 CUDA wheel 后，重新执行 10 个 `libero_spatial` 任务的 5 步稳定化和 1 步 adapter smoke，结果为 10/10 通过。LIBERO 相关定向回归为 `46 passed`。

loss 在三步内下降只说明优化器能够改变参数，不能说明 Policy 已学会 PickPlace。吞吐来自极小网络、极小 batch 和短计时，也不能用于估算候选 Policy 的训练成本。

## 复现命令

```bash
MUJOCO_GL=egl LIBERO_CONFIG_PATH=$HOME/.libero PYTHONPATH=. \
  conda run --no-capture-output -n libero-kecrl \
  python scripts/check_libero_cuda_training.py \
  --dataset-root /home/zl202621/third_party/LIBERO/datasets \
  --task-id 0 --demonstration-indices 0,1 \
  --initial-state-index 0 --samples-per-demonstration 8 \
  --optimizer-steps 3 --device cuda:1
```

## 当前结论与下一步

CUDA wheel、真实 HDF5、两路 RGB、本体状态、反向传播、优化器和在线 adapter 已连通，GPU 运行时不再阻塞 LIBERO 学习开发。当前仍没有 Policy、SPI 或 Module 的性能证据。

候选行为克隆 Policy、structured context 和 role-restricted loader 的连通检查现已完成，见 [`libero-bc-policy-pipeline-check-2026-10-21.md`](libero-bc-policy-pipeline-check-2026-10-21.md)。下一步仍需在 source 与 SPT validation 数据上做固定上限的学习 pilot，确认视觉策略确实能学习和在线执行；通过后再实现匹配预算的 A/B/C SPT 迁移比较。
