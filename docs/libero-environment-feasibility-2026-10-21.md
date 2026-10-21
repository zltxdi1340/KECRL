# LIBERO 环境配置与可行性检测（2026-10-21）

本记录只覆盖环境、模拟器、任务文件、初始状态、完整演示数据和 KECRL adapter 的 smoke 检查，不是训练结果，也不是 SPT/SPI 正式实验。

## 外部依赖与隔离

- 官方 LIBERO 仓库：`Lifelong-Robot-Learning/LIBERO`。
- 已固定 checkout：`8f1084e3132a39270c3a13ebe37270a43ece2a01`。
- checkout 位置：`/home/zl202621/third_party/LIBERO`（仓库外，不上传到 KECRL）。
- 唯一保留的联合接口环境：`libero-kecrl`，Python `3.10.22`，用于同时导入 KECRL 与 LIBERO 并执行 adapter/data smoke。早期 Python 3.8 `libero` 临时环境在联合环境验证通过后已删除。
- 主要版本：`robosuite 1.4.0`、`mujoco 3.2.3`、`bddl 1.0.1`、`gym 0.25.2`、`numpy 1.22.4`、`torch 2.4.1+cpu`、`torchvision 0.19.1+cpu`、`robomimic 0.2.0`、`h5py 3.11.0`。
- Python 包通过清华 PyPI 镜像安装；演示数据通过 `hf-mirror.com` 的 Python `requests` 下载。
- `LIBERO_CONFIG_PATH` 指向外部配置目录，例如 `$HOME/.libero`；配置中的 BDDL、初始状态和资产都指向外部 checkout。
- `MUJOCO_GL=egl` 用于无窗口 offscreen 渲染。

官方 `requirements.txt` 会通过 `robomimic` 拉取 PyTorch。为了避免改变现有 `kecrl` 环境，也避免在接口检查阶段提前安装大型 CUDA runtime，当前 `libero-kecrl` 采用 CPU PyTorch wheel，并使用仓库内的 [`configs/libero_runtime_constraints_v1.txt`](../configs/libero_runtime_constraints_v1.txt) 固定已通过仿真的依赖组合。这个选择只适用于本次接口 smoke，不代表后续训练使用 CPU。

重建本次环境的命令如下；LIBERO checkout 和演示数据放在仓库外：

```bash
conda create -y -n libero-kecrl python=3.10.22 \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main \
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
conda activate libero-kecrl
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /path/to/LIBERO
git -C /path/to/LIBERO checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01
python -m pip install torch==2.4.1+cpu torchvision==0.19.1+cpu \
  --index-url https://download.pytorch.org/whl/cpu \
  --trusted-host download.pytorch.org
python -m pip install -r /path/to/LIBERO/requirements.txt \
  -c /path/to/KECRL/configs/libero_runtime_constraints_v1.txt \
  --prefer-binary -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install -e /path/to/LIBERO --no-deps \
  --config-settings editable_mode=compat
```

在 `~/.libero/config.yaml` 中把 `benchmark_root`、`bddl_files`、`init_states`、`datasets` 和 `assets` 分别指向该 checkout 下的 `libero/libero`、`libero/libero/bddl_files`、`libero/libero/init_files`、`datasets` 和 `libero/libero/assets`。执行检查前设置：

```bash
export LIBERO_CONFIG_PATH="$HOME/.libero"
export MUJOCO_GL=egl
```

## 检查命令

在仓库根目录执行：

```bash
conda activate libero-kecrl
MUJOCO_GL=egl LIBERO_CONFIG_PATH=$HOME/.libero \
  python scripts/check_libero_environment.py \
  --suite libero_spatial \
  --demo /home/zl202621/third_party/LIBERO/datasets/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5
```

脚本入口是 [`scripts/check_libero_environment.py`](../scripts/check_libero_environment.py)。它逐任务创建 offscreen 环境、加载第一个固定初始状态、执行有限次零动作，并输出 RGB 形状、奖励、终止状态和 smoke 标记。`formal_result` 固定为 `false`。

## 已完成结果

### 任务与模拟器

`libero_spatial` 的 10 个任务全部通过 1 步零动作检查：

- 10/10 BDDL 文件可加载；
- 10/10 任务的固定初始状态可加载；
- 10/10 任务可创建 Panda + MuJoCo offscreen 环境；
- 10/10 任务返回 `agentview_image`，检查分辨率为 `32×32×3`、类型为 `uint8`；
- 10/10 任务完成单步交互，没有出现创建或渲染异常。

单任务较高分辨率复核中，任务 0 返回 `64×64×3` RGB；固定初始状态张量为 `(50, 92)`。

### 演示数据接口

已从 Hugging Face 镜像下载完整 `libero_spatial` 演示集。数据集固定到 revision `f13aa24a3da8c43c7225569f28c562979fa0e35a`，预期文件大小与哈希记录在 [`configs/libero_spatial_dataset_manifest_v1.json`](../configs/libero_spatial_dataset_manifest_v1.json)。审计结果为：

- 10/10 HDF5 文件大小和 SHA-256 与固定 revision 一致；
- 10 个文件与 `task_order_index=0` 的 10 个任务一一对应；
- 共 500 条 demonstration、62,250 个时间步、6,237,050,764 字节；
- 每个任务均有 50 条 demonstration；
- 动作为有限值 `(T, 7)`，范围为 `[-1, 1]`；
- `agentview_rgb` 和 `eye_in_hand_rgb` 均为 `(T, 128, 128, 3)`、`uint8`；
- 状态维度均为 92，关节、夹爪和末端位姿字段长度一致；
- 未发现额外 HDF5 文件。

第一条 demonstration 包含：

- `states`: `(98, 92)`；
- `actions`: `(98, 7)`；
- `agentview_rgb`: `(98, 128, 128, 3)`，`uint8`；
- `eye_in_hand_rgb`、关节状态、末端位姿和夹爪状态等观测字段。
- 对应 HDF5 文件 SHA-256：`ff6f26121653c77280eb40a38773a74141c11a8509f3466058cb56dd2cc60ead`。

使用可复现 smoke 脚本将第一条 demonstration 的最终状态重新载入对应 LIBERO 环境后，`check_success()` 返回 `True`，环境再次执行零动作时返回奖励 `1.0` 且 `done=True`。这验证了“演示数据状态—模拟器—成功判定”的基本闭环。

审计命令如下，输出明确标记 `formal_result: false`：

```bash
LIBERO_CONFIG_PATH=$HOME/.libero \
  python scripts/audit_libero_dataset.py \
  --manifest configs/libero_spatial_dataset_manifest_v1.json \
  --dataset-root /path/to/LIBERO/datasets \
  --verify-task-map
```

### KECRL adapter

新增 [`src/environments/libero_adapter.py`](../src/environments/libero_adapter.py)，其策略观测只包含：

- `agentview_rgb`；
- `eye_in_hand_rgb`；
- 7 维 `joint_states`；
- 2 维 `gripper_states`。

LIBERO 原始观测中的对象位置、对象相对末端位姿、`object-state` 和完整模拟器状态不会暴露给 Policy。adapter 校验 7 维连续动作及 `[-1, 1]` 边界，并将任务成功和外部时间上限分别返回为 `terminated` 与 `truncated`。

在 `libero-kecrl` 环境中，10/10 `libero_spatial` 任务通过以下有界检查：加载固定初始状态、执行 5 步零动作稳定化、读取两路非空 `64×64×3` RGB，再执行 1 步策略动作。训练演示与同一模拟器状态的在线 `agentview` 图像方向一致；同向平均绝对像素差约 `2.90`，上下翻转后约 `55.44`，因此 adapter 不额外翻转图像。

```bash
MUJOCO_GL=egl LIBERO_CONFIG_PATH=$HOME/.libero \
  python scripts/check_libero_adapter.py \
  --suite libero_spatial --image-size 64 \
  --stabilization-steps 5 --max-episode-steps 10
```

## 已知限制

- LIBERO 官方依赖仍使用旧版 Gym；启动时会提示 Gym 已停止维护，这是上游兼容性提示，不是本次 smoke 失败。
- robosuite 会提示没有 private macro 文件；本次 offscreen 任务可正常运行，暂未执行上游宏配置脚本。
- 当前 `libero-kecrl` 使用 CPU PyTorch，因此只验证模拟器、数据和 adapter 接口，没有验证 GPU 训练吞吐。主机的两张 RTX 4090 和 CUDA 驱动已由既有 `kecrl` 环境验证可用；学习 pilot 前将在同一个 `libero-kecrl` 环境中把 CPU wheel 替换为兼容的 CUDA PyTorch wheel，再复跑依赖、adapter、反向传播和吞吐 smoke，不再创建第二个 LIBERO 训练环境。
- adapter 只完成环境接口转换，没有改变 SPT、SPI、Module 或 Knowledge 契约。
- 没有开始训练、checkpoint 选择、任务族划分或正式 A/B/C 迁移比较。

## 结论与下一步

LIBERO 作为第一阶段环境在“安装—完整 spatial 数据—固定初始状态—受限策略观测—RGB 交互—成功判定—演示状态回放”层面可行。数据准备和基础 adapter 已不再是当前阻塞点；尚未验证的是学习算法、GPU 吞吐和 SPT/SPI 迁移主张。

下一步按以下顺序进行：

1. 已将 `libero_spatial` 建模为“在不同空间初态下抓取黑碗并放到盘子”的候选 Skill Family，并完成十任务 BDDL 前提审计，结果见 [`libero-spatial-skill-family-audit-2026-10-21.md`](libero-spatial-skill-family-audit-2026-10-21.md)；
2. 下一步冻结来源 SPI、validation SPI、held-out query SPI，以及每个 SPI 内不重叠的 demonstration/固定初态划分；
3. 定义 A（更新前 SPT）、B（来源 SPI 反馈更新后的 SPT）和 C（等数据等预算普通预训练）的相同初始化、适应预算和 checkpoint 选择边界；
4. 为联合环境安装 CUDA PyTorch 并做有界吞吐 smoke，然后才启动小规模学习 pilot；pilot 通过后再冻结正式训练协议。
