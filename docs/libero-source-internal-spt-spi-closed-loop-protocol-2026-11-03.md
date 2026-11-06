# LIBERO source内部SPT/SPI：360 episodes配对闭环冻结协议（2026-11-03）

**已冻结，尚未运行。** 固定[上一轮离线诊断](libero-source-internal-spt-spi-diagnostic-2026-11-03.md)全部18个500-update端点，比较离线初始化/适应差异能否对应真实控制能力。当前没有闭环成功率结论；该协议仍为source candidate诊断，不是正式迁移、SPT候选接受或Module qualification。

## 固定对象与数据角色

- task0/1，训练seed20261022/23/24，A原始SPT/B上下文FOMAML/C普通预训练，2×3×3=18个既有Policy。
- 每个Policy固定使用500-update最终端点；不使用0/50/150/300点，不选择有利seed或checkpoint，不重训或更新生成器。完整路径、文件SHA-256与tensor weight hash逐一写在[冻结配置](../configs/libero-source-internal-spt-spi-closed-loop-diagnostic_v1.json)的`endpoints`中。
- 上一轮source结果SHA-256为`66a4fb7fac6288a710072271e440f6cea42167f4fb782b9a774c7554af6d0169`，独立审计为`c9b8b0d5174374cdd0f9c7493bc9211e0f30ee14a7f0be4c5224527a3f7ca68b`；本轮必须匹配其中全部18个`curve[500]`，不得替换或缩小清单。
- 只使用两个source任务的官方`candidate_nonregression`初态0..19，均已有诊断使用历史，不能称未见初态。qualification20..39、reserved40..49及validation/held-out/final evaluation继续封存。
- 不读HDF5训练/示教数据，不读取source task3/4/7的评价初态。示教与官方初态是独立编号集合，不作同数字配对。

## 预算与设备

| 项目 | 冻结值 |
|---|---|
| 每个模型episodes | 20 |
| 全部新episodes | 360；不复用历史成功率 |
| 每个task×seed×arm分母 | 20 |
| 每个task×arm描述性分母 | 60；同20个场景跨三训练seed重复 |
| 稳定化 | 每次reset后5个七维零动作，共1800 |
| 每episode Policy动作上限 | 300 |
| Policy动作总上限 | 108000 |
| 受控native `environment.step`调用上限 | 109800，分开记录零动作与Policy动作 |
| native horizon | 305，等于稳定化5+Policy上限300 |
| 环境seed | 所有模型固定20261022 |
| adapter | 每模型新建一个、复用到该模型全部初态，共18个 |
| 执行顺序 | task0/1 → seed22/23/24 → A/B/C → 初态0..19递增 |
| Policy | 既有CNN-only、128px双相机、确定性动作、eval/inference mode、无hidden state |
| 推断与渲染 | 物理GPU1；CUDA_VISIBLE_DEVICES=1；进程cuda:0；EGL device1 |
| 数值设置 | torch线程1、强制确定性、TF32关闭、CUBLAS workspace `:4096:8` |
| 学习更新 | optimizer/backward/SPT/Knowledge/Module操作均0 |

`native step`是受控的Python environment.step调用，不把物理积分子步或constructor/reset内部运算伪称Policy动作。记录attempted与completed两种计数，异常时不隐去已消耗调用。执行前检查指定GPU空闲；不切换设备抢占其他作业。

固定runtime为Python3.10.22、PyTorch2.4.1+cu124、NumPy1.22.4、robosuite1.4.0、MuJoCo3.2.3、LIBERO revision `8f1084e3132a39270c3a13ebe37270a43ece2a01`。复用上一轮[v2确定性等价池化](libero-source-internal-spt-spi-execution-revision-2026-11-03.md)，不改变Policy参数或初始化。

## 初态隔离与reset配对门槛

冻结阶段只核对两个初态容器的原始文件checksum及193-byte pickle头部元数据，数值初态解码0；没有执行环境或Policy。该metadata核对见`/tmp/libero-source-internal-closed-loop-bank-header-audit-20261103-v1.json`。

未来运行复用已有`load_candidate_prefix`选择性读取器：完整容器checksum核对后，从已pin的NumPy元数据头部读取**前20×92个float64**，到边界即停止，不用torch.load/pickle反序列化初态容器。两个task共解码40行candidate，只向显式注入的adapter传20行副本；不会调用默认加载全50行初态的adapter构造路径。只记录来源和摘要，不保存初态向量值或live private simulator state。

每次完成5步稳定化后，以公开的两路RGB、joint和gripper observation计算SHA-256。每个task/state以第一个A模型为参照；同task的其余8个模型（含其他seed和两臂）必须位级相同，共40个参照、320次精确比较。检查发生在被检查episode的首个Policy动作之前。

这是未来运行必须通过的门槛，**本轮未检验真实环境是否通过**。若不同，不放宽容差、不换状态、不选seed，不把不一致场景悄悄删除；立即保留失败episode、事件和所有实际计数，中止剩余运行。

## 终止、记录与分析规则

成功仅取adapter公开task success；失败只在Policy动作达到300时有效记为time_limit。提前异常、stabilization期间成功、无穷/NaN、异常终止边界、权重改变或预算不符都属于technical failure，不混入普通失败分母。第301个Policy动作和第6个稳定化动作不能被执行。

每条轨迹保存reset摘要、全部动作、逐步公开joint/gripper状态及相对post-stabilization reset的偏移，记录task/arm/训练seed/环境seed/初态索引/端点文件与tensor hash、终止原因、动作步数和reward。没有私有对象坐标、sim state、原始初态向量或RGB像素archive。完成和中止文件均只新建，不覆盖。

主读出是task×训练seed×arm的成功数/成功率，以及B-A、B-C同一task/seed/state的四格配对：both success、left only、right only、both failure。两个相反方向都保留。另报告B是否在每个task的三个训练seed均严格多成功，作为描述性方向检查，不作为SPT接受门槛。

只对完整360条进行主汇总；中止保留partial并声明主结果无效，不补跑替换或事后改变分母。没有显著性检验、独立同分布episode或未见泛化主张，也不合并为完整KECRL验证。Policy成功率不进入Knowledge Bank。无论结果如何，不自动开放封存数据、不接受活动SPT或注册Module。

## 执行入口和检查

[执行入口](../experiments/run_libero_source_internal_spt_spi_closed_loop_diagnostic.py)默认仅preflight；[代码绑定](../configs/libero-source-internal-spt-spi-closed-loop-execution-binding_v1.json)冻结配置及runner、Policy、adapter、前缀读取器、runtime源文件和相关测试的hash。默认preflight不导入仿真器、不打开初态容器；可额外只读复核18份CPU权重张量。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
conda run --no-capture-output -n libero-kecrl python \
  experiments/run_libero_source_internal_spt_spi_closed_loop_diagnostic.py \
  --mode preflight --verify-tensors
```

未来360-episode运行命令如下，**本轮未执行**：

```bash
CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 \
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
conda run --no-capture-output -n libero-kecrl python \
  experiments/run_libero_source_internal_spt_spi_closed_loop_diagnostic.py \
  --mode run \
  --output /tmp/libero-source-internal-spt-spi-closed-loop-20261103-v1.json \
  --artifacts /tmp/libero-source-internal-spt-spi-closed-loop-artifacts-20261103-v1
```

相关专项74项通过，包括合成360-episode完整预算、reset不一致首动作前中止、恰好5步稳定化、失败attempted/completed计数、300步边界、输入角色/端点/代码hash拒收及旧文件保留。CPU回归 **781 passed in 15.11s**，排除依赖可选Crafter的`tests/test_crafter_*.py`、`tests/test_spatial_gather_spt.py`、`tests/test_smoke.py`；不声称全量pytest。它们是接口和协议验证，不是闭环研究结果。最终权重核对、工作区与pin检查见`/tmp/libero-source-internal-closed-loop-freeze-verification-20261103-v1.json`。

下一步是使用以上固定18个端点执行已冻结的360 episodes，不新增训练。完成或技术中止后另写dated result report，再决定研究方向；validation、held-out、qualification继续封存。
