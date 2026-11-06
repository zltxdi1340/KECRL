# LIBERO source内部SPT/SPI配对闭环执行记录（2026-11-03）

后续只读分析：[六组配对轨迹报告](libero-source-internal-six-group-paired-trace-diagnostic-2026-11-03.md)已完成，保留全部正反四格与跨seed对照；当前不支持统一修改close时机、夹爪权重或状态抑制，继续保持500和现有loss。该分析未新增模型/环境/训练或封存数据访问，以下原闭环结果不变。

**已完成360条闭环轨迹并通过独立审计。** task0的A/B/C成功数为8/60、12/60、12/60，task1为0/60、11/60、1/60。B的合计表现部分改善，但收益集中在特定训练seed；两个task均未达到B相对A或C的三seed严格改善方向规则。不接受候选SPT，不注册Module。

本轮实际执行[已冻结协议](libero-source-internal-spt-spi-closed-loop-protocol-2026-11-03.md)，此前“已冻结、未运行”的报告保留为执行前快照，不改写其代码绑定。所有实验均为source内部诊断，未访问validation、held-out、qualification或final evaluation。

闭环指Policy每一步从当前相机和关节/夹爪观测产生动作，环境推进后，再根据新观测继续控制，直至成功或达到300个Policy动作。该对照检验离线动作拟合与初始化收益能否转化为真实控制能力。18个Policy均已完成500-update适应，本轮仅执行推断，没有训练或元更新。

## 固定设置与范围

- source内部探针task0/1，三个训练seed20261022/23/24，A原始SPT初始化/B上下文FOMAML候选初始化/C普通预训练初始化。三臂各自经过同预算500-update适应，全部最终端点保留。
- 每模型20个既往已使用的candidate初态0..19，共18模型、360个全新episodes。环境seed20261022、每次reset后5个零动作、每episode最多300个Policy动作；不选择checkpoint、seed或初态。
- 预算上限108000个Policy动作，另1800个稳定化零动作；native horizon305。每模型新adapter复用到全部20初态，顺序固定task→seed→A/B/C→0..19。
- 实际推断/渲染固定物理GPU1，进程cuda:0、EGL device1；强制确定性、TF32关闭，复用source训练v2的等价确定性池化。GPU0其他作业不动。
- 只数值解码两个source初态bank的前20行，向adapter显式注入；不使用默认全50行加载路径，不访问qualification/reserved/validation/held-out/final evaluation。不读HDF5、不保存初态向量、private live simulator state或RGB像素。
- 每个task/state的公开post-stabilization reset观测必须在全部九模型间位级一致，共40个参照、320次精确比较。不一致保留失败并中止，不换样本或放宽容差。

配置SHA-256：`833299119f2bc34006255de2d4d85fb80ecb843994530468668e0687c6e1d83a`。执行绑定SHA-256：`19e83ab5d13393ed3f5fc00e605c41f2ea30b04b2f96608e894e974b109e2e7f`。运行入口、18个checkpoint路径/文件hash/tensor hash与runtime版本详见[协议](libero-source-internal-spt-spi-closed-loop-protocol-2026-11-03.md)。

## 产物与审计

逐条公开轨迹与事件目录：`/tmp/libero-source-internal-spt-spi-closed-loop-artifacts-20261103-v1/`。主结果目标：`/tmp/libero-source-internal-spt-spi-closed-loop-20261103-v1.json`。运行指令与工作区快照单列在`/tmp/libero-source-internal-closed-loop-run-start-20261103-v1.json`。

[独立审计](../experiments/audit_libero_source_internal_spt_spi_closed_loop_diagnostic.py)不导入执行入口或Policy，只读取已保存的公开轨迹、元数据和18份CPU端点张量；不新增环境、Policy前向或初态bank读取。它重算360条轨迹的计数、逐帧偏移、reset配对和B-A/B-C四格结果。审计代码与阈值已在完整结果产生前绑定，记录于`/tmp/libero-source-internal-closed-loop-audit-plan-20261103-v1.json`。

相关专项53项通过，新增11项审计测试；CPU回归792 passed in17.20s，排除可选Crafter依赖的三组测试，不声称全量pytest。编译检查通过。

## 闭环结果

每臂每task共60条，三个训练seed各20条；初态0..19跨seed重复，不视为60个独立场景。

| task | A原始SPT | B FOMAML | C普通预训练 |
|---|---:|---:|---:|
| 0 | 8/60 = 13.33% | 12/60 = 20.00% | 12/60 = 20.00% |
| 1 | 0/60 = 0.00% | 11/60 = 18.33% | 1/60 = 1.67% |

全部训练seed结果如下，分母每格20，不作筛选：

| task | seed | A | B | C |
|---|---:|---:|---:|---:|
| 0 | 20261022 | 0 | 11 | 4 |
| 0 | 20261023 | 6 | 0 | 5 |
| 0 | 20261024 | 2 | 1 | 3 |
| 1 | 20261022 | 0 | 0 | 0 |
| 1 | 20261023 | 0 | 0 | 1 |
| 1 | 20261024 | 0 | 11 | 0 |

task0的B-A成功数Delta为`+11/-6/-1`，B-C为`+7/-5/-2`；B的合计优势依赖seed20261022，另两seed方向相反。task1的B-A为`0/0/+11`，B-C为`0/-1/+11`；B的11次成功全部来自seed20261024。两个task的B-A和B-C三seed严格改善flag均为false。不能仅用合计或挑选55%的单seed结果声称稳定迁移收益。

完整配对四格按task汇总如下，每行共60对；seed级四格在主JSON内全部保留。

| task / 比较 | 两者成功 | 仅B成功 | 仅参照成功 | 两者失败 | B成功数净变化 |
|---|---:|---:|---:|---:|---:|
| 0：B-A | 0 | 12 | 8 | 40 | +4 |
| 0：B-C | 4 | 8 | 8 | 40 | 0 |
| 1：B-A | 0 | 11 | 0 | 49 | +11 |
| 1：B-C | 0 | 11 | 1 | 48 | +10 |

task0 B-C合计持平，内部仍有两个方向各8对相反结果，不能写成逐场景等价。全部360条中44条success、316条time_limit；没有技术中止，不把两任务总成功数作为完整KECRL总分。

## 与离线读出的关系

上一轮task1的500端点query MSE均值A/B/C为0.052681/0.052996/0.053334，闭环却为0/11/1次成功；该组均值的MSE排名没有对应闭环排名。task0的C有最低query MSE，闭环与B同为12/60。这里只描述两种不同指标在本协议下的结果，不拟合预测关系、不把MSE差换算为成功率效应。

本轮提供了多个source SPI反馈更新候选生成器后、固定预算适应Policy的实际控制读出。它超出纯离线BC拟合，但同一source任务与已使用candidate初态仍不足以证明封存SPI迁移、完整Skill Evolution或Module资格，更不能推出环境因果Knowledge。

## 实际预算和provenance

| 项目 | 实际完成 |
|---|---:|
| 新adapter / episode | 18 / 360 |
| reset attempted / completed | 360 / 360 |
| Policy forward / action | 99457 / 99457 |
| 稳定化零动作 | 1800 |
| native environment.step | 101257 |
| reset参照 / 精确比较 | 40 / 320 |
| 数值解码candidate行 | 40，两bank各0..19 |
| qualification/reserved数值行 | 0 |
| optimizer / backward / HDF5 payload | 0 / 0 / 0 |

Policy/native/零动作的attempted与completed计数全部相等，没有隐藏失败调用或补样本。99457低于108000上限，成功提前终止导致实际动作少于上限；1800个稳定化动作完整执行。候选初态只数值解码前20行，官方初态容器没有torch/pickle反序列化；qualification/reserved尾部没有数值解码。没有读取validation/held-out数据或private live simulator state。

运行UTC时间`2026-11-03T04:42:23.685797`至`2026-11-03T05:13:15.460244`，elapsed1851.77454s（约30分51.8秒）。物理GPU1推断/渲染；Python3.10.22、PyTorch2.4.1+cu124、CUDA12.4、NumPy1.22.4、robosuite1.4.0、MuJoCo3.2.3、LIBERO revision8f1084e3132a39270c3a13ebe37270a43ece2a01。实际GPU快照另见`/tmp/libero-source-internal-closed-loop-runtime-observation-20261103-v1.json`。

主结果：`/tmp/libero-source-internal-spt-spi-closed-loop-20261103-v1.json`，121173308 bytes，SHA-256 `4cba11fd4e5cb22d471ee681ae0e18a7b77c4698f904439a36e37ef9f4ba3ad0`。

独立审计：`/tmp/libero-source-internal-spt-spi-closed-loop-independent-audit-20261103-v1.json`，SHA-256 `bec300a1adeaedc324dbc6f89325ce53bb13993a5378e08c88ce26cc7a2314e2`。复核360个单集文件及其hash与合并记录一致、99817行公开轨迹、99457个动作、18份端点文件/tensor hash、全部配对四格及reset摘要；公开偏移向量精确重算，L2误差门槛abs1e-12，不作事后放宽。所有检查通过，额外Policy/环境/optimizer/初态bank调用0。

原配置、执行绑定、18份checkpoint和源诊断产物均保持；不复制checkpoint进仓库。最终工作区、链接与hash检查见`/tmp/libero-source-internal-closed-loop-run-verification-20261103-v1.json`。可选Crafter依赖排除范围保持，不将其缺失写成LIBERO失败。

## 结论与下一步

这次闭环对照已经实际完成。B在两个task的合计成功数高于A，在task1高于C，但尚无跨三个训练seed的稳定优势，且绝对成功率仍低。当前证据不足以提升候选版本或声称可靠迁移。

下一步只读现有全部六个task×seed组的配对轨迹，保留B改善、退化和持平三个方向，检查动作/close时机及公开状态漂移与seed差异。继续固定500-update端点，不新增训练或环境交互，不选择有利seed；再据证据决定后续单变量候选或正式SPT/SPI协议。活动SPT、Knowledge和Module状态均不变，validation、held-out、qualification继续封存。
