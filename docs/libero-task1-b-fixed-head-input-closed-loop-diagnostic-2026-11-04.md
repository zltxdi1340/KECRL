# task1 B同一候选权重双输入闭环冻结记录（2026-11-04）

执行状态更新：用户随后授权运行，120条fresh轨迹及独立审计现已完成，详见[实际结果](libero-task1-b-fixed-head-input-closed-loop-results-2026-11-04.md)。真实历史0/60、重复当前4/60，全部新增成功在seed24；以下保留执行前冻结阶段快照，其中“尚未运行”只描述冻结时状态。

**协议、执行入口和代码哈希已独立冻结，元数据预检通过；本轮尚未运行真实闭环，没有新增成功率结论。** 研究模型前向、初态数值解码、环境构建/交互、训练和新HDF5读取均为0。合成测试中的模型/假adapter调用与研究预算分开。

## 目的和固定对照

[第5步交叉读出](libero-task1-b-step5-crossed-readout-diagnostic-2026-11-04.md)发现运动内容项在50/60个已用场景大于权重项，但这只是指定参照下的计算分解。此前[120条路由闭环](libero-task1-b-routed-history-closed-loop-diagnostic-2026-11-04.md)为不同读出权重的baseline3/60、历史候选0/60，无法直接分离内容影响。因此本项固定同一候选权重，在两种输入路由下全部fresh执行，检验输入差异是否改变实际控制及成功结果。

| 项目 | 已冻结设置 |
|---|---|
| 对象 | source task1 B，三个seed20261022/23/24 |
| 权重 | 同seed两臂同一编码器500、候选读出500及support scaler |
| 两臂 | candidate_real_history / candidate_repeat_current |
| 共同启动 | 前4动作均用同一baseline启动头，reset清空历史 |
| 第5步起 | 运动输入[t,t-1,t-2,t-3,t-4]或[t,t,t,t,t]；两臂夹爪均读当前 |
| 初态 | 仅已用candidate_nonregression0..19，按序复用adapter |
| episodes | 3seed×2输入×20初态=120，全新执行，不复用旧成功率 |
| 环境 | seed20261022，稳定化5，Policy horizon300，native305 |
| 最大成本 | 36000 Policy动作/编码器/读出、600零动作、36600受控native step |
| 内部算术 | 最多53760次Linear，attempted/completed分记 |
| 设备 | CPU inference单线程；EGL固定物理GPU1且执行前检查占用 |
| 新训练/目标读取 | optimizer、backward、HDF5及新专家目标均0 |

具体门槛、版本、记录和停止条件以[完整协议](libero-task1-b-fixed-head-input-closed-loop-protocol-2026-11-04.md)为准。配置保留完整端点及来源hash，不重拟合scaler、调整loss或选checkpoint。旧候选停止决定不变，本项不自动推进SPT、Knowledge或Module。

## 实现与验证

新增[双输入runtime helper](../experiments/libero_fixed_head_input_runtime.py)，复用旧编码器/参数加载/冻结状态哈希，仅扩展显式运动输入路由；旧已执行helper和runner不改。新[执行入口](../experiments/run_libero_task1_b_fixed_head_input_closed_loop.py)独立管理120条轨迹、同权重身份、reset/动作/观察/表示及第5步夹爪的精确配对；保存公开状态、完整动作、派生表示和实际输入索引，不记录private state或RGB像素。

默认仅作metadata检查，核对36个输入文件的字节hash、完整support端点、父结果和独立审计、历史及本轮代码绑定，不调用初态读取器或解码NPZ/PT参数。冻结检查只读历史JSON/文件字节，未读取初态容器；未来执行前才按选择性前缀合同解码candidate20行。

65项新旧专项与相关测试通过（1.24s）；随后CPU回归1212项通过（17.20s），三份新Python文件编译、CPU环境检查通过。专项用随机合成头/表示独立核对NumPy参考和旧真实历史路径逐位一致，检查同权重、当前夹爪隔离、跨episode reset、非因果索引/错路由/参数不一致拒收、配对前缀及异常保留；不是本轮研究读出或环境结果。CUDA在这些检查中主动隐藏，环境工具的torch.available=false表示CUDA不可用，Torch2.4.1+cu124本身正常。

回归使用`python -m pytest -q tests --ignore-glob='tests/*crafter*.py' --ignore=tests/test_spatial_gather_spt.py --ignore=tests/test_smoke.py`；可选Crafter依赖及真实EGL/环境交互不在本轮检查范围，不将其写成LIBERO失败或闭环已通过。最初一次经conda传stdin的验证包装未执行Python且没有生成日志，随后改用直接`python -m pytest`命令取得上述1212项结果；无研究调用或预算变更。

## 冻结provenance与执行方式

| 对象 | SHA-256 |
|---|---|
| [配置](../configs/libero-task1-b-fixed-head-input-closed-loop-diagnostic_v1.json) | b5ff7ec57bd75aa27a5c3eef37fec36105bca763a8a33614633213ef8a48f5c4 |
| [代码绑定](../configs/libero-task1-b-fixed-head-input-closed-loop-binding_v1.json) | 532440c6bbe0d5454d305a143d8568134800ba1b64ea2c3c874785acd564c160 |
| 执行入口 | 41e8b58f6b8e1bd01b1bc6eb30aed1ba101f43e15400a635d222972e81ca42ef |
| 新runtime helper | 5cd329559a4aa96a15a20ac1bc4cef22121f5ca775d8205d4026b8d0630571ec |
| [测试](../tests/test_libero_task1_b_fixed_head_input_closed_loop.py) | cd9ba0eff5e1b2c927c18bcaa334b57e10991ced6d69db5e074d1a5d52da6d5c |
| 协议 | b055d2fc3a0ba96c1f018e5ee0656ece30c4d7292dfcf4e28a9b3b500bb20b95 |

开始时保存`/tmp/libero-task1-b-fixed-head-input-closed-loop-start-20261104-v1.json`，保护1196个既有文件（含用户AGENTS.md和.vscode），仅三份当前进度文档允许本轮追加。HEAD保持e525d19cb45a40cc6df421b6c33b96445500c2b0，不提交、清理或复制checkpoint进仓库。

最终验证记录为`/tmp/libero-task1-b-fixed-head-input-closed-loop-freeze-verification-20261104-v1.json`，记录最终预检、配置/绑定/输入hash、测试命令与结果、1196份既有文件不变、新增7文件及三个进度文档、diff/文档链接/空白检查和研究输出尚不存在。它是冻结验证记录，不是研究闭环result。

未来准确执行命令如下；本轮只运行了不含--execute的预检：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 \
conda run -n libero-kecrl python -m experiments.run_libero_task1_b_fixed_head_input_closed_loop --execute
```

未来结果路径为`/tmp/libero-task1-b-fixed-head-input-closed-loop-20261104-v1.json`，轨迹目录为同前缀`-artifacts-20261104-v1`；当前两者及failed结果均不存在，不预写空研究结果。排他创建，技术失败另存并使主结果无效，不替换或补跑样本。

## 结论及下一步

当前只能确认对照已具体冻结且入口通过验证，尚不能判断删去历史是否改善控制。下一步检查GPU1资源并按冻结入口执行120条，再独立核对原参数、全部配对门槛及实际预算；汇报三seed四格和逐对动作/终止差异后再决定训练修改。

若重复当前稳定提高成功数，只支持这一固定候选和执行合同的输入路由影响；若成功率持平，也须检查动作和终止过程，不能直接排除内容效应。重复当前可能改变表示分布，baseline自身也弱；不将任一方向结果直接写成必要记忆、唯一根因、正式SPT/SPI结果或Knowledge机制。validation、held-out、qualification、reserved和final evaluation继续封存。
