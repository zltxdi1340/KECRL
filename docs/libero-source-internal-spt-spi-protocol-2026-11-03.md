# LIBERO source 内部 SPT/SPI 可行性对照（2026-11-03）

本轮将控制诊断收束到一次动作接口核对，随后进入**非正式、仅离线的 SPT/SPI 可行性对照**。当前低闭环成功率作为已知限制保留；本轮不尝试解释所有失败，不访问 validation、held-out 或 qualification，不形成 Module。

## 问题与三臂

问题是：在共享技能族内，多个 source SPI 的 support/query 反馈能否改善一个未参与本次 source 更新的 SPI 的固定预算学习？任务都属于已暴露的 source 集合，因此结果不能称未见任务泛化或正式迁移。

- A：原始 SPT 生成 Policy 初始化，不用 source 反馈更新。
- B：上下文条件化 FOMAML，source SPI 上一步 SGD support 适应后，以 query 梯度通过原始生成器 Jacobian 更新隔离候选。实现 `J_G^T stopgrad(g_query)`，不反传内循环 Hessian。
- C：普通多任务 BC 预训练，使用与 B 相同的两类 source 数据、上下文生成参数和模型容量；对未适应 Policy 的 support/query 损失求平均，不使用适应后元目标。

继续复用已有 PickPlace SPT/SPI 定义与 CNN-only 双相机 Policy。原始配置文件保持；本轮 effective Policy 和 SPI 明确记为 CNN-only、均匀七维 MSE。训练在隔离候选上发生，不接受或切换活动 SPT；普通 BC 的 C 不称 Skill Evolution。

## 任务、数据角色与防泄漏

source 任务角色仍为 `0,1,3,4,7`。仅 task0/1 的 table-region 几何有另一个 source 任务可替代，按此结构规则作两个内部探针：

| 探针 | source 更新任务 |
|---|---|
| 0 | 1,3,4,7 |
| 1 | 0,3,4,7 |

两个训练侧都覆盖原四类几何。角色不重命名为 held-out；探针及其 query 在历史诊断中已暴露，两个 fold 使用独立参数和固定协议，不跨 fold 选择设置。

每个任务原 `meta_support` demos0..29 和 `meta_query` demos30..49 不变。source 更新的 support 供 B 内循环，source query 是 B 外循环训练反馈；C 获得相同批次。**这些 source query 不再承担该 fold 的独立评价。** 探针只用自身 support 适应，query 不参与任何参数更新；每个 fold 的全部 9 个适应 fit 完成后才创建其 query dataset 和读取评价 payload。只用 `t>=1`：`obs[t-1] -> action[t]`，不渲染首帧、不读 simulator 初态、不补造 t0。

## 固定预算与停止

- 三 seed：20261022/23/24；配对初始化。每个 fold/seed 的 B/C 各做50次生成器 Adam 更新；每次包含四个 source SPI，每 SPI 一批 support 与一批 query，batch8、sequence10、stride10。B 内循环一步 SGD、lr0.001；生成器 Adam lr0.0003。
- B/C 均各有400次数据梯度、200次生成器 VJP、50次生成器更新，使用相同窗口。B 另有200次一次性 Policy 的 SGD 更新，C 无此操作。匹配数据梯度、生成器更新与容量，**不声称参数写入次数、耗时或 FLOP 完全相同**。
- 每个探针三臂均适应500次 Adam，lr0.0003、clip1、betas0.9/0.999、eps1e-8、weight_decay0。相同窗口顺序，均匀七维 loss；不加 jitter、phase loss 或 gripper 权重。
- 总计18个适应 Policy、9000次适应更新；600次 source 生成器更新、1200次临时内循环更新、4800次 source 数据梯度和2400次 VJP 单列。保存0/50/150/300/500的90个日志 checkpoint，固定500端点，不选择 checkpoint。
- 只在 hash/角色/预算不符、数值非有限或资源不足时中止并保留失败，不按效果续训或改超参。物理GPU1必须空闲，GPU0的其他作业不受影响；host cache 上限8GiB。

代码测试完成后绑定科学配置及完整运行代码 hash，再启动上述固定预算。独立数值测试须区分 FOMAML 与二阶 MAML，并核对生成器的权重/偏置梯度映射、普通预训练梯度、活动参考不变性和 t>=1 对齐。合成测试不是研究结果。

## 指标和解释

全部 seed/fold 报告 support/query 学习曲线、七维和平移/旋转/夹爪 MSE、夹爪符号准确率及配对 B-A、B-C 差。均值和中位数仅作描述，不做事后显著性判断。夹爪连续 MSE 与 runtime 符号命令分别解释；离线收益不能代替闭环成功。

本轮环境交互为0，闭环默认稳定化5步保留。独立闭环比较、正式迁移、候选接受和 Module qualification 均需另有冻结协议；不因本轮结果好看自动开启。Knowledge Evolution 仍需环境机制证据。

配置：[source-internal diagnostic v1](../configs/libero-source-internal-spt-spi-diagnostic_v1.json)。执行入口：[runner](../experiments/run_libero_source_internal_spt_spi_diagnostic.py)。实际运行和审计另记 dated result report。
