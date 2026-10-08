# Crafter PPO 失败归因与训练器校准记录（2026-10-08）

本轮是非正式诊断，所有结果均为 `formal_result=false`。没有注册 Module，
没有更新 Knowledge Bank 或 SPT，也没有启动正式训练。

## 代码与边界

`CrafterEnvironmentAdapter` 增加了显式 `diagnostics=true` 路径。普通接口仍只
返回公开 inventory 和已经确认的 world setup；诊断路径额外记录 health、food、
drink、energy、achievement、奖励分项和终止原因。PPO rollout 记录首次获得
wood=1/2/3 的步数、动作计数、熵、奖励分项、代表性 RGB 视频，并可保存策略、
优化器和 Python/NumPy/Torch/CUDA RNG checkpoint。

rollout 将 256 步定义为外部截断，Crafter 原生环境长度设为 10000。死亡是原生
done，bootstrap value 清零；外部截断保留 bootstrap value。这个定义只用于本轮
校准，正式实验仍需在冻结协议中确认。

## 固定动作回放

结果目录：`results/crafter_replay_order_audit_20261008_v3/`。

3 个环境 seed、3 个独立 `PYTHONHASHSEED` 进程、每个进程连续重放 3 次同一动作流：

- native 模式在 seed 850000 的公开轨迹第 111 步、seed 850001 的第 123 步发生
  分歧；seed 850002 本次未分歧；
- 临时按世界对象插入顺序排序 `_balance_object` 后，所有 seed、进程和重复都一致；
- 安装的 Crafter 包没有修改。

这确认了集合遍历顺序是一个可复现性来源，但没有证明它是唯一来源。后续比较运行
应固定环境进程和 hash seed，或采用经过审计的稳定排序 wrapper，并在结果中记录该
选择。

## PPO 更新单位校准

配置：`configs/crafter_wood3_ppo_calibration_cuda_pilot_v1.yaml`；结果：
`results/crafter_wood3_ppo_calibration_cuda_pilot_20261008_v3/aggregate.json`。

3 个 seed、每个 arm 恰好 8192 次训练交互，avgpool Policy、奖励、动作 allowlist、
256 步 horizon、外部截断 bootstrap 和 qualification 协议相同。episode arm 每个
episode 更新一次；fixed-step arm 累计 1024 步更新一次。两种模式都做优势归一化，
区别只在归一化范围和更新 batch。

| 更新单位 | qualification 均值 | mean updates | explained variance | value loss | approx KL | clip fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| episode | 0.250 | 51.0 | 0.0574 | 0.1793 | 0.000749 | 0.0 |
| fixed 1024 steps | 0.217 | 8.0 | 0.0198 | 0.1374 | 0.000214 | 0.0 |

训练回合终止原因也没有改变主要瓶颈：episode arm 为 death=109、success=34、
external truncation=10；fixed-step arm 为 death=130、success=14、external
truncation=7。资格成功率分别为 `[0.30, 0.15, 0.30]` 和 `[0.20, 0.20, 0.25]`，
均远低于候选 0.8，不能据此选择训练器或宣称性能提升。

## 当前判断与下一步

训练器的单 episode 高方差风险已被测量，但本轮没有显示固定步数 batch 带来稳定
成功率改善。fixed-step 的 value loss 略低，episode 的 explained variance 略高，
均不足以形成方法结论。失败仍以死亡为主，因此表示对照应延后到训练器边界固定后，
并继续用 health/food/drink/energy、首次 wood 阈值和视频区分：

1. 采集前死亡；
2. 首次采集后无法完成重复采集；
3. 已完成资源采集但在终止前受到伤害；
4. 仅在外部 horizon 截断。

下一轮可在固定环境 wrapper 后扩大交互预算，先验证 wood=1/2/3 的学习曲线，再
使用同一空间 CNN 和 embedding 做 CNN-only/CNN+GRU 的按交互步数表示对照。所有
新运行继续使用独立结果目录并保持 `formal_result=false`。
