# Crafter SPT→SPI 正式运行状态

日期：2026-10-18。状态：E1 与 E2 正式预算已启动，尚无完整 seed/fold 结果；不得提前解释方法效果。

## 冻结方案

- 协议：[formal v1 配置](../configs/crafter_gather_spt_transfer_protocol_formal_v1.json)，30 个独立 seed（101--130）。
- E1：wood→coal 直接迁移，最大交互预算 6,846,720 步。
- E2：三个 leave-one-binding-out fold，唯一主比较为每个 seed 内等权平均三折的 B−C 归一化 query-success AUC，最大预算 20,540,160 步。
- E1+E2 合计上限 27,386,880 步。功效规划点为 AUC 差 0.10、seed 级标准差假设 0.20；这是显式规划假设，不是 pilot 方差估计。
- 配置、协议、关键源码及运行依赖由独立 manifest 和 detached SHA-256 封存。E1 manifest SHA-256：`8fb6b93f2f23fa448a3fd72225b19a0e51edf8ad44af8ec0571a3e574bcdff35`；E2 manifest SHA-256：`ab5efd52d3cdeee3dd0c05224176ad0b468a04be1879972789084089164fd3dc`。Git 基线为 `29898dd83b15b9ce40540a6353a3b82de63e05a1`，运行相关文件哈希由 manifest 约束。

## 运行位置

| 实验 | 输出目录 | GPU | 进程 PID | 启动日志 |
| --- | --- | ---: | ---: | --- |
| E1 | `results/crafter_gather_spt_transfer_e1_formal_v1/` | 0 | 37374 | `/tmp/crafter_gather_spt_transfer_e1_formal_v1.log` |
| E2 | `results/crafter_gather_spt_transfer_e2_formal_v1/` | 1 | 37375 | `/tmp/crafter_gather_spt_transfer_e2_formal_v1.log` |

训练现已结束：E1 写出 90 份结果（3 臂 × 30 seed），E2 写出 270 份结果（3 fold × 3 臂 × 30 seed）；训练进程正常退出，GPU 空闲。

E1 独立 comparability audit 已结束，但 `valid=false`、`provenance_valid=false`，共 270 条错误，全部为跨角色初始 RGB 哈希重复：target support/evaluation 90 条、source/target support 60 条、source/evaluation 60 条、source 内 support/query 60 条。规范化场景记录交集为 0；例如 B 臂 seed 101 的 source support/query 有 24 个重复 RGB，source/target support 有 22 个重复 RGB。冻结审计将这类视觉状态复用视为角色隔离失败，因此不得把 E1 作为通过审计的正式效应报告。审计明细在 `results/crafter_gather_spt_transfer_e1_formal_v1/analysis_verified/comparability.json`。

E2 的逐 fold provenance/RGB 审计和 seed-cluster 主分析现已结束，但三折均未通过严格 comparability audit。每折各有 270 条错误，构成为：target support/evaluation 90 条、source/target support 60 条、source/evaluation 60 条、source 内 support/query 60 条；规范化场景记录交集仍为 0，重复来自初始 RGB 哈希。E2 analyzer 返回 `valid=false`、`formal_result=false`、`candidate_statistics.status=invalid_input`，`primary_effect=null`，因此没有可报告的正式 B−C 数值。验证明细在 `results/crafter_gather_spt_transfer_e2_formal_v1/analysis_verified/verification.json`。

因此，本轮 E1 和 E2 的训练账本均已完成，但两项都未通过冻结的角色隔离门槛；本轮不能回答 FOMAML 更新后的 SPT 是否改善新 SPI 学习效率。下一步应修复场景池生成/去重规则，加入“canonical scene + initial RGB”双重跨角色去重，在新协议版本中重新冻结后再重跑；当前原始结果只保留作失败诊断，不用于方法效果结论。

正式结果只回答受控 gather 族中 FOMAML 更新后的 SPT 是否改善新 SPI 学习效率；不代表自然 Crafter Module 资格、完整 KECRL、Knowledge Evolution 或持续任务链已验证。
