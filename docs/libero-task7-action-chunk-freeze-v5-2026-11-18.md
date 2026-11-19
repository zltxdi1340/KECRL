# task7 v5 科学冻结（2026-11-18）

已冻结[H4/k1 科学方案](libero-task7-action-chunk-plan-v5-2026-11-18.md)，状态 `scientific_frozen_execution_not_bound`。冻结前113项定向合成/父模块测试通过；环境检查通过。研究载荷读取及SHA、训练更新与物理步均为0。

- [科学配置](../configs/libero-task7-action-chunk_v5.json)：SHA256 `fd945fce0843625fc279b4cb4e77b217a676d505cb123a09ea02f48fa063b56e`。
- [科学绑定](../configs/libero-task7-action-chunk-freeze-binding_v5.json)：SHA256 `1c4b17799c395616c29246e1ef84acc48906196aa7d0639e1aed44aa3675fa65`，994项代码/元数据依赖。
- 证据：`logs/libero-task7-action-chunk-freeze-20261118-v5/` 的 initial、tests-initial、tests-final、environment、freeze-integrity。早期34项及最后113项均为人造数据/空白网络，非研究结果。

两臂登记1,504,541个非骨干参数；控制未来三头冻结跳过，实际训练1,499,144；候选实际训练1,504,541。t=7/8/9边界、示教末端、双层mask、首动作执行、冻结头隔离及旧v3控制首动作/loss/梯度/两次Adam逐位一致已验证。联合监督分布变化明确记录，不能据此声称已证明动作时序是物理瓶颈。

此绑定本身拒绝研究执行。后续执行层及独立审计须经完整人造流程、新代码绑定后，才能分别开放训练与44条交接。本轮不运行120条完整评估；数据封存与Policy500/稳定化5/现七维MSE公式保持。
