# LIBERO本轮周内收尾计划（2026-11-05）

用户要求继续实验并在本周完成LIBERO。当前已经有可复用的同族定义、数据/接口审计、task0/1的三seed A/B/C离线适应和360条闭环，以及task4/7的BC和机制诊断。task0的B/C均12/60，task1的B为11/60且集中在一个seed，尚未得到稳定的元学习优势。

本周收尾目标是完成**这一轮source内部LIBERO可行性验证**：补齐task4/7的同方法成绩，给出支持/反证、几何覆盖差异和可复现交付。研究结论可以是否定或不确定；不以成功率必须提高作为完成条件。不把该目标改称整个LIBERO正式迁移、Module qualification或KECRL主实验已完成。

| 工作 | 计划顺序 | 完成标准 |
|---|---|---|
| task4/7 A/B/C补齐 | 11月5日起优先执行 | 三seed、每Policy500、18个端点、五点support/query曲线及来源预算全部核对 |
| 同端点配对闭环 | 训练独立审计通过后立即执行 | 360条、5步稳定化、candidate0..19、40个reset reference和320次精确比较；独立审计通过 |
| 四任务证据整理 | 闭环审计后 | A/B/C逐seed、B-A/B-C四格、离线/闭环排序差异；task0/1几何覆盖与task4/7几何外推分层，不混总分 |
| 收束与汇报 | 本周后续工作日 | 设置/预算/实际成本/哈希/失败和边界完整；交付老师口头汇报摘要和下一阶段明确决策 |

执行依据是[task4/7冻结协议](libero-source-internal-geometry-extension-protocol-2026-11-05.md)。冻结后不增加updates、挑checkpoint/seed、改loss或随结果追加训练。原task1小历史头机制分支保持停止；原始观测敏感性只作为待议问题，本轮不继续扩张零散诊断。

validation2/6、held-out5/8/9、qualification20..39、reserved40..49与final evaluation继续封存。若下一阶段要开展正式迁移，必须先独立冻结任务/数据角色、算法、总预算、checkpoint规则、统计和qualification阈值，再使用对应封存数据；本周source结果不替代这些手续。Knowledge Evolution仍需独立的环境机制证据。

执行完成更新：上述补齐、360条闭环、两份独立审计及四任务总表均已完成，见[收尾结果](libero-four-task-abc-completion-2026-11-05.md)。source内部验证达到本轮交付标准；正式迁移/qualification仍未开展，下一步只先冻结正式方案和统计门槛，不将本文件的周内目标改写为整个LIBERO主实验已完成。
