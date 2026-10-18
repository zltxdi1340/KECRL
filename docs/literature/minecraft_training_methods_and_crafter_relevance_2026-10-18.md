# Minecraft 与 Crafter 训练方法及其对 KECRL 的启示

调研日期：2026-10-18
范围：Minecraft、MineRL/MineDojo、Crafter 及其近似开放世界环境中的训练方法，重点检索 2021--2026 年论文。文中区分经同行评审论文与 arXiv 预印本；方法结论按各自任务、输入、奖励、动作空间和训练预算解释，不横向比较未经对齐的成功率。

## 核心结论

已有工作反复遇到与本项目相似的问题：像素输入下的探索和状态识别、稀疏奖励与长程信用分配、资源/工具前置依赖、存活风险、跨任务干扰和技能复用。它们没有一种通用技巧同时解决所有问题，而是改变了训练条件或问题分解：加入大规模人类视频和模仿学习、用内在奖励或课程推进探索、用世界模型在想象中训练、用语言模型与技能图规划长任务，或用分层/条件策略控制执行。

这说明我们遇到的低成功率并不反常，也不能仅凭低成功率判断“Crafter选错了”或“SPT无效”。但不少公开成果依赖海量数据、任务专用奖励、预训练模型、脚本技能或不同动作接口，不能直接当作我们当前 CNN+PPO 基线的合理预期。最接近本项目研究问题的是跨任务技能复用和持续技能内化工作；它们的表示和学习机制与本项目的 SPT 更新并不相同。

## 代表性方法

| 工作 | 方法与主要发现 | 与当前问题的关系及边界 |
|---|---|---|
| [Crafter：Benchmarking the Spectrum of Agent Capabilities](https://arxiv.org/abs/2109.06780)，Hafner，2021 | 用程序生成的生存世界、RGB 观察和成就评价，覆盖采集、工具制作、生存、探索和长程推理。作者明确把深度探索、泛化、视觉表示、长程推理与信用分配列为难点。 | 与我们的环境和“采集只是更长任务链中的一个环节”最接近。它本身是基准，不是针对死亡或低成功率的解决方案。 |
| [Learning to Generalize with Object-centric Agents in Crafter](https://arxiv.org/abs/2208.03374)，Stanić 等，2022 | 发现调 PPO 超参数可显著改善基线；库存显示为前馈策略提供了近似状态记忆；没有库存时循环策略更好。面对未见物体数量，object-centric 表示改善了 OOD 泛化。 | 支持我们分别检查超参数、可见状态/记忆和空间表示的必要性。论文训练 1M 步及以上、评价成就集合，和当前 wood qualification、gather SPI 预算不同；不能据此认定 GRU 必定有益。 |
| [Video PreTraining (VPT)](https://papers.neurips.cc/paper_files/paper/2022/hash/9c7008aff45b5d8f0973b23e1a22ada0-Abstract-Conference.html)，Baker 等，NeurIPS 2022 | 先用少量标注视频训练逆动力学模型，再给大量网络视频生成动作标签进行行为克隆，随后可继续强化学习。随机初始化 RL 在钻石工具任务上几乎得不到奖励；VPT 初始化加 RL 可在 2.5% 的 10 分钟 episode 中制作钻石镐，并以按物品里程碑的奖励推进技术树。 | 直接说明从像素、随机策略和稀疏终点奖励开始会面临探索冷启动；演示先验和中间里程碑可缓解。OpenAI 报告 IDM 标注视频规模约 70,000 小时；该方法使用大规模行为先验和专门奖励，不是等预算 PPO 基线，也不验证 SPT 更新。 |
| [MineDojo](https://arxiv.org/abs/2206.08853)，Fan 等，NeurIPS 2022 | 提供大量 Minecraft 任务及网络视频、Wiki、论坛等知识；MineCLIP 将视频/文本对齐作为任务相关奖励，减少手工密集奖励需求。 | 给出“外部语义数据可帮助任务发现和奖励”的路线。若引入这种奖励，奖励模型就成为额外先验/教师；A/B/C 必须共享同一奖励，并把结论限定为该奖励条件下的迁移。 |
| [DreamerV3：Mastering Diverse Domains through World Models](https://www.nature.com/articles/s41586-025-08744-2)，Hafner 等，Nature 2025 | 世界模型在潜在想象轨迹中训练行为。论文展示仅凭稀疏奖励、无人工演示或任务课程，在 Minecraft Diamond 任务达到钻石里程碑；使用约 100M 环境步的评价预算。 | 证明纯 RL 并非绝对做不到，但成功依赖不同的模型学习机制和很大交互预算。它不能解释我们当前 PPO 低成功率的单一原因，也不能与 8k 或 100k 步 Crafter 诊断作等预算比较。 |
| [Craftax](https://arxiv.org/abs/2402.16801)，Matthews 等，ICML 2024 | Craftax-Classic 用 JAX 重写 Crafter，报告最高约 250 倍速度；论文示例中 PPO 运行 1B 交互、单 GPU 不到一小时。扩展后的 Craftax 仍需要深度探索、长期规划与记忆，作者发现已有探索和环境设计方法未带来明显进展。 | 提醒我们将环境吞吐和任务难度分开诊断：加速可以让 25k/100k/更大预算试验便宜许多，但不会自动解决探索或技能迁移。Craftax 不是原生 Crafter 的完全同一实现。 |
| [Multi-task curriculum learning in a complex, visual, hard-exploration domain: Minecraft](https://arxiv.org/abs/2106.14876)，Kanitscheider 等，2021 | 按任务学习进展自动选择课程；在 episode 内鼓励获取新物品，并随着某物品变得可靠而动态撤掉该奖励。组合课程和新颖物品奖励提升了任务学习效率。 | 对应“状态/物品覆盖不足”和逐步扩展任务难度。使用新颖度奖励会改变探索目标；当前稀疏奖励迁移协议不能事后加入而仍称同一实验。 |
| [Auto MC-Reward](https://openaccess.thecvf.com/content/CVPR2024/html/Li_Auto_MC-Reward_Automated_Dense_Reward_Design_with_Large_Language_Models_for_Minecraft_CVPR_2024_paper.html)，Li 等，CVPR 2024 | LLM 根据任务、环境观察和失败轨迹编写密集奖励；Reward Critic 检查，Trajectory Analyzer 提出修订，再迭代训练。论文展示其用于钻石任务、避开熔岩和寻找稀疏目标。 | 针对奖励稀疏、失败无法提供梯度的问题。其任务专用奖励会把专家解释注入训练；若用来改善执行，应单独设协议，并对全部比较臂一致应用，不能把收益归因到 SPT。 |
| [Plan4MC：Skill Reinforcement Learning and Planning for Open-World Long-Horizon Tasks](https://arxiv.org/abs/2303.16563)，Yuan 等，2023 | 通过内在奖励学习寻找、操作等基础技能，用 LLM 建立技能关系图，再沿图搜索组合任务计划；报告了多项需要顺序执行十余个技能的任务。 | 与“采集、挖矿、工具制作是否可形成技能族并被组合”直接相关。其低层技能通过 RL 学习，高层依赖 LLM 先验和技能图；它不检验 FOMAML 更新 SPT 是否比普通预训练好。 |
| [Voyager](https://arxiv.org/abs/2305.16291)，Wang 等，2023 | 使用 GPT-4 自动课程、可执行代码技能库、环境反馈和错误修订；技能可组合并在新世界中检索使用。论文报告了跨新世界的技能复用。 | 说明技能库和可复用动作链是 Minecraft 长期学习的一条可行路线，也与 SPT/SPI 的目标相邻。Voyager 的技能是 LLM 生成/修订的程序，控制接口更抽象；不是从 RGB 策略参数中学习共享初始化。 |
| [DEPS](https://arxiv.org/abs/2302.01560)，Wang 等，NeurIPS 2023 | Describe/Explain/Plan/Select：根据控制器执行反馈解释失败、重排或修订子目标，并估计子目标可达性，避免一次性长计划持续走偏。 | 针对长程计划和执行失败积累。它改善的是高层规划闭环，不等同于低层 CNN 的朝向、接近和采集动作策略。 |
| [JARVIS-1](https://arxiv.org/abs/2311.05997)，Wang 等，2023 | 将多模态语言模型规划、目标条件控制器和包含知识及真实生存经验的记忆结合，覆盖短任务和长任务。论文报告钻石镐长程任务最高成功率约 12.5%。 | 表明短期动作熟练不自动转化成长程成功；记忆和计划反馈有帮助，但复杂终点仍难。其数字不能与我们的 32 步受控 SPI 直接比较。 |
| [Optimus-2](https://openaccess.thecvf.com/content/CVPR2025/html/Li_Optimus-2_Multimodal_Minecraft_Agent_with_Goal-Observation-Action_Conditioned_Policy_CVPR_2025_paper.html)，Li 等，CVPR 2025 | 以 MLLM 做高层规划，以 Goal-Observation-Action conditioned policy 做低层控制；用动作引导行为编码器汇总历史，训练数据包含 25,000 个视频、约 30M 个 goal-observation-action 样本。 | 针对观察、动作、语言目标之间的对齐和历史依赖。它侧重行为数据预训练和大规模模型，不是同一 RGB PPO 初始化下的表示消融。 |
| [Optimus-3](https://arxiv.org/abs/2506.10357)，Li 等，2025 预印本 | 将领域数据不足、异构任务干扰和视觉多样性列为主要难题，采用知识增强数据生成、任务级 MoE 路由及多模态推理增强 RL。 | 与我们跨材料绑定时可能出现的共享参数干扰和视觉场景变化有关；但“论文把它列为瓶颈”不代表本项目已证明是同一原因，且该方案依赖大规模数据和模型系统。 |
| [Dreamer 4](https://arxiv.org/abs/2509.24527)，Hafner 等，2025 预印本 | 从大量无标注视频学习可交互世界模型，在模型想象中训练策略；作者报告无需环境交互，仅凭离线数据取得 Minecraft 钻石，序列涉及超过 20,000 个鼠标/键盘动作。 | 把真实环境采样成本转成离线视频和世界模型成本。它不是低预算 PPO 的小修正，关键瓶颈移到了数据、模型准确性和动作条件化。 |
| [PEAM](https://arxiv.org/abs/2605.27762)，Guo 等，2026 预印本 | 将失败—纠正轨迹通过行为克隆与对比目标写入参数化 MoE/LoRA 记忆，并使用隔离 adapter 减少旧技能遗忘。 | 与“技能经验如何沉淀进可复用参数、失败如何用于更新”最接近近期方向之一。它是多模态大模型上的经验内化，不是 SPT 生成初始化的 FOMAML；截至调研日为 arXiv 预印本，不能当作已被独立复现的定论。 |
| [Odyssey](https://www.ijcai.org/proceedings/2025/22)，Liu 等，IJCAI 2025 | 建立包含 40 个原子技能和 183 个组合技能的开放世界技能库，结合 Minecraft Wiki 问答微调模型完成长程规划和自主探索。 | 展示“原子技能→组合技能→任务规划”的系统形态，和 SPI/Module/Pipeline 的组合执行愿景相关。技能本身为库内控制程序/能力，不能作为 SPT 经验更新效果的证据。 |

## 与本项目诊断逐项对照

| 我们关心的现象 | 文献中的对应证据与常见处理 | 对当前结果的解释 |
|---|---|---|
| 探索不足、稀疏终点奖励 | Crafter 将深度探索和长程信用分配作为基准挑战；VPT 使用演示预训练和物品里程碑；Plan4MC 使用内在奖励/寻找技能；Auto MC-Reward 与 MineCLIP 使用任务相关密集奖励；Dreamer 使用世界模型想象。 | 这些工作确认这是领域共性。它们使用的附加先验不同，不能只拿成功率对比；我们的 A/B/C 应保持奖励和外部信息一致，才能回答 SPT 是否增加迁移收益。 |
| 死亡和生存压力 | 生存是 Crafter 环境机制；Minecraft 长程工作还需处理熔岩、敌对生物、资源消耗和长 episode。Auto MC-Reward 会从死亡轨迹修订危险奖励，JARVIS-1 会把生存经验写入记忆。 | “生存困难是共性”有依据；但文献没有证明我们自然 wood 低成功率主要由死亡单独造成。当前受控 gather fixture 明确移除了敌人、水和熔岩，并提供木镐与充足初始资源，因此 E1/E2 不回答自然生存问题。 |
| RGB 表示、空间动作和阶段条件化 | Crafter OOD 工作显示 object-centric 表示对对象数量变化有益，库存可承担记忆作用；Optimus-2 用目标/观察/动作共同建模，并用历史行为编码；VPT 用大规模视觉行为数据形成动作先验。 | 文献说明表示与历史上下文可能重要，但并未替我们分清 CNN、动作 head、状态覆盖和 PPO 优化的贡献。项目里的冻结策略 cross-readout 只是诊断，不能代替自主策略成功率。 |
| 状态覆盖、跨任务差异和干扰 | 课程学习按掌握进度调任务和新物品奖励；CrafterOOD 显式测对象分布变化；Optimus-3 将异构任务干扰列为问题，并用 MoE task routing。 | E2 的 leave-one-binding-out 比较与跨绑定泛化问题相关。仍需报告每个 holdout fold，避免把三种材料的总体均值当作对所有 SPI 泛化。 |
| 技能复用和长任务组合 | Plan4MC、Voyager、JARVIS-1、Odyssey 都把复杂目标拆成技能/子目标并重用；PEAM 探索把成功与纠正经验内化到参数。 | 这支持 SPT/SPI 研究问题具有领域意义。但这些方法多用代码技能、LLM 规划器、技能库检索或大模型 adapter；目前没有替代“更新前 SPT vs 更新后 SPT vs 同预算普通预训练”这个因果对照。 |

## 对 KECRL 实验路线的建议

1. **保留当前受控 SPT 迁移问题。** 项目已经实际跑通 `SPT → Instantiate → SPI → support adaptation → independent evaluation`。E1 wood→coal 的 v3 pilot 是两 seed 校准，终点为 A `15/80`、B `7/80`、C `19/80`；它没有显示 B 优于普通预训练 C，也不足以否定 SPT。该事实应促使我们按候选协议完成冻结和正式对照，而不是用 pilot 排名作结论。详见[当前进度报告](../crafter-spt-spi-progress-report-2026-10-16.md)与[迁移协议草案](../crafter-spt-spi-transfer-protocol-2026-10-15.md)。
2. **保持“组件迁移”和“自然任务生存”两条证据链分开。** 受控场景提供木镐、移除敌人/水/熔岩，隔离 gather SPI 的目标绑定迁移；自然 Crafter 的生存、工具制作、挖矿前置依赖和连续任务链需要之后独立验证。组件成功不能直接升级为自然 Module 资格。
3. **不要把外部先验悄悄加进当前协议。** 演示、物品新颖度奖励、MineCLIP 奖励、LLM 生成奖励、有效动作 mask 或脚本技能都可能提升执行，但会改变研究处理。如果要采用，应另立协议版本，并让比较臂共享同一来源，或明确把它作为新的研究因素。
4. **大预算与环境吞吐分开处理。** DreamerV3 的结果表明 100M 量级探索可能解决更长任务；Craftax 说明 JAX 重写能显著提高迭代速度。前者不是我们当前预算的合理成功率基线，后者也不能证明原生 Crafter 的机制变简单。若将来切换基准，需保留任务、动作、观测、奖励和转移契约，并重新做协议，不应把不同环境的分数拼在一起。
5. **技能复用主张需要跨 SPI 的独立评价。** B 对 A 有利才支持来源经验经 SPT 更新后超过无来源经验；B 对 C 有利才支持该更新路径超过普通预训练。即使出现这种结果，也只支持本协议限定的 gather 族和适应预算，不自动证明 Knowledge Evolution、所有 SPT、Module 注册或完整 KECRL。

## 参考文献

以下链接优先指向论文、会议论文集或作者项目页。2025--2026 的 arXiv 工作明确标记为预印本；引用它们时应保留该状态。

1. Hafner, D. (2021). [Benchmarking the Spectrum of Agent Capabilities](https://arxiv.org/abs/2109.06780). arXiv:2109.06780.
2. Stanić, A., Tang, Y., Ha, D., & Schmidhuber, J. (2022). [Learning to Generalize with Object-centric Agents in the Open World Survival Game Crafter](https://arxiv.org/abs/2208.03374). arXiv:2208.03374.
3. Baker, B. et al. (2022). [Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos](https://papers.neurips.cc/paper_files/paper/2022/hash/9c7008aff45b5d8f0973b23e1a22ada0-Abstract-Conference.html). NeurIPS 2022. 数据规模说明：[OpenAI 项目说明](https://openai.com/index/vpt/).
4. Fan, L. et al. (2022). [MineDojo: Building Open-Ended Embodied Agents with Internet-Scale Knowledge](https://arxiv.org/abs/2206.08853). NeurIPS 2022, Datasets and Benchmarks.
5. Kanitscheider, I. et al. (2021). [Multi-task curriculum learning in a complex, visual, hard-exploration domain: Minecraft](https://arxiv.org/abs/2106.14876). arXiv:2106.14876.
6. Hafner, D. et al. (2025). [Mastering diverse control tasks through world models](https://www.nature.com/articles/s41586-025-08744-2). Nature 640, 647--653.
7. Matthews, M. et al. (2024). [Craftax: A Lightning-Fast Benchmark for Open-Ended Reinforcement Learning](https://arxiv.org/abs/2402.16801). ICML 2024.
8. Li, H. et al. (2024). [Auto MC-Reward: Automated Dense Reward Design with Large Language Models for Minecraft](https://openaccess.thecvf.com/content/CVPR2024/html/Li_Auto_MC-Reward_Automated_Dense_Reward_Design_with_Large_Language_Models_CVPR_2024_paper.html). CVPR 2024.
9. Yuan, H. et al. (2023). [Skill Reinforcement Learning and Planning for Open-World Long-Horizon Tasks](https://arxiv.org/abs/2303.16563). NeurIPS 2023 Foundation Models for Decision Making Workshop.
10. Wang, G. et al. (2023). [Voyager: An Open-Ended Embodied Agent with Large Language Models](https://arxiv.org/abs/2305.16291). arXiv:2305.16291.
11. Wang, Z. et al. (2023). [Describe, Explain, Plan and Select: Interactive Planning with LLMs Enables Open-World Multi-Task Agents](https://arxiv.org/abs/2302.01560). NeurIPS 2023.
12. Wang, Z. et al. (2023). [JARVIS-1: Open-World Multi-task Agents with Memory-Augmented Multimodal Language Models](https://arxiv.org/abs/2311.05997). arXiv:2311.05997.
13. Li, Z. et al. (2025). [Optimus-2: Multimodal Minecraft Agent with Goal-Observation-Action Conditioned Policy](https://openaccess.thecvf.com/content/CVPR2025/html/Li_Optimus-2_Multimodal_Minecraft_Agent_with_Goal-Observation-Action_Conditioned_Policy_CVPR_2025_paper.html). CVPR 2025.
14. Li, Z. et al. (2025). [Optimus-3: Towards Generalist Multimodal Minecraft Agents with Scalable Task Experts](https://arxiv.org/abs/2506.10357). arXiv:2506.10357, preprint.
15. Hafner, D., Yan, W., & Lillicrap, T. (2025). [Training Agents Inside of Scalable World Models](https://arxiv.org/abs/2509.24527). arXiv:2509.24527, preprint.
16. Guo, Y. et al. (2026). [PEAM: Parametric Embodied Agent Memory through Contrastive Internalization of Experience in Minecraft](https://arxiv.org/abs/2605.27762). arXiv:2605.27762, preprint.
17. Liu, S. et al. (2025). [Odyssey: Empowering Minecraft Agents with Open-World Skills](https://www.ijcai.org/proceedings/2025/22). IJCAI 2025.
