# 当前来源500端点的专家逐帧预测缓存可用性（2026-11-07）

**在已声明的G归档和canonical逐帧缓存范围内，当前30个端点有0个可直接复用的逐帧专家预测。** 因此目前不能完成“同专家观察下，末段平移动作已拟合不准，还是自由闭环时才偏离”的数值分解。不能把旧18模型误差套给新30模型，也不能从整体MSE反推末段误差。

这是用户“继续”后的只读缓存核对，承接[原生目标与终点审计](libero-source-candidate-native-goal-pose-results-2026-11-07.md)。该审计已发现：原生目标要求XY严格<3cm；101条未完成中99份保存终点超线，另2份线内无影子plate记录，同时保留6条原生成功与影子接触不一致。它只支持优先检验末期平移与落盘，不证明某个新loss有效。继续500 updates、稳定化5步及现有loss，G停止、H未开展和所有封存角色保持。

## 核对范围、定义与得到的结果

[元数据审计入口](../experiments/audit_libero_source_expert_prediction_cache_availability.py)只读取当前端点配置、旧source-only逐帧结果/配置、G执行manifest、writer源码和G目录文件名。没有读取G training.json的数值指标、任何原始rollout内容、预测/观测NPY/NPZ数组、checkpoint、HDF5或初态；目录文件名包括历史归档类别，不等于访问其payload。没有声称整台机器所有路径都不存在缓存。

| 项目 | 已有内容 | 对当前问题的可用性 |
|---|---|---|
| 本轮30个500端点 | task0/1/3/4/7、seed20261106/07/08、A/B，文件SHA与weight hash固定 | 需要这些精确权重的逐帧预测 |
| canonical旧逐帧缓存 | task0/1、seed20261022/23/24、A/B/C，共18模型，meta_support | 18个预测文件存在，但权重hash与当前30个全部不同，0/30可复用 |
| G当前离线评价 | 每checkpoint/角色只有有效步数、七维/平移/旋转/夹爪MSE与符号准确率六个汇总字段 | 能描述整体拟合，不能恢复帧ID、末段或分量错误发生位置 |
| RGB/本体专家输入缓存 | G dataset缓存及旧逐帧runner输入为RAM张量，结束/切task后清理；旧归档只保存target/prediction/ID/mask | 声明归档没有可直接供新30模型读出的RGB/本体输入 |

匹配要求task、seed、arm、500 update、checkpoint文件SHA和weight hash全部一致，不能仅凭同一个task、A/B标签或都写500而复用。旧缓存数据定义的transfer和dataset manifest SHA与G一致；其中合法target/帧ID可作为未来输入核验参照，但不能把旧预测转换成新Policy预测，也没有因此在本轮解码旧数组。

G声明根目录`/tmp/libero-formal-G-artifacts-20261105-v2`仅检查直接文件名/类型：1600个PT、6405个JSON、1个JSONL，没有NPY/NPZ。未打开其中的权重、trajectory或training.json。`Training.evaluate_task`的源码只归档六个汇总量，`Training.dataset/clear_cache`为内存缓存；结论限于这个writer契约和已检查的声明范围。

## 数据角色与解释限制

后续优先拟检查source meta_support demos0..29，属于当前Policy训练拟合诊断。G的source meta_query已经参与生成器外循环反馈，不能称为未见泛化或独立最终评价。本轮不新增query、validation、held-out、qualification、reserved或final payload读取。

“示教最后20个合法动作帧”是可预设的时间分层，不自动对应物理放置、接触或释放阶段。平移动作误差的单位是控制器动作单位，不能直接换算成原生root位置的3cm。即使专家观察上的误差较小，也还不能证明闭环分布偏移是唯一原因或保证新loss带来成功。

## 下一读出预算草案，尚未冻结执行

没有现成匹配缓存时，新的逐帧专家预测需要新的CNN前向及合法专家数据读取。以下只根据既有G manifest的action长度元数据计算，实际新前向/训练/环境/数据数组读取均0。它不是执行授权、formal协议或已经完成的实验。

建议保留全部30个固定端点，只用source-support，`obs[t−1] → action[t]`、t≥1、CNN-only、sequence10/stride10、batch32、无增强。每demo末段为最后min(20,L−1)个合法动作帧，与其余合法帧对照；保存全部预测及身份，不能只保留末段。闭环5步稳定化不变，离线专家读出不运行环境或额外稳定化。

| task | source合法动作帧 | 窗口 | 每模型主批次 | 六模型主批次 | 唯一末20帧 |
|---|---:|---:|---:|---:|---:|
| 0 | 2895 | 304 | 10 | 60 | 600 |
| 1 | 4059 | 422 | 14 | 84 | 600 |
| 3 | 3064 | 317 | 10 | 60 | 600 |
| 4 | 4442 | 458 | 15 | 90 | 600 |
| 7 | 4188 | 432 | 14 | 84 | 600 |

主批次378，每模型首个原批次固定重复1次，共30重复批次，合计408。主有效预测111888行、重复9210行；主计算槽115980、重复9600（含padding）。每RGB/本体模态共享合法专家行18648，目标动作18648行，不需要action0事件标签；独特末段帧3000。提议读取5个source HDF5、加载30个指定checkpoint，训练更新/环境/初态预算0；这些将是后续新读出的读取额度，本轮没有使用。

下一阶段须单独冻结科学定义、实际GPU/数值运行设置、字段、完整预算与停止规则，实现并绑定入口后才运行。应要求新全support汇总复现当前G同权重500端点参考量、首批重复精确、配对输入不变和权重前后hash相同；容差与失败处理须运行前固定。预算仅一遍主读出加一次首批重复，不自动补跑、不挑checkpoint、seed或loss权重。需要独立审计，并继续封存正式角色。

## 验证、provenance与结论

9项新增provenance匹配测试通过（同标签异权重、异seed、错误update、缺hash和覆盖旧结果均拒绝），两份Python文件编译通过；本脚本是独立的元数据核对，没有重跑上一阶段132项专项或完整CPU回归。运行约0.109秒，数值runtime/import和payload路径guard通过、0拒绝尝试；耗时短因为没有CNN推理、backward或环境调用。

| 产物 | SHA-256 |
|---|---|
| 缓存核对结果：`/tmp/libero-source-expert-cache-availability-20261107-v1.json` | `5825a528179bd30191a9bdda0cd2b1669b3d6892f60f57aa3bca0bb10fcddc46` |
| 过程/guard记录：`/tmp/libero-source-expert-cache-availability-process-20261107-v1.json` | `84a4c7d82812adf1caa430b75fcc2a52c788fd5f2304dc050b3e3c5461799e6d` |
| 未执行预算草案：`/tmp/libero-source-expert-late-translation-readout-budget-proposal-20261107-v1.json` | `dca5e1880c27c1668a6e4928ec64461c79cb4eb5360d2cde365d0c3db90ca73b` |

结果JSON保留全部30条匹配身份和输入源码SHA，限定核对范围；专项日志为`/tmp/libero-source-expert-cache-availability-focused-tests-20261107-v1.log`。最终保护文件与谱系核对见`/tmp/libero-source-expert-cache-availability-completion-20261107-v1.json`。原生判据审计的completion、原始采集、旧预测数组、checkpoint及绑定均保留原字节，README/roadmap/500计划同步当前缺项。

下一步是冻结这次有界读出的口径和执行规则，再实现、绑定并运行；不是直接修改loss或继续训练。当前仍保持500 updates、5步稳定化和原loss，不改宽原生成功判据，不使用影子接触作为训练标签，也不注册Module或更新SPT/Knowledge。
