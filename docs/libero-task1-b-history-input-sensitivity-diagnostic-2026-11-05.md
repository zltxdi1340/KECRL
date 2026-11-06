# task1 B固定历史输入表示敏感性冻结记录（2026-11-05）

执行状态更新：用户授权的60批读出与独立审计已完成，见[实际结果](libero-task1-b-history-input-sensitivity-results-2026-11-05.md)。主条件三个seed均为真实历史成本略低，未支持此代理下历史更脆弱；以下保留冻结阶段快照，“尚未执行”只描述当时状态。

**配置、入口、测试、协议及哈希已冻结，metadata预检通过；本轮尚未执行研究读出，没有新的敏感性结论。** 真实缓存数值解码、研究Linear/CNN前向、训练、环境、初态读取均0；合成测试与科学预算分开。

## 目的、范围和对照

[此前专家目标误差](libero-task1-b-fixed-head-expert-target-error-results-2026-11-04.md)显示真实历史整体更准，而此前同权重闭环只有重复当前在seed24增加4次成功。分布偏移仍是待检验假设，本项先定位固定动作头对历史表示变化的响应。

本轮冻结**表示空间代理诊断**：使用已有公开观察导出的256维融合特征，不重算RGB/本体输入或编码器，不声称检验了实际部署观测分布。固定task1 B三seed、同一候选500头及support scaler，使用全部source meta-query demos30..49、每seed2518个完整历史帧。原500端点、稳定化5步及现有loss保持，无新训练或环境。

| 项目 | 冻结设置 |
|---|---|
| 方向 | 每专家帧256维±1，PCG64测量seed20261105；三模型共用同一方向表 |
| 幅度 | support标准化坐标RMS 0.01与0.05，均使用正负成对扰动 |
| 位置 | 当前帧、过去四帧、全部五帧同向变化 |
| 主比较 | 全窗口0.05扰动，两输入能量匹配，比较各自相对干净输出的平移MSE成本 |
| 次报告 | 0.01剂量、当前/过去位置、旋转/运动6维/夹爪/七维、五个既有专家分层 |
| 成本 | S=0.5*(MSE_plus+MSE_minus)-MSE_clean，可为负；主差S_history−S_repeat |
| 独立响应指标 | 成对动作响应能量及RMS/epsilon，和误差方向项分开 |

干净输入先按旧合同完成float32标准化/串接，再精确转为float64；扰动加入已有float64读出算术空间，不重新量化。epsilon不是实测部署噪声或物理量；原target只作固定参照，不假定合成表示仍对应真实状态及同一正确动作。

方向按目标帧窗口定义，重叠窗口中的同一历史特征可能得到不同delta；本项不构造时间上一致的扰动轨迹，也不模拟连续传感器噪声。

整窗口两输入每帧扰动L2²同为256*epsilon²。当前单独扰动时，两输入能量不同，仅作次级定位；过去单独扰动时重复当前输出按合同不变，是路由检查，不能记作鲁棒性收益。同线性头的整窗口logit变化应相同；tanh后的差别应联系工作点与原target误差解释，不能称为上游编码器放大或历史物理因果机制。

## 门槛和预算

身份输出须重放上轮两输入预测；零扰动精确重放身份。过去帧干预不能改变当前夹爪，任一条件两臂夹爪一致。匹配输入能量、coherent logit增量及正负logit对称余差<=1e-10，身份桥接与MSE分解余差<=1e-12。异常、角色、hash、时钟、参数或预算失配保存failed、主结论无效，不补帧/seed、重试或放宽门槛。

| 项目 | 冻结预算 |
|---|---:|
| 身份矩阵 / 零扰动矩阵 | 6 / 6批 |
| 扰动矩阵 | 48批 |
| Linear算术总数 / 行数 | 60批 / 151080行 |
| 逻辑预测条件表 / 行数 | 84 / 211512 |
| 目标 / feature / 候选头 / 父预测归档解码 | 1 / 3 / 3 / 3 |
| 专家帧×模型配对 | 7554 |
| 新CNN/训练/环境/HDF5/RGB/PT参数/初态 | 全0 |

同epsilon/sign的当前读出在current_only和coherent_all复用；past_only复用干净当前输出，历史夹爪也复用对应当前读出。因此逻辑条件数大于实际矩阵调用，不存在隐藏额外前向。CPU单线程、Python3.10.22、NumPy1.22.4、CUDA隐藏；默认只作metadata核对。详细公式、索引和解释边界见[完整协议](libero-task1-b-history-input-sensitivity-protocol-2026-11-05.md)。

## 实现与验证

新增[执行入口](../experiments/run_libero_task1_b_history_input_sensitivity.py)、[配置](../configs/libero-task1-b-history-input-sensitivity-diagnostic_v1.json)和[测试](../tests/test_libero_task1_b_history_input_sensitivity.py)，复用既有合法帧计划、标准化与目标误差函数；历史已执行文件及绑定不改。入口保存全部三seed、所有预定幅度/方向/位置及分层，记录attempted/completed、共享方向表、动作/logit、输入能量及控制余差，不按结果挑条件。

58项专项及相关专家误差/独立审计/runtime接口测试通过（0.85s），两份新Python编译、CPU环境检查通过。合成测试验证20次矩阵/seed和28张逻辑表的复用计数、三位置输入能量、零扰动、过去负对照、同logit扰动下不同tanh响应、负MSE成本保留、预算失败计数、metadata零解码及旧输出保护。未重复全量pytest，可选Crafter不在本轮范围。CUDA隐藏下环境工具available=false表示CUDA不可用，Torch本身正常。

| 冻结对象 | SHA-256 |
|---|---|
| 配置 | a8574a7d3c0ded9cf1fdbedb86dd8e791dee99f8e1b5a43ca0c7e23acf2e8735 |
| [代码绑定](../configs/libero-task1-b-history-input-sensitivity-binding_v1.json) | 1ac34527fa7a48fae54e2ed9892dff2cee8c6a3ddbd57924f24093164f3fd3f2 |
| 执行入口 | 7dc42fa840aeea2764fdf4c8136d82346a67ac002cfac3148e9cecbb2eaf02c4 |
| 测试 | 763762313e5cdd4d76a901a6b0750e657fc60544296f60dbb425bc127aab3097 |
| 协议 | 8c54b1fdc7dff64bbf0f998222c372f62d15d56aceff9573ff783fe71ab00643 |

默认预检核对34个源文件pin、本轮三份执行绑定及父专家读出的冻结链和独立审计，不解码真实NPZ。初始工作区快照为`/tmp/libero-task1-b-history-input-sensitivity-start-20261105-v1.json`；1215份既有文件受保护，仅README、roadmap和下一步计划追加，AGENTS.md/.vscode及无关改动保持。HEAD仍为e525d19cb45a40cc6df421b6c33b96445500c2b0。

最终验证记录为`/tmp/libero-task1-b-history-input-sensitivity-freeze-verification-20261105-v1.json`，保存最终预检、预算独立核算、hash、测试和文件保全凭据。它是冻结验证记录，不是研究结果。

未来执行命令如下；本轮仅执行了不含--execute的metadata预检：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
conda run -n libero-kecrl python -m experiments.run_libero_task1_b_history_input_sensitivity --execute
```

未来结果`/tmp/libero-task1-b-history-input-sensitivity-20261105-v1.json`及同前缀artifacts目录当前不存在。排他创建，一个共享plan/方向NPZ与三seed全部条件NPZ留在/tmp，不复制权重入仓库，不覆盖旧结果。

## 当前决定和下一步

当前只能确认诊断已具体冻结、入口和控制逻辑通过合成验证，尚不能判断历史更敏感或支持真实分布偏移。下一步按60批冻结预算读出并独立审计；先报告主成本差、动作响应、线性控制与分层反证，再决定是否需要原观测层的另冻验证或训练修改。

保持500端点、稳定化5步、现有loss及旧候选停止；validation、held-out、qualification、reserved和final evaluation继续封存，不更新SPT/Knowledge、不注册Module。
