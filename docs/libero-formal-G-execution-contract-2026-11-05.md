# LIBERO G阶段执行绑定说明（2026-11-05）

本说明落实已经冻结的[正式科学协议v2](libero-formal-transfer-protocol-v2-2026-11-05.md)，不改变主假设、数据角色、seed、loss、500更新端点或接受门槛。用户本次明确要求实现并绑定执行入口、阶段门控、选择性初态读取后启动G。当前命令只执行G；即使准入通过，也在生成独立审计及完成收据后结束，不自动启动H。

执行入口为[run_libero_formal_transfer.py](../experiments/run_libero_formal_transfer.py)，默认`--mode preflight`只做metadata/hash核对。正式运行必须显式`--mode run --stage G`，载入[执行manifest](../configs/libero-formal-G-execution-manifest_v1.json)和[执行绑定](../configs/libero-formal-G-execution-binding_v1.json)，全部输出使用新的仓库外路径。绑定文件发生变化即拒绝执行，不覆盖历史产物、不续跑替代失败数据。

## 数据访问顺序

1. [metadata收集器](../experiments/prepare_libero_formal_execution.py)只检查source0/1/3/4/7和validation2/6的HDF5形状、轨迹长度、文件SHA与初态容器头。全容器校验是字节读取；没有读取RGB/action/joint/gripper数值或解码初态。结果`/tmp/libero-formal-G-metadata-collection-20261105-v1.json`固定G离线评价批数为25200，低于全研究120000上限；这不是模型评价。
2. 执行绑定核验和GPU1空闲检查通过后，训练20组来源生成器，每seed从相同初始A克隆B/C，固定50次外层更新、五个来源SPI共享窗口。A保持不变，B/C隔离候选不写入活动SPT。独立source审计核验60个生成器、1000次配对外层更新和采样子流后，才能开始目标Policy适应；完整训练审计在全部适应后再次核验权重和事件前缀。
3. source五任务A/B共200个Policy、validation两任务A/B/C共120个Policy各适应500次。每任务三个臂共享目标采样子流。validation每任务全部60个support fit完成后才允许创建该任务query dataset。0/50/150/300/500全部保存及离线记录，闭环只用500，不选checkpoint。
4. [独立训练审计](../experiments/audit_libero_formal_transfer.py)核验1660个checkpoint、全部采样子流、真实优化器设置、2200个dataset sweep、25200个离线batch和实际更新计数。审计器只读取保存的CPU权重、公开日志和metadata，不导入训练/执行器，不构造Policy、环境或初态。
5. 训练审计通过后，[选择性读取器](../src/environments/libero_initial_state_reader.py)仅解码source0..19共100行和validation10..29共40行。通过字节扫描跳过validation0..9；跳过行不缓冲成状态向量、不做数值解码。禁止pickle/Torch反序列化初态容器。数组全阶段复用，validation全局10映射局部0，并在轨迹中同时记录两种编号。qualification、reserved、held-out文件不作数值读取。
6. 每个模型新建adapter，按授权全局编号递增执行。固定环境seed20261022、5步零动作稳定化、最多300个Policy动作、native horizon305。同task/global初态的公开reset摘要跨全部seed/臂逐位一致；失配立即中止，不放宽容差或替换场景。
7. 全部6400条轨迹完成并保存后，独立重算终止、公开状态偏移、预算及冻结的G决定。只有完整审计通过才写`completion.json`；`result.json`在审计前标记为待核验，不能提前当成有效结果。G拒绝代表准入失败，H主检验未开展。

## 预算和真实成本

G包含160000次适应Adam、2000次生成器Adam、5000次临时SGD，总计167000次优化更新；来源数据梯度20000、生成器VJP10000单列。6400条闭环最多1920000个Policy动作，另32000个稳定化零动作，native总上限1952000。最多320个adapter、140个数值初态行、140个公开reset reference和6260次逐位比较。

逐项记录attempted/completed计数，异常不会补写成已完成。Policy对象创建也单列，包含模板构造、克隆、临时SPI实例和固定权重重载：G训练11020个、闭环320个，总11340个；它不等于前向或优化次数。源训练/目标训练（含评价）/离线评价/cache耗时均记录，后两项包含在前两项中，不能相加。记录有效预测行、峰值tensor cache、GPU峰值内存、版本、进程、设备、代码/配置/checkpoint/file hash与事件日志。

物理GPU1映射进程cuda:0/EGL1；GPU0不占用。启动检查GPU1无计算进程、显存≤512MiB、利用率≤5%；确定性算法和等价池化预检通过后才开始训练。每阶段活动wall上限16小时，cache上限8GiB。当前磁盘空间有限，启动需至少9GiB可用；运行保留512MiB余量，低于即技术中止，不删除其他实验。历史吞吐估计G仿真约9小时，训练和审计另计，不是完成时间保证。

SIGINT/SIGTERM、数值异常、角色/哈希/初态错位、预算或资源问题均保留partial、`failure.json`与尝试/完成计数。没有自动重试、追加seed或修改科学协议。H还需要独立的执行manifest、绑定和有效G准入收据；G绑定请求H会在payload读取前拒绝。H若获授权，使用不重叠的20个新seed重训source，不复用G权重。

## 绑定前验证与下一步

70项新专项测试通过：合成6400条G覆盖全局/局部映射及reset比较；真实历史数据/环境调用为0。受控微型网络联测覆盖来源50次外层更新和配对500次适应；B/C插桩前后更新逐位一致。metadata测试禁止HDF5数值切片，检查越界角色、错误哈希、预算、已有输出和H封存；另以保存的合成产物检查独立source审计拒绝缺seed、错采样、提前适应和错误优化器。非Crafter CPU回归日志保存至`/tmp/libero-formal-G-execution-cpu-regression-20261105-v1.log`，70项包含在其中；可选Crafter依赖相关测试排除，CPU检查时主动屏蔽CUDA，不表示GPU故障。

实现期间合成初态fixture曾因`torch.save`生成的ZIP根目录名与官方容器不同而被严格拒绝；已在绑定前修正fixture为官方格式，未放宽读取器、未接触真实初态数值。冻结v1/v2科学文件与原实验产物不修改；`AGENTS.md`、`.vscode/`及无关工作区修改保留。运行状态和实际启动证据另见[执行记录](libero-formal-G-execution-2026-11-05.md)，避免改写本执行绑定说明。

下一步完成执行哈希绑定及metadata预检，检查GPU/磁盘后启动G；随后等待独立完整审计和预注册准入决定。未完成时不能汇报正式主假设结论，不更新Knowledge、不接受活动SPT、不注册Module，qualification预算始终为0。
