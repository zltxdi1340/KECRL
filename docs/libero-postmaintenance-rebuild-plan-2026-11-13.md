# LIBERO维护后真实重跑方案（2026-11-13）

用户已要求重新生成实验文件、重跑并上传GitHub。本次建立新的真实运行记录；原94项缺失清单、旧SHA、旧时间戳及历史判断保持，不生成冒充旧运行的completion或测试日志。新运行不等于恢复原字节；任何与历史tensor hash的匹配/不匹配均原样记录。

## 第一段：无需解封query的A_original重跑

固定task7、seeds20261106/07/08，沿用原A生成器初始化、结构化上下文、CNN-only、均匀七维masked action MSE、Adam lr0.0003/betas0.9,0.999/eps1e-8/weight_decay0、clip norm1、batch8、sequence10/stride10，各500 updates。原Policy500、未来稳定化5步、辅助λ1和正式评价封存保持。本段不重训SPT、不运行环境或选择checkpoint。

只使用source task7 meta_support demos0..29，obs[t−1]→action[t]、t≥1。原长度元数据固定432窗口、4188合法帧；只解码actions及四个公开观察数组，禁止query demos30..49、private属性、states、初态及其他任务。整份HDF5 SHA校验会物理读取未投影字节，不能称物理字段隔离；query数组数值解码为0。

新文件`libero-postmaintenance-task7-A-rebuild_v1.json`与单独execution-binding在原payload哈希/读取前记录：配置、代码/测试/方案、原始数据预期SHA、运行版本、设备和预算。它不是历史冻结的修复或绕过；旧入口仍保持fail closed。

运行固定物理GPU1（UUID GPU-d90bd7b2-16f6-c0fe-fbc1-7ed4060e95ff），GPU0不使用。GPU1须无其他compute进程、显存≤512MiB、利用率≤5%；否则在payload前停止。Python3.10.22、NumPy1.22.4、Torch2.4.1+cu124、h5py3.11.0、CUDA12.4，OMP/BLAS/Torch线程1，TF32关闭、确定性pool沿用原实现，hash seed0。

预算：3个Policy各500，共1500实际optimizer/backward/训练前向；0 source outer update。每Policy14个batch32固定表示读出+1次首批重复，共42+3次capture forward；总HDF5打开2次、公开数组切片2160次，whole-file SHA1次；host observation tensor cache≤1GiB，主段wall≤900秒，输出使用独占持久results目录。0/50/150/300/500快照全部保存，500是唯一端点，不按结果改seed或预算。

CPU独立审计不读HDF5、不训练或调用CNN。它重新生成全部窗口/采样流，核对500步、真实Adam计数、各快照文件/tensor摘要、全部frame IDs，并以NumPy float64独立重建保存feature的线性head/logits/tanh（abs1e-5/1e-6）。与旧端点的tensor hash比较仅判重跑一致性，不替代新运行证据，也不能恢复旧时间/日志。

## 尚待澄清的B来源范围

重建B_fomaml需要原source tasks0/1/3/4/7的50步元训练，原训练反馈包含meta_query demos30..49。此前当前诊断阶段这些query封存，因此已向用户单独询问是否仅为来源重建开放。回答前本段只运行A，不以A替代B、不制造B权重或完整18头结果。validation、held-out、qualification、reserved、final及环境初态始终不在本次询问范围。

后续B、辅助头、四参照与有限网格/约束诊断必须建立各自新的输入谱系和执行绑定，不能把新文件填入旧SHA绑定。历史测试和测时不可从新的测试生成同名旧证据。GitHub上传代码、配置、方案和真实运行摘要，原始数据/模型/大数组留个人数据盘。
