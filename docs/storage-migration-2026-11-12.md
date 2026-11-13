# KECRL 存储迁移记录（2026-11-12）

已按用户要求将项目迁到 `/data/zl202621/KECRL`，保持目录分类、Git 历史、既有修改和未跟踪文件。项目、LIBERO、用户缓存和 Conda 包缓存四组复制均经 `rsync -aHAXnci` 校验，无差异；路径切换及运行检查通过后，已移除固态盘上的原副本。旧地址保留兼容链接。

## 迁移范围

| 原地址（现为链接） | 新的实际地址 | 迁移逻辑字节数 |
| --- | --- | ---: |
| `/home/zl202621/project/KECRL` | `/data/zl202621/KECRL` | 19,585,404,433 |
| `/home/zl202621/third_party/LIBERO` | `/data/zl202621/KECRL/third_party/LIBERO` | 6,904,610,461 |
| `/home/zl202621/.cache` | `/data/zl202621/KECRL/cache` | 7,961,813,905 |
| `/home/zl202621/.conda/pkgs` | `/data/zl202621/KECRL/cache/conda/pkgs` | 768,960,963 |

合计 **35,220,789,762 字节（35.22GB / 32.80GiB）**，为各源目录的逻辑大小之和，不等同于精确释放的物理块数。迁移后系统盘可用空间约663GiB，个人数据盘配额为1TiB。

使用 `rsync -aHAX` 保留隐藏文件、权限、时间、硬链接、ACL 和扩展属性，不排除被 Git 忽略的实验产物。复制校验不解码 HDF5/NPZ、加载模型或分析封存数据；物理字节读取用于本次获授权的文件迁移与校验，不能记为实验分析或新研究结果。

现有 Conda 环境仍在 `/home/zl202621/.conda/envs/`，保留原解释器及已安装包位置。没有迁移整个用户主目录、删除实验结果、提交或推送 Git。

## 后续默认位置

- 代码和工作目录：`/data/zl202621/KECRL`。
- 新数据集：`datasets/`；已有 LIBERO 数据集：`third_party/LIBERO/datasets/`。
- 模型及 checkpoint：`checkpoints/`；实验产物和完成证据：`results/`；日志：`logs/`。
- 下载和运行缓存：`cache/`，涵盖 pip、Conda 包、Hugging Face、Torch、UV、Matplotlib、Numba、Triton 和 W&B。
- 临时工作目录：`tmp/`；完成证据仍应进入 `results/` 或 `logs/`。

[storage_env.sh](../scripts/storage_env.sh) 设置这些路径及 `TMPDIR` 等环境变量，并在 `/data` 未挂载或项目目录不可写时拒绝配置。[with_storage.sh](../scripts/with_storage.sh) 加载配置、切到项目根目录并执行传入命令，保留命令退出码。

`~/.bashrc` 和 `~/.profile` 已追加加载配置的片段，原正文保留。新 Bash 登录及交互会话自动生效；当前会话可以执行：

```bash
cd /data/zl202621/KECRL
source scripts/storage_env.sh
```

也可以直接使用独立启动器，例如：

```bash
/data/zl202621/KECRL/scripts/with_storage.sh /home/zl202621/.conda/envs/libero-kecrl/bin/python scripts/check_environment.py
```

`~/.libero/config.yaml` 的五个 LIBERO 路径已改为新目录；旧目录链接同时保留。`.gitignore` 新增 `cache/`、`tmp/`、`third_party/` 忽略规则，原有 compact result 例外保持。现有项目 [AGENTS.md](../AGENTS.md) 已增加对应的存储、迁移和冻结边界要求。

已有相对 `results/...` 输出在新工作目录下写入数据盘。启动器不会重写程序显式指定的绝对路径；历史协议内的 `/tmp/...` 定义保留原内容，不能据本次迁移直接运行。未来协议须在新版本中声明持久输出位置及相应绑定。

## 历史冻结与研究边界

原 HEAD 为 `2beb6271ee66dfba423d517762ff92acfd83444f`，迁移后不变；既有4项已跟踪修改及434项未跟踪条目全部保留。迁移前1581项定义文件SHA已记录；本轮只对其中 `AGENTS.md`、`README.md`、`.gitignore` 作存储说明修改，其余1578项一致。README 原正文及 AGENTS/.gitignore 原正文均保留。

最新科学配置SHA仍为 `e345f813a492edf823e7a39cd518e511081d2541659d1d097f57c80fa41064c1`，科学绑定SHA仍为 `26ea46b5324112c3583d13f3e923566f5039cc38d4a8c0dbfe83ac12581bdc1a`。没有修改旧协议、配置、SHA 或历史结果。

**兼容链接仅保证旧路径可访问，不保证原冻结可验证。** 953项闭包中仍有30项缺失，6份源JSON及25份源NPZ仍全部缺失；另外917项现存依赖的 `Path.resolve()` 地址已因迁移变化。旧工具将真实路径与SHA一起绑定，故当前不能声称旧冻结校验通过，更不能用链接绕过。后续研究执行仍需恢复真实缺失产物，并为迁移后的路径建立有来源记录的新版本/绑定；不得改旧SHA、补空产物或删依赖。

原Policy500 updates、稳定化5步、现有loss及正式评价封存保持。本轮没有研究训练、SLSQP搜索、LIBERO/Crafter环境交互或新可行性结论；历史151项测试通过仍仅为历史记录。

## 验证与证据

- 四组完整迁移 checksum 校验通过；旧链接解析及原副本移除检查通过。
- Shell语法、Git差异格式、缓存路径、Python临时目录、新登录会话默认值检查通过；启动器缺参数、子命令退出码传递及缺少数据盘时的停止行为通过。
- 运行现有最小CPU接口smoke，结果为 `completed`，输出独占 `results/storage-migration-smoke-20261112-221241/`。它仅验证受控模拟接口及新存储可写，没有覆盖旧 `results/smoke/`。
- `scripts/check_environment.py` 在新目录通过：Python3.10.22、Torch2.4.1+cu124；检查时显式隐藏CUDA。没有修改共享训练实现，未运行pytest或研究预算实验。
- 切换预检曾两次在任何重命名前停止：新增缓存/第三方目录及Git对空格文件名的引号影响了完整状态字符串比较；改为分别核对原条目和新增存储条目，并核对全部1581项定义文件SHA后才切换。没有丢失或覆盖原工作区。

持久迁移证据在 `/data/zl202621/.kecrl-migration-20261112-221241/`：`initial.json`、四组copy/final-sync/checksum日志、`copy-verification.json`、`switched-paths.json`、`storage-configuration.json`、`storage-path-checks.json`、`cpu-interface-smoke-check.json`、`pre-removal-integrity.json`、`post-migration-provenance.json`，以及用户Shell/LIBERO配置的修改前副本。该目录位于个人私有目录中，配置副本不加入Git。

维护后尚存的两份旧handoff核对文件已另复制到 `logs/recovery-metadata-before-storage-migration-20261112-221241/`，原 `/tmp` 文件未改写；这不是缺失实验产物的恢复。
