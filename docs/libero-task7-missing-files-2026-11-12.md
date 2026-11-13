# task7 缺失文件与恢复清单（2026-11-12）

2026-11-13更新：[A来源真实重跑](libero-postmaintenance-A-rebuild-results-2026-11-13.md)已在持久盘重新生成15份匹配原SHA的文件字节，另79项尚无已核验对应物；没有重填原/tmp路径或修改以下原始94项核对快照。映射见[真实恢复摘要](evidence/libero-postmaintenance-A-rebuild-summary-2026-11-13.json)。

核对时间：2026-11-12T23:05:49.886341+08:00。项目实际目录为 `/data/zl202621/KECRL`。**当前核对范围内94个唯一文件缺失；917项现存依赖发生真实路径变化；6个未来执行文件尚未实现。**

范围为当前有界约束可行性诊断、上轮有限网格的五份科学/执行元数据及上轮结果报告，不代表整个项目所有历史产物的缺失总量。新增统计来自补入父级验证证据、旧结果及原Policy引用；当前诊断的30项依赖、6份源JSON和25份源NPZ仍与此前一致。

生成机器清单的步骤只打开五份本地配置/绑定及一份已有报告，其他引用目标只作stat/真实路径核对；工作区指导、交接和迁移说明另行阅读。表内SHA全部来自原元数据或报告，未读取/哈希研究数值文件、解码归档、运行训练或搜索；文件存在性不代表完整性或执行准入通过。

## 去重汇总

| 优先级 | 用途 | 缺失数 |
| --- | --- | ---: |
| P0 | 冻结依赖/父链元数据 | 30 |
| P1 | 本轮直接输入：P/U源JSON | 6 |
| P1 | 本轮直接输入：shared归档 | 1 |
| P1 | 本轮直接输入：current features归档 | 6 |
| P1 | 本轮直接输入：原辅助头归档 | 18 |
| P2 | 当前科学冻结验证证据 | 8 |
| P2 | 上轮科学冻结验证证据 | 6 |
| P2 | 上轮执行验证证据 | 11 |
| P2 | 上轮有限网格结果 | 2 |
| P3 | 原Policy checkpoint（历史恢复项） | 6 |
| 合计 | 按原始绝对路径去重 | 94 |

P0为当前冻结闭包的缺失父链/依赖；P1为未来搜索的直接数值输入；P2为当前/上轮验证与结果证据；P3为来源Policy的历史恢复项。四类均需保留原来源和SHA。P3中的6个checkpoint不属于本诊断准许读取的输入，不得因本清单而加载或重训。

953项冻结闭包的存在性分解：30缺失 + 917存在但真实路径变化 + 6存在且路径未变。后两类本次未重新计算内容SHA。即使补齐缺失文件，旧Path.resolve/SHA绑定仍不能自动适用于迁移后的目录。

## 恢复与后续顺序

1. 把本清单或administrator-paths.txt交给管理员，按原始路径查维护前快照、旧盘副本或可信备份；当前没有确认可用的备份位置。
2. 找到的原文件先保留在个人数据盘的独立恢复目录，并记录备份来源、导出时间和原路径，保留全部字节；本轮没有创建该目录或任何占位产物。
3. 恢复核验需遵守研究访问门。不得补空JSON、伪造completion、用新跑的文件冒充旧证据、改旧SHA或删除依赖。没有原文件时继续标为缺失，另行冻结重建方案。
4. 父链和输入真实恢复后，为新持久路径建立有来源记录的新版本/绑定；再继续入口、guard、独立审计和合成验证。科学冻结不授权立即读取数值或启动搜索。

原Policy500 updates、稳定化5步、现loss和正式评价封存保持；历史151/204项通过均为历史记录。本次没有声称重验通过或产生研究结果。

## 完整缺失路径及原SHA

### P0 · 冻结依赖/父链元数据（30）

| 原始路径 | 原SHA-256 |
| --- | --- |
| `/tmp/libero-task7-direction-penalty-tradeoff-driver-20261110-v1.py` | `4db21205b9c86bbb8b9825c0c6b4181e1737c9d7f60a0f8c8f4e424bce9e2fd6` |
| `/tmp/libero-task7-endpoint-gradient-20261109-v1-process.json` | `ba994551f3b9d0fa081af163d805a0d0652819bf6a6d0d830d45b58b5f63033e` |
| `/tmp/libero-task7-endpoint-gradient-20261109-v1.json` | `f56da26798fc9d43f718f37af75d50f0c1c9db7d2d6c1b734e941d7ae7d39e54` |
| `/tmp/libero-task7-endpoint-gradient-20261109-v2.json` | `43ace1221276fdfd68aaaaed9fda6366e0120d608856762d7f956dbee2886601` |
| `/tmp/libero-task7-endpoint-gradient-artifacts-20261109-v1/completion.json` | `531c9f8428a13ef045986c948fe2bb3df82f9b719581773b8dd5ee5e80b0d5f1` |
| `/tmp/libero-task7-endpoint-gradient-artifacts-20261109-v1/independent-audit.json` | `caddfc87b11374225fe486b285613b99fde15ec7cbb7ace8cf41299986fb8a3e` |
| `/tmp/libero-task7-endpoint-gradient-artifacts-20261109-v2/completion.json` | `f5c2ee4ed8b51e545c92d40043ca881774d8d475c63d873049bf550f074a6b29` |
| `/tmp/libero-task7-endpoint-gradient-artifacts-20261109-v2/independent-audit.json` | `698fe898fc7525416556e1175a368831bcf827a7261f2cd9d304383289c1908e` |
| `/tmp/libero-task7-endpoint-gradient-v1-budget-deviation-20261109-v1.json` | `daced34af4a4bbd67794b670383e643dbf21727b890fb3b262293ec2eb12d991` |
| `/tmp/libero-task7-fixed-representation-optimization-20261109-v1.json` | `33021ef69f0040396d33923b5606c62248815c55d0c3655d76cfcc94515b1132` |
| `/tmp/libero-task7-fixed-representation-optimization-artifacts-20261109-v1/completion.json` | `4c79178ede41348124177ccd6c53caf6e1e567f3b4fcf6a0c7fc1e2c63fc1832` |
| `/tmp/libero-task7-fixed-representation-optimization-artifacts-20261109-v1/independent-audit.json` | `88acec5c4276bec92778237448abdf7da71eab2fb189c5c14509e500458ad6d2` |
| `/tmp/libero-task7-joint-t-tradeoff-gradient-20261111-v1-process.json` | `7cff0ca7892c4876357bc782527d8daa1e24879d5cd63e2b5ea68d58f38c42e2` |
| `/tmp/libero-task7-joint-t-tradeoff-gradient-20261111-v1/completion.json` | `fd5f972a4092ea74eb026aed4ba374ada4886b81e9488381b0f7cd4622eebf48` |
| `/tmp/libero-task7-m-priority-direction-20261111-v1-process.json` | `6bedaabb41f0bcaf3437080127e51023079acf7a42faecd3bba1be28c4607aab` |
| `/tmp/libero-task7-m-priority-direction-20261111-v1.json` | `8d869c68ed5543ce6794f5eb0e5bf6024a948aa18781f9be8cbd4c3202d930fc` |
| `/tmp/libero-task7-m-priority-direction-artifacts-20261111-v1/completion.json` | `9e2180fb5b2755e2f1541b4da5710a8df944c233d52bcfea04893459a1cc965b` |
| `/tmp/libero-task7-needed-direction-reporting-binding-20261109-v1.json` | `bd7df78e4583de90142ba68d7f6bdd72c7ff62f7574031c4abebf1fa05db3fe3` |
| `/tmp/libero-task7-original-m-bound-joint-t-direction-20261111-v1-process.json` | `5c432f70c000700eaf1356d695fab9c51b400805f8b8cb90882b0a72e82053bb` |
| `/tmp/libero-task7-original-m-bound-joint-t-direction-20261111-v1.json` | `35be3fa0440ea272cf8b8a95b7aa5a6ffbe3a6b4515a226afde1fbebded32ae9` |
| `/tmp/libero-task7-original-m-bound-joint-t-direction-artifacts-20261111-v1/completion.json` | `ea45029c864275de2368abde9bf7539af77354116dcfd5d6db0824454817ad33` |
| `/tmp/libero-task7-reference-joint-reachability-20261111-v1-process.json` | `e437b0ea25ee6e55557562af2d2cdc28262fd3ea66266bff8a297db6ec55db8b` |
| `/tmp/libero-task7-reference-joint-reachability-20261111-v1/completion.json` | `c6f24db92786c8dc3583ae4b32d5624e64e3bdd7137cca1e874cc856c1c0a0ee` |
| `/tmp/libero-task7-stepwise-m-nonincrease-acceptance-20261111-v1-process.json` | `5b614ce3ba2c9ebe45b81795fc9ebec44316e8e2f8c869997a6dfe92d232cf16` |
| `/tmp/libero-task7-stepwise-m-nonincrease-acceptance-20261111-v1.json` | `e3bc2ec68d94566cad5777695cba3d084b33c6ff526c2229734008a574f7ea50` |
| `/tmp/libero-task7-stepwise-m-nonincrease-acceptance-artifacts-20261111-v1/completion.json` | `733c90695c44475dd946c60d5dbf0f63f1b1ccaf0f3bc912d6469ac71dfaae87` |
| `/tmp/libero-task7-stepwise-m-reference-tradeoff-20261111-v1-process.json` | `617c3cd948cf10f5c563c2c7c74f51827146df4219a0171aa6a43339bc0f5106` |
| `/tmp/libero-task7-stepwise-m-reference-tradeoff-20261111-v1/completion.json` | `290fa88ed90d95d65276ec403b4771f918ad0a2e2ae5b95a87b5425f52909d9e` |
| `/tmp/libero-task7-terminal-gradient-geometry-20261111-v1-process.json` | `65ac86fce94fea579db218bf3445d84c2d91b38c9f619099d6ae12b1a488deae` |
| `/tmp/libero-task7-terminal-gradient-geometry-20261111-v1/completion.json` | `ca83a93c1575bb0517ab95c0c5bf3b49b1488bcd7b045da9db77970fd213ab38` |

### P1 · 本轮直接输入：P/U源JSON（6）

| 原始路径 | 原SHA-256 |
| --- | --- |
| `/tmp/libero-task7-original-m-bound-joint-t-direction-artifacts-20261111-v1/independent-audit.json` | `ab7772504153076722b9a66775b32106a46e2fb030ba66dd3346db4cd5cca9ef` |
| `/tmp/libero-task7-original-m-bound-joint-t-direction-artifacts-20261111-v1/primary-result.json` | `942078ca737ad5779b3104a29d1557e89226d4163e6bfaecbcebf83bf4935dde` |
| `/tmp/libero-task7-stepwise-m-nonincrease-acceptance-artifacts-20261111-v1/independent-audit.json` | `cf9af192c284d13bda471357e6f0d065b5ba71034c8f0fc7cb29e786bdd3295c` |
| `/tmp/libero-task7-stepwise-m-nonincrease-acceptance-artifacts-20261111-v1/primary-result.json` | `3de285874988df32ef5e7114c3ed4cd6c604a5bb042f45cb39f9596b074b7cde` |
| `/tmp/libero-task7-t-priority-direction-artifacts-20261111-v1/independent-audit.json` | `bcd3fcaa30b69d1604cdb8a377b8cd0df7fb828580300c4acf663ec3896f8e4d` |
| `/tmp/libero-task7-t-priority-direction-artifacts-20261111-v1/primary-result.json` | `bc1997a63248460bfcbdeb7b5be1db55ae038200900544fbc0d1f5507c030cf9` |

### P1 · 本轮直接输入：shared归档（1）

| 原始路径 | 原SHA-256 |
| --- | --- |
| `/tmp/libero-source-expert-component3-observability-artifacts-20261107-v1/task7-shared.npz` | `8f1538cc1677e7a7e05f7d0292308230d8e0f0a98d2547955cf37cf6d176863b` |

### P1 · 本轮直接输入：current features归档（6）

| 原始路径 | 原SHA-256 |
| --- | --- |
| `/tmp/libero-source-expert-component3-observability-artifacts-20261107-v1/task7-seed20261106-A_original-features.npz` | `9fa10f9abd87f86f9fa85f5f86c0b2e9c43f7a976c3d7152816bc1160e17e36e` |
| `/tmp/libero-source-expert-component3-observability-artifacts-20261107-v1/task7-seed20261106-B_fomaml-features.npz` | `48276e5e67ed55198d12845fa2767571a9b0127cb3eef26f4a0537ba548d6bde` |
| `/tmp/libero-source-expert-component3-observability-artifacts-20261107-v1/task7-seed20261107-A_original-features.npz` | `16ae32045daa66b5663e109d55c07b33ad19349e69e1d2f3eae524cdfdbf1357` |
| `/tmp/libero-source-expert-component3-observability-artifacts-20261107-v1/task7-seed20261107-B_fomaml-features.npz` | `078d7fc66aa6347639bbf568603ae63c35bac667211554ae102308b326daa5ff` |
| `/tmp/libero-source-expert-component3-observability-artifacts-20261107-v1/task7-seed20261108-A_original-features.npz` | `34be04be0b2fe895628a25245d0473a621302b582ac8c8543c66d078f0059f96` |
| `/tmp/libero-source-expert-component3-observability-artifacts-20261107-v1/task7-seed20261108-B_fomaml-features.npz` | `b6cbb6418e743bb3e53f0da0a04859e313909d7c5181362b66238de486fdea82` |

### P1 · 本轮直接输入：原辅助头归档（18）

| 原始路径 | 原SHA-256 |
| --- | --- |
| `/tmp/libero-task7-residual-harm-protection-artifacts-20261108-v1/head-task7-seed20261106-A_original-fold0-harm_protected_affine.npz` | `82b97ef4bc38a2b147a421d1001b8d2d8d3af568cf105c7459226f87cde19741` |
| `/tmp/libero-task7-residual-harm-protection-artifacts-20261108-v1/head-task7-seed20261106-A_original-fold1-harm_protected_affine.npz` | `a322833f5ac262333ce21e4cafde781c82ff42509ce2b38c1b1bcae99f7dc572` |
| `/tmp/libero-task7-residual-harm-protection-artifacts-20261108-v1/head-task7-seed20261106-A_original-fold2-harm_protected_affine.npz` | `7ac26f9dbf207515653cac0a2f6e45cd463726354e067be645a21b3073f1967a` |
| `/tmp/libero-task7-residual-harm-protection-artifacts-20261108-v1/head-task7-seed20261106-B_fomaml-fold0-harm_protected_affine.npz` | `52a4f5891e8775716adc68869e767be646cd93d786ba6ce3984db9272cf79578` |
| `/tmp/libero-task7-residual-harm-protection-artifacts-20261108-v1/head-task7-seed20261106-B_fomaml-fold1-harm_protected_affine.npz` | `a8b3eb521df9cf24ac6cd16bc6f8be487b99528b826d0506c164ab72c1f6351a` |
| `/tmp/libero-task7-residual-harm-protection-artifacts-20261108-v1/head-task7-seed20261106-B_fomaml-fold2-harm_protected_affine.npz` | `a30a5652ad015fd704540ad9e88c0f905911fc2fafdb041dc931a9bc38679747` |
| `/tmp/libero-task7-residual-harm-protection-artifacts-20261108-v1/head-task7-seed20261107-A_original-fold0-harm_protected_affine.npz` | `89fda4837e0569cfe28ceeb043178657070319dcca26078728aa522d224741c4` |
| `/tmp/libero-task7-residual-harm-protection-artifacts-20261108-v1/head-task7-seed20261107-A_original-fold1-harm_protected_affine.npz` | `f5e92a517e892251d0484c799740684e32b4eacadb541a1ed7bb2b3a7e2c911a` |
| `/tmp/libero-task7-residual-harm-protection-artifacts-20261108-v1/head-task7-seed20261107-A_original-fold2-harm_protected_affine.npz` | `d2c3ea9de3022ba8de8eae3ad5937ac568ddc511c6cc72476bb02ba0456bbb4e` |
| `/tmp/libero-task7-residual-harm-protection-artifacts-20261108-v1/head-task7-seed20261107-B_fomaml-fold0-harm_protected_affine.npz` | `114bc85adbd6ae3e9f96fa9ab915e4e01dcf2209e777a8cb7872d5737ffa89f8` |
| `/tmp/libero-task7-residual-harm-protection-artifacts-20261108-v1/head-task7-seed20261107-B_fomaml-fold1-harm_protected_affine.npz` | `204c01017961b9abfcb236179e9b8a7d8946ecffcf25adef0798f63639c0cf0c` |
| `/tmp/libero-task7-residual-harm-protection-artifacts-20261108-v1/head-task7-seed20261107-B_fomaml-fold2-harm_protected_affine.npz` | `7aab3dca7a90b2dbf0f191960541cea77bf207a5352e257f34ac8481fc3441d5` |
| `/tmp/libero-task7-residual-harm-protection-artifacts-20261108-v1/head-task7-seed20261108-A_original-fold0-harm_protected_affine.npz` | `1cb1f50a19cbf31f3b52bb4610d995651676bf65fd8900cfcf4fa3525c3b3231` |
| `/tmp/libero-task7-residual-harm-protection-artifacts-20261108-v1/head-task7-seed20261108-A_original-fold1-harm_protected_affine.npz` | `c7f21c6f83303231bd9c66a012b279f94b242a3a2dd561f4bce902685e7708f8` |
| `/tmp/libero-task7-residual-harm-protection-artifacts-20261108-v1/head-task7-seed20261108-A_original-fold2-harm_protected_affine.npz` | `b22c97755910193ee595230ba03d99aae5ca807879fdf2d26c27bf83d1ea1432` |
| `/tmp/libero-task7-residual-harm-protection-artifacts-20261108-v1/head-task7-seed20261108-B_fomaml-fold0-harm_protected_affine.npz` | `05af7c575ee5c10c933a86a5d2a0dfa2290679d271a358d4e251d2bfe6da1880` |
| `/tmp/libero-task7-residual-harm-protection-artifacts-20261108-v1/head-task7-seed20261108-B_fomaml-fold1-harm_protected_affine.npz` | `348ae4eb4e2a2e90d7e004965ac7d62909e247e15fa9d94ae9a3395d60b72f3f` |
| `/tmp/libero-task7-residual-harm-protection-artifacts-20261108-v1/head-task7-seed20261108-B_fomaml-fold2-harm_protected_affine.npz` | `8a72ae8075b97ab83fb16342bf7bf3f08077a8a946c16c23531d451b8a050866` |

### P2 · 当前科学冻结验证证据（8）

| 原始路径 | 原SHA-256 |
| --- | --- |
| `/tmp/libero-task7-constrained-feasibility-environment-20261112-v1.log` | `55d67e99a3ff0fbe20bee50cbc2b1b88e94a523decb57a8b4dd822ad211ea5c2` |
| `/tmp/libero-task7-constrained-feasibility-freeze-initial-20261112-v1.json` | `7eb23cb20aabfed61c87a373177f1416b7cce3d6ac71b1458520c9c9ae36cefb` |
| `/tmp/libero-task7-constrained-feasibility-tests-bound-20261112-v1.log` | `82a7e192bf4ac40c9ba4276db9959bc1dba9b27499644a52cf3a318d564b3b19` |
| `/tmp/libero-task7-constrained-feasibility-tests-bound-20261112-v1.xml` | `623597d01080245f04f8908731338d29bb12f782c8e5688bb953439e474d3c65` |
| `/tmp/libero-task7-constrained-feasibility-tests-final-20261112-v1.log` | `e64b680ad2aa1ccabdb90ca5ee8c8adfba41587e46a4053c94d7b14323b7632a` |
| `/tmp/libero-task7-constrained-feasibility-tests-final-20261112-v1.xml` | `9d3e7451314fdd8a058595877b3d742282f2491e8c14ef4e6bd3842825d04944` |
| `/tmp/libero-task7-constrained-feasibility-tests-initial-20261112-v1.log` | `6312b1c53bf4545cf17d762df246eed04d42a6d195cbdc1d7eb5bec98a8c4836` |
| `/tmp/libero-task7-constrained-feasibility-tests-initial-20261112-v1.xml` | `b64556575af60403cf80f645ae47fab323d4deac6690e103bbb0caca794aaf11` |

### P2 · 上轮科学冻结验证证据（6）

| 原始路径 | 原SHA-256 |
| --- | --- |
| `/tmp/libero-task7-reference-joint-reachability-environment-20261111-v1.log` | `55d67e99a3ff0fbe20bee50cbc2b1b88e94a523decb57a8b4dd822ad211ea5c2` |
| `/tmp/libero-task7-reference-joint-reachability-freeze-initial-20261111-v1.json` | `7619fdbfd48a6b756ce8147a0ef5e289ea0a0e65d1c6368e83acf706c67e8403` |
| `/tmp/libero-task7-reference-joint-reachability-tests-final-20261111-v1.log` | `7923940942c240444197029ca7c212e015e9e1d2a98390084f0ebfeea61d884b` |
| `/tmp/libero-task7-reference-joint-reachability-tests-final-20261111-v1.xml` | `fb9333a5871b78091c0d9dbf98e3e18ce3033694ec76bae4be2389d6a0810be0` |
| `/tmp/libero-task7-reference-joint-reachability-tests-initial-20261111-v1.log` | `55f19922c5d031b7470e205df19bb6eea77cd2d7bec8d19600a12274dfe37fc4` |
| `/tmp/libero-task7-reference-joint-reachability-tests-initial-20261111-v1.xml` | `f7deac5d79fb01e20a4bd2764aad1f1e4839cd4393f410dc5ca02618364f792f` |

### P2 · 上轮执行验证证据（11）

| 原始路径 | 原SHA-256 |
| --- | --- |
| `/tmp/libero-task7-reference-joint-reachability-execution-environment-20261111-v1.log` | `55d67e99a3ff0fbe20bee50cbc2b1b88e94a523decb57a8b4dd822ad211ea5c2` |
| `/tmp/libero-task7-reference-joint-reachability-execution-human-envelope-20261111-v1.json` | `2efa9084f740d0fdf74a08a9788effffadefa1479809d6235d05f6c18fb99a25` |
| `/tmp/libero-task7-reference-joint-reachability-execution-initial-20261111-v1.json` | `6e950ce0b613cd7c8e9d736606778675c9d5ac9a1496ed6df422b4bcb3ad02ac` |
| `/tmp/libero-task7-reference-joint-reachability-execution-tests-bound-20261111-v1.log` | `901f69553d45fbee5fdac048bc226aa4f28d325c2ddfbfcf7df2aaa00edbc68e` |
| `/tmp/libero-task7-reference-joint-reachability-execution-tests-bound-20261111-v1.xml` | `fcf27fd210adc5f0e4129f5e56a45cd4f8212886260e710b6c73513b653df33b` |
| `/tmp/libero-task7-reference-joint-reachability-execution-tests-final-20261111-v1.log` | `b11e14a1f923a0aee5747a3c75add66d66a7b8818da96881a9b820bec9ab05d0` |
| `/tmp/libero-task7-reference-joint-reachability-execution-tests-final-20261111-v1.xml` | `ffb634f36faba14d4da799775dd0f0eee9c2d784f767fe299d705acc336c8f97` |
| `/tmp/libero-task7-reference-joint-reachability-execution-tests-initial-20261111-v1.log` | `a6b2aabe0127d52ea842af4b15c923b5d1ca1c452f47d5dd5b662410da3e691d` |
| `/tmp/libero-task7-reference-joint-reachability-execution-tests-initial-20261111-v1.xml` | `ab0ef67774b79ec21b46876199275108009b3170a4f0861cb673d528c9fe7ccf` |
| `/tmp/libero-task7-reference-joint-reachability-execution-tests-v2-20261111-v1.log` | `1ae1f4838976193817f35580fb5696d4cbe83a6f902d9026be0839fc599a8a88` |
| `/tmp/libero-task7-reference-joint-reachability-execution-tests-v2-20261111-v1.xml` | `08826cd3bbda16f94c4fe9be200d3d951791f29c100d536dd2552f2311a06e13` |

### P2 · 上轮有限网格结果（2）

| 原始路径 | 原SHA-256 |
| --- | --- |
| `/tmp/libero-task7-reference-joint-reachability-20261111-v1/independent-audit.json` | `71cdb6a873d4b636ef18e65914ca1477b4ad955ca955d9e12bfd07cb3f8a5b33` |
| `/tmp/libero-task7-reference-joint-reachability-20261111-v1/primary-result.json` | `e65fabc965671d8ecef23acbd4345c299f2ddb5b1cfe333ac9f5a1818dc5a875` |

### P3 · 原Policy checkpoint（历史恢复项）（6）

| 原始路径 | 原SHA-256 |
| --- | --- |
| `/tmp/libero-formal-G-artifacts-20261105-v2/G-task7-seed20261106-A_original-u500.pt` | `c5cb770a456185a5ef4885311ba4aab452aa287039c8d6b9fda48d9f895a9909` |
| `/tmp/libero-formal-G-artifacts-20261105-v2/G-task7-seed20261106-B_fomaml-u500.pt` | `cba1dc95887721f1cd44068e72aff914cadf863dcda593e38d31f86c48949eaa` |
| `/tmp/libero-formal-G-artifacts-20261105-v2/G-task7-seed20261107-A_original-u500.pt` | `2058d4d47f5293e77c596f5a104529708b03a6a89052a8423c82e2030ed13ace` |
| `/tmp/libero-formal-G-artifacts-20261105-v2/G-task7-seed20261107-B_fomaml-u500.pt` | `c8d007ce68f23ce67306398adbf9e8438b401a1b6f667d515f012881bef11a26` |
| `/tmp/libero-formal-G-artifacts-20261105-v2/G-task7-seed20261108-A_original-u500.pt` | `b264be47a6ca54a1370fbdd53a2d5f0f96615f8c2f19207afef3f977e41b1538` |
| `/tmp/libero-formal-G-artifacts-20261105-v2/G-task7-seed20261108-B_fomaml-u500.pt` | `5f8740383959991e9abfc0dcc2069d5b7b067c36e26a6fdc576d16f79b7316df` |

## 待实现文件（不计入94个缺失文件）

| 角色 | 新目录下计划路径 | 状态 |
| --- | --- | --- |
| primary | `/data/zl202621/KECRL/experiments/run_libero_task7_bounded_constrained_feasibility.py` | 待实现，无历史SHA |
| audit | `/data/zl202621/KECRL/experiments/audit_libero_task7_bounded_constrained_feasibility.py` | 待实现，无历史SHA |
| guard | `/data/zl202621/KECRL/experiments/libero_task7_bounded_constrained_feasibility_access.py` | 待实现，无历史SHA |
| runtime_tests | `/data/zl202621/KECRL/tests/test_libero_task7_bounded_constrained_feasibility.py` | 待实现，无历史SHA |
| execution_contract | `/data/zl202621/KECRL/docs/libero-task7-bounded-constrained-feasibility-execution-contract-2026-11-12.md` | 待实现，无历史SHA |
| execution_binding | `/data/zl202621/KECRL/configs/libero-task7-bounded-constrained-feasibility-execution-binding_v1.json` | 待实现，无历史SHA |

## 可交接文件与核对来源

- [完整JSON清单](../results/recovery-inventory-20261112-230549/inventory.json)：1017个去重引用的状态、来源字段及读取边界。
- [缺失文件CSV](../results/recovery-inventory-20261112-230549/missing-files.csv)：94行，含原路径、原SHA、恢复优先级与来源。
- [管理员原路径列表](../results/recovery-inventory-20261112-230549/administrator-paths.txt)：94条原路径，可用于快照查找。
- [迁移路径CSV](../results/recovery-inventory-20261112-230549/relocated-dependencies.csv)：917行，单列路径变化，不记作丢失。
- [待实现CSV](../results/recovery-inventory-20261112-230549/not-yet-implemented.csv)：6行，与丢失文件分开。

机器清单写入个人数据盘results/，本Markdown另保存完整94行路径/SHA；不把尚不存在的实验产物或研究payload加入Git。所有原科学配置、绑定、历史报告及交接正文均保持原内容。

读取来源：

- `configs/libero-task7-bounded-constrained-feasibility-diagnostic_v1.json`（本地元数据SHA `e345f813a492edf823e7a39cd518e511081d2541659d1d097f57c80fa41064c1`）。
- `configs/libero-task7-bounded-constrained-feasibility-freeze-binding_v1.json`（本地元数据SHA `26ea46b5324112c3583d13f3e923566f5039cc38d4a8c0dbfe83ac12581bdc1a`）。
- `configs/libero-task7-reference-joint-reachability-diagnostic_v1.json`（本地元数据SHA `f66a06736891529a45fa2cd5b791ca6841163ab99760e1e273c11f1f5880d145`）。
- `configs/libero-task7-reference-joint-reachability-freeze-binding_v1.json`（本地元数据SHA `613dbf8a471f600ce30433aa4598be0666fee9848b5e4957c3fac79fe994136d`）。
- `configs/libero-task7-reference-joint-reachability-execution-binding_v1.json`（本地元数据SHA `251869eabe25c43b06a73c537fe76bb0816bba8e3479112038492195eed37550`）。
- `docs/libero-task7-reference-joint-reachability-results-2026-11-11.md`（本地元数据SHA `fa5dd7b3deaee984297a7af50eb56afcab0b2e2d588a034562b076c27860afb1`）。

核对逻辑：按原始绝对路径去重，拒绝同路径冲突SHA；区分FileNotFound、访问失败与非普通文件；只对真实存在的引用计算真实路径。独立复核包括原绑定集合一致性、CSV/JSON行数、原SHA来源、94行Markdown完整性和未来文件分类，结果另存输出目录。
