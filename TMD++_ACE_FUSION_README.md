# TMD++ Stage 4.2:ACE-Step base + image-ref + ACE逐层融合

基线:`Yangxiaoda1/T2MD-Bench`(未污染)。ACE 加载逻辑借用了你 repo2 的思路并改写为 base + 逐层驱动。

## 这次交付的文件

| 文件 | 作用 | 状态 |
|---|---|---|
| `ovi/modules/fusion_acestep_layerwise.py` | **核心**:Wan(30层) ↔ ACE-base(24层) 真·逐层双向融合 | 新增 |
| `ovi/utils/acestep_loader.py` | ACE base 加载 + VAE 编解码 + encoder_hidden + **AceDiTRunner(逐层驱动)** | 新增/改写 |
| `examples/Ovi/train_tmdpp.py` | SFT 训练模块:image-ref 首帧 clean + 共用调度 + 双 loss | 新增 |
| `inference/tmdpp_infer.py` | 与训练严格镜像的推理 | 新增 |
| `ovi/configs/training/finetune_tmdpp.yaml` | 训练配置 | 新增 |
| `examples/Ovi/run_tmdpp_overfit.sh` | 过拟合启动脚本 | 新增 |

## 关键设计决策

1. **逐层**。每个 fusion rendezvous 取双方 hidden 快照,同 step 双向 cross-attn,
   再各跑原生层。深度不齐(30 vs 24)用调度表吸收:24 个 rendezvous,其中 6 个跑 2 个 video 层
   (已验证 `sum=30`)。这是相对你 repo2「塔级 + 跨 step 延迟」的根本修正。

2. **不改两塔原生 forward**。融合是额外残差子层,`o_proj` 与 `gate` 零初始化
   ⇒ 训练起点融合残差恒为 0(已验证),模型起步 = 两塔各自预训练行为,稳定可控。
   ACE 的 text/lyric/timbre cross-attn 原样保留 ⇒ 满足「复用」。

3. **时间步对齐口径(训推一致的关键)**:两塔共用同一个 `FlowMatchScheduler` 和同一个 timestep;
   ACE 侧 `timestep_r = timestep`,meanflow 的 (t−r) 项=0,退化为标准 rectified flow。
   训练 / 推理走同一套单一 timestep ⇒ 结构上保证 train==infer。

4. **image reference**:取 video latent 首帧作为 clean ref,覆盖加噪首帧,`first_frame_is_clean=True`,
   首帧不计 loss;推理每步同样覆盖首帧。与推理引擎原有 i2v 路径完全对应。

5. **冻结**:video VAE / ACE VAE / T5 / ACE encoder / 两塔 backbone 全冻,只训 `v_fuse`/`a_fuse`。



## 过拟合验收标准(Stage 4.2 sanity)

先 10 条干净样本、480×480、关 CFG(`guidance=1.0`):
- `loss_video`/`loss_audio` 稳定下降,不 NaN;
- 同 prompt + 同 ref 推理结果**逐渐贴近 GT**(你说的三段对比:GT / 独立推理 / 过拟合后推理 → 第三段音画 align);
- 训推一致:训练 step 的 timestep 调度与推理一致,输出无系统性偏移。

通过后再:升 720、扩到高质量子集、按 proposal 进入 full 训练。


