# TMD++ Stage 4.2:ACE-Step base + image-ref + 真·逐层融合

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

1. **真·逐层(不是塔级)**。每个 fusion rendezvous 取双方 hidden 快照,同 step 双向 cross-attn,
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

## 我已验证 vs 你必须在有权机器上核对

**已验证(不依赖权重,实跑过)**:全部模块语法通过;调度表 30↔24 正确;融合子层能处理
3072↔2048 维度差与不同序列长度;零初始化残差恒为 0;参数可训练。

**我无法在此验证、上机第一步必须核对的点(按风险排序)**:

1. **ACE handler 的属性名**。`acestep_loader.py` 里用了 `handler.vae` / `handler.text_encoder` /
   `handler.text_tokenizer` / `handler.silence_latent` / `handler.model.decoder` / `handler.model.encoder`
   / `handler.config`。这些沿用你 repo2 的写法,但 **base 与 turbo 的 handler 接口可能不同**,
   第一步先 `print(dir(handler))` 对一遍,不对就改这一个文件。

2. **flow-matching target 的符号/尺度**。我让 ACE DiT 在 Wan 的 `scheduler.training_target` 下当
   velocity 预测器。若 ACE 训练时的速度参数化符号/尺度不同,`audio_target` 在 train_tmdpp 里**一处**改即可
   (这是 audio loss 不降时的首要排查点)。

3. **ACE latent 帧数 T vs video 帧数 F 的时间对应**。融合 cross-attn 不需要等长(已支持),
   但 beat-motion 对齐质量取决于两条时间轴的物理对应;过拟合阶段先不管,full 训练前要确认采样率/帧率换算。

4. **`scheduler.step` 签名 / `FlowMatchScheduler` 的 import 路径**。train_tmdpp / infer 里按 train_t2av 同源假设,
   按你仓库实际路径改 import,并确认 `step(pred, t, sample)` 的参数顺序。

5. **`load_fusion_checkpoint` 只灌 video 主干**。需要它支持「部分加载 + 缺失 key 不报错」(ACE 权重不在这个 ckpt 里);
   若严格匹配 key,改成 `strict=False` 的 load。

6. **滑动窗口 mask**。`AceDiTRunner._build_masks` 在 `attention_mask=None`(batch=1 无 padding)下走 base 原生 4D mask 逻辑,
   语义与 ACE 原 forward 一致;若 base 默认开 sliding window 且过拟合时想关,在 runner 里把 sliding 传 None 即可。

## 过拟合验收标准(Stage 4.2 sanity)

先 10 条干净样本、480×480、关 CFG(`guidance=1.0`):
- `loss_video`/`loss_audio` 稳定下降,不 NaN;
- 同 prompt + 同 ref 推理结果**逐渐贴近 GT**(你说的三段对比:GT / 独立推理 / 过拟合后推理 → 第三段音画 align);
- 训推一致:训练 step 的 timestep 调度与推理一致,输出无系统性偏移。

通过后再:升 720、扩到高质量子集、按 proposal 进入 full 训练。

## 一句话总结改了什么

把「ACE 整塔跑完再塔级耦合」改成「ACE 拆层、与 video 每层同 step 双向融合」,
并补上训练侧缺失的 image-ref i2v 路径,用单一共用调度保证训推一致 —— 让过拟合时音画对齐真正能涌现。
