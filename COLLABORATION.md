# H3 Studio × SelfLift 协作契约

> 这份文档写给**两边都读**。左边是我负责的 `ComfyUI-H3Studio` fork，右边是另一个
> Agent 负责的 `comfyui-SelfLift` fork。目标是让两边都能放心地大改，同时不把对方的
> 地基刨了。
>
> 适用范围：本机 ComfyUI **0.45.0**，MiniMax H3（`comfy/ldm/minimax/model.py`）。

---

## 1. 分工

| | H3 Studio | SelfLift |
|---|---|---|
| 仓库 | `mr7thing/ComfyUI-H3Studio`（fork of `shootthesound/ComfyUI-H3Studio`） | `mr7thing/comfyui-SelfLift-snapshot`（fork of `facok/comfyui-SelfLift`） |
| 负责 | 本文作者 | 另一个 Agent |
| 角色 | **条件侧**：关键帧/参考/时间线/reel，做"给模型什么" | **采样侧**：两段分辨率渐进采样、TST 注意力校正，做"怎么采" |
| 上游 | `upstream` = shootthesound | `upstream` = facok |

共同依赖的是 **ComfyUI 核心的 H3 实现**，两边都不改核心（ComfyUI-H3Studio 的
row-aug patch 是运行时替换，**不改核心磁盘文件**）。

---

## 2. 为什么现在不打架（机制）

两边都碰到了 `MiniMaxH3Model._forward`，但碰的是**不同的槽位**：

| 槽位 | 类型 | 谁在用 | 能共存？ |
|---|---|---|---|
| `MiniMaxH3Model._forward` | 类属性 | **H3Studio**（整体替换） | — |
| `MiniMaxH3Model._cond_video_rows` / `_cond_audio_rows` | 类属性 | **H3Studio**（整体替换） | — |
| `MiniMaxH3.extra_conds` | 类属性 | **H3Studio**（包装） | — |
| `comfy.ldm.minimax.model.PackedLayout` | 类属性 | **H3Studio**（替换；0.45 下是 no-op） | — |
| `model_options["wrappers"]["diffusion_model"][key]` | **dict → list** | SelfLift TST | ✅ 可多条共存 |
| `model_options["sampler_post_cfg_function"]` | **list** | SelfLift masked_model | ✅ 可多条共存 |
| `model_options["transformer_options"]["optimized_attention_override"]` | **单槽** | SelfLift TST | ⚠️ 见 §5.1 |

关键在核心 `forward`（`comfy/ldm/minimax/model.py:690`）是**调用时**才取 `self._forward`：

```python
graph_out = WrapperExecutor.new_class_executor(
    self._forward, self,
    get_all_wrappers(WrappersMP.DIFFUSION_MODEL, transformer_options)
).execute(x, timestep, context, transformer_options, ...)
```

所以 TST 的 wrapper 套在 **H3Studio 补丁版的外面** —— 两个都生效，而且
**谁先安装、谁后安装都一样**。

已实测（`h3studio_selflift_coexist_check.py`，17 项全过）：

- H3Studio 的 `install()` 成功后，类属性 `_forward` 确实被换掉
- 补丁后的 `_forward` 签名接受 executor 传的一切：`x / timestep / context /
  transformer_options / minimax_payload / denoise_mask / audio_denoise_mask / **kwargs`
- TST 的 wrapper 真的跑到了那个被替换的 `_forward`
- `minimax_payload`、`denoise_mask`、`audio_denoise_mask` 全部原样穿过 wrapper
- 先后顺序不影响结果

---

## 3. 唯一的交汇面：条件字典契约

两边真正共享的数据只有**条件字典**。核心的 `MiniMaxH3AddGuide` /
`MiniMaxH3ReferenceToVideo` 用这些键，H3 Studio 也用这些键，SelfLift 读其中一部分。

### 3.1 顶层键

| 键 | 谁写 | 谁读 | 含义 |
|---|---|---|---|
| `minimax_keyframes` | H3Studio（`h3_studio.py:2020`）、核心 guide | 核心 `extra_conds`、**SelfLift `_resize_keyframes`** | 关键帧列表 |
| `minimax_refs` | H3Studio（`h3_studio.py:1921`）、核心 guide | 核心 `extra_conds` | 参考块列表 |
| `minimax_frame_count` | H3Studio（`h3_studio.py:2021`） | 核心 | 片段帧数 |
| `minimax_prompt_text` | H3Studio（`h3_studio.py:1919`） | H3 Regional Prompt | 原文 prompt |
| `minimax_h3_studio_meta` | （建议新增，见 §7.1） | （建议由对方读取） | 节点侧意图元信息 |

### 3.2 关键帧字典的字段

存在 `minimax_keyframes` 里的每一项：

| 字段 | 一定有？ | 含义 |
|---|---|---|
| `resolved_frame_index` | 是 | 该关键帧锚在第几个像素帧（核心用它算 RoPE 锚点） |
| `latent` | 视觉关键帧有 | 已编码的视觉条件 latent，**5D `[B,C,T,h,w]`** |
| `noise_aug` | 仅 strength < 1.0 **且** row-aug 补丁生效 | `= strength × 0.999`，这一行的调制时间步标签 |
| `audio_latent` | 音频关键帧/声音锚点有 | 已编码的音频条件 |
| `audio_noise_aug` | 同上，strength < 1.0 时 | `= strength × 1.0` |
| `audio_steps` | 声音锚点有 | 音频步数 |
| `h3_audio_end_frame` | motion-context 音频有 | 该音频窗口是**尾端对齐**的（`extra_conds_patch.AUDIO_END_KEY`） |

> ⚠️ **`strength` 本身被 `kf.pop("strength")` 丢掉了**（`h3_studio.py:2012`）。
> 存进条件里的只有派生的 `noise_aug`。对方拿到条件后**无法知道**某个关键帧
> 被弱化过、弱化到多少 —— 除非 row-aug 补丁没生效（那时连 `noise_aug` 都没有）。

### 3.3 核心不变量（双方都不能单方面破坏）

> **关键帧的 `latent` 必须与生成分辨率同网格。**

核心 `PackedLayout` 用**目标网格**算关键帧行数：

```python
# comfy/ldm/minimax/model.py（keyframes 分支）
vt = video_latent.shape[2]
n  = vt * frame_rows          # frame_rows 来自目标 latent 的网格
```

而 `_embed_and_pack` 用布尔掩码把条件行填进布局预留的位置，行数必须严格相等：

```python
all_video_rows[~img_update] = cond_video_rows   # 行数不等 → 直接崩
```

**参考图（`minimax_refs`）不受此限** —— 它们用自己的网格（`_frame_grid(blk["latent_h"], blk["latent_w"])`），与生成分辨率无关。

---

## 4. 我这边可以大改的地方（自由区）

H3 Studio 节点本体（`H3StudioImageToVideo`）及周边，我这边**可以放心大改**：

- **节点的输入项**（新增关键帧类型、新增参考类型、新增 dial）
- **节点的输出项**（目前是 `positive` / LATENT / `v2v_denoise`）
- **时间线编辑器前端**（`web/h3_timeline.js`，460KB，完全自持）
- **reel / 导出 / 音频 lane / fx lane**
- **motion_context、sound_anchors、timed_text、motion_scale/curve** 等实验特性
- **H3 Studio 自己的补丁**（row-aug / extra_conds / middle-frame / time-warp）
- **强度旋钮的实现方式**（现在走 flow-blend + per-row 时间步标签）

唯一的红线是 §5。其余随便改。

---

## 5. 我这边必须守的边界（禁区）

### 5.1 不要写 `optimized_attention_override`

这是核心 `wrap_attn`（`comfy/ldm/modules/attention.py:232`）认的唯一注意力改写入口，
**单槽**：谁后写谁覆盖，前一个静默消失。

SelfLift 的 TST 独占这个槽（`h3_tst.py:223`）。**我这边不碰它。**

> 反过来对对方也一样：如果要再加一个注意力层的改动，**不能各写各的**，
> 必须和 TST 串起来（override 的第一个参数就是下层的 `func`，可以链式套）。
> 这是两边最需要提前对齐的一处。

### 5.2 不要新增/改变 packed segment 的种类或顺序

TST 假设「**video 行是 packed 序列的最后一段**」，并按
`rows_per_frame = ((H+1)//2) × ((W+1)//2)` 定位视频行。

核心 0.45 的段序是 `[text, cond, ref_img, audio, video]`，video 确实最后
（已实测 `frames × rows_per_frame` 与布局的 video 段行数严格相等）。

**如果我新增一种会进入 packed 序列的条件段（新的 kind），必须先跟对方打招呼**，
否则 TST 会定位错行。H3 Studio 现在只在既有的 `cond / cond_audio / ref_img /
ref_audio` 四个 kind 里活动。

### 5.3 不要把关键帧 latent 改成非生成分辨率

见 §3.3。这条同时约束 H3 Studio 和对方。

### 5.4 替换 `_forward` 时必须保留 masked-row 的 union

ComfyUI 0.45 在 `seg_t` 和 `unique_t` 之间插了差分扩散的 masked-row 时间步计算，
而 `rows_to_mod_index()` 会**按行**去 `t_row` 里查这些值。

本 fork 已经修好（见下），**这是 ComfyUI 升级后最容易再次踩坏的地方**。

---

## 6. 已知软肋（目前只能各让一步）

| # | 现象 | 根因 | 现状 |
|---|---|---|---|
| 6.1 | 低分辨率段里，strength < 1.0 的关键帧噪声被抹平 | H3 Studio 先 flow-blend 把噪声混进 latent，SelfLift 的 `_resize_keyframes` 再做双线性 + 均值匹配 | 时间步标签不受影响（标签在 timestep 层不在像素层），所以**强度旋钮仍然精确**，只是噪声纹理被空间平滑 |
| 6.2 | motion_context 的接缝连续性在 SelfLift 流程下打折 | 钉住的尾帧也在 `minimax_keyframes` 里，低分辨率前缀里被一起降分辨率 | 高分辨率段用全分辨率的；接缝可能不如纯 H3 Studio 流程干净 |
| 6.3 | TST 在 `highres_tiling` 下静默失效 | wrapper 按注册顺序跑，TST 坐在 tiling wrapper 外面，分块 attention 触发长度保护 → 跳过 + warning | 对方自己的文档已写明；**开 TST 就得关 highres_tiling** |
| 6.4 | v2v 会被静默退化成满 denoise | H3 Studio 的 `v2v_denoise` 是第三个输出，SelfLift 吃的是 SIGMAS 而非 KSampler 的 denoise | **必须接** `BasicScheduler.denoise` |

---

## 7. 我建议做的调整（具体、可执行）

### 7.1 在关键帧字典里保留 `strength`（我这边做，对方受益）

现在 `strength` 被 pop 掉了。我建议改成保留：

```python
kf["strength"] = s          # 新增：原始强度，0..1
kf["noise_aug"] = s * _VIS_T # 已有：派生的时间步标签
```

这样对方在 `_resize_keyframes` 里就能知道「这个关键帧是弱化的」，
必要时可以在缩放后**按比例重新加噪**，而不是让双线性把噪声抹平 ——
直接解决 §6.1。

**成本**：一个字段，不改任何现有语义（核心忽略未知键）。

### 7.2 加一个 `minimax_h3_studio_meta` 键（我这边做，对方可选读）

用来告诉对方「这个关键帧是什么」，好让他们在低分辨率段区别对待：

```python
meta = {
    "kind": "keyframe" | "motion_context" | "sound_anchor",
    "strength": 0.65,
    "pinned": True/False,      # motion_context 钉住的帧
    "audio": True/False,
}
```

有了它，对方可以对 `pinned` 的帧选择**不降分辨率**（宁可模糊也不要几何漂移），
直接解决 §6.2。

### 7.3 让 H3 Studio 显式声明「关键帧已按生成网格准备」（我这边做）

在条件里带一个布尔（或直接依赖 §3.3 的事实），让对方的 `_resize_keyframes`
可以放心假设「latent 的分辨率 == 生成分辨率，只在低分辨率段才需要缩」。

其实现在已经是这个语义，只是没写下来。写下来 = 双方都不再猜。

### 7.4 对方的两个建议（给对方参考）

1. **改 `_resize_keyframes` 时用 `dict(kf)` 拷贝整条，不要重建字典** ——
   现在代码已经是 `kf = dict(kf)` ✓，这样 `noise_aug` / `strength` /
   `h3_audio_end_frame` 这些字段都会跟着走。**请保持。**
2. **如果要加第二个注意力层改动，和 TST 串接，别覆盖**（见 §5.1）。

---

## 8. 验证协议

两边各改各的，但**都要跑对方的检查**。

| 脚本 | 位置 | 查什么 | 什么时候跑 |
|---|---|---|---|
| `h3studio_api_drift_check.py` | `works/comfy-dev/` | H3 Studio 引用的 30 个核心符号在当前 ComfyUI 里还在不在 | ComfyUI 升级后 |
| `h3studio_patch_compat_check.py` | `works/comfy-dev/` | row-aug 锚点命中没命中；补丁版与原版的时间步表**逐项差分** | ComfyUI 升级后 / 改了 `h3_row_aug_patch.py` 之后 |
| `h3studio_selflift_coexist_check.py` | `works/comfy-dev/` | H3 Studio 的类级 `_forward` 替换 × TST wrapper 叠加，**17 项** | **对方每次改完 SelfLift** |

跑法（**必须屏蔽 GPU**，否则会拉起 CUDA 抢用户正在跑的任务的显存）：

```powershell
$env:CUDA_VISIBLE_DEVICES=''
$py='C:\Users\dream\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Comfyui\ComfyUI\.venv\Scripts\python.exe'
& $py 'C:\Users\dream\Desktop\works\comfy-dev\h3studio_selflift_coexist_check.py'
```

**规矩**：任何一边动了「会影响对方的槽位」，先在对方仓库提一条 issue 或直接跑一遍对方的
检查脚本，把结果贴过去。不要等对方踩到。

---

## 9. 版本与同步

- H3 Studio fork 版本号：`1.6.2+mr7thing.1`（PEP 440 local version，能和上游的
  `1.6.2` 区分开）
- 两个 fork 的 `origin` 指向自己的仓库，`upstream` 指向上游
- 同步上游的流程：

  ```bash
  git checkout master
  git fetch upstream && git rebase upstream/master
  # 改完必跑 §8 的三个脚本
  git push --force-with-lease origin master
  ```

- ComfyUI Desktop 升级后，**先跑检查脚本再重启**，别让补丁静默退化成 fallback
  （fallback 不会报错，只是关键帧强度变软 —— 很难一眼看出来）

---

## 10. 一句话总结

> **H3 Studio 管「给模型什么」，SelfLift 管「怎么采」。**
> 两边唯一的交汇面是条件字典 `minimax_keyframes`，唯一的硬约束是
> 「关键帧 latent 必须与生成分辨率同网格」。
> 除此以外各自随便改，改完跑对方的检查脚本。
