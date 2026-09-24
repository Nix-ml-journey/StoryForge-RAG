# StoryForge-RAG 系统审计报告

**日期:** 2026年9月24日  
**审计员:** Claude Code RAG系统工程师  
**范围:** 数据准备 → Ingest → Retrieval → 提取事实 → 生成 → 评估 → 代理循环

---

## 验证更新（2026-09-24，无 HF 积分期间 · 已在本机验证）

| 项 | 原判定 | 当前状态 |
|----|--------|----------|
| **P0.1** story_json ↔ 重建 ingest | 关键缺陷 | ✅ **已修复 + 本机验证**：`reset_and_ingest.py` 输出 `From story_json: 72 · stale: 0 · no story_json: 0`，72 个文件 / 2138 chunks，`validate_chroma_metadata.py` 报告 `PROBLEMS: none`。（原审计说“manifest 从未被用”是夸大；真正缺口只在重建路径。） |
| **P0.2** 本地 Ollama 提事实 JSON | 关键缺陷 | ✅ **已修复（代码 + 测试）**；⏳ 本机 Ollama 冒烟探针尚未记录结果 |
| **P0.3** 无评估时 facts=0 仍 ACCEPT | 关键缺陷 | ✅ **已修复**（`decide_action` 先查 facts；迭代记录 `has_eval`） |
| **P0.4** 嵌入模型/前缀不匹配检测 | 标为 P0 | **降为 P1，仍未做**（BGE 修复后已整库重建；缺的是集合指纹自动检测） |
| 离线后续（文档 + 校验脚本） | — | ✅ **已验证**：本机 `.\.venv\Scripts\python.exe -m pytest -q` → **163 passed**；`validate_chroma_metadata.py` 已在真实库上跑通 |

### 本机重建后的元数据（`validate_chroma_metadata.py`）

| 字段 | 非空占比（按 chunk） | 解释 |
|------|------------------|------|
| Author | **93.6%** | 35 个经典作品（Kafka / Lovecraft / Frankenstein / Jekyll …）已填作者 = 2002 chunks |
| Display_title | **93.6%** | 同上 |
| Summary | **6.4%** | 恰好相反：只有 37 个 Firestone 故事有 summary（136 chunks / 2138）；经典作品 summary 为空（HF 自动摘要离线不可用） |
| section | **45.8%** | 16 个故事完全没有 section 标签 |

**缺 Author 的 37 个标题不是“作者不详的书”**，而是 **Firestone Idle RPG** 游戏 wiki 的角色 / 故事页（Amun、Anzo、Arvie、Asmondai、Astrid、Belien …）。不需要“去查作者”，用统一约定即可（见下文“Firestone 元数据约定”）。另外这 37 条记录都带着模板占位值：`id: "id_01"`（37 条重复）和 `Is_series: true` 但 `series_name` 为空，建议顺手改掉（检索/评估不读取 `id`，属于元数据卫生，不是 bug）。

### 重建后的 retrieval_eval（必须用 venv 运行）

| | top1 | top3 | fact_coverage |
|---|---|---|---|
| Phase 1 基线（rerank-before-diversity） | 0.80 | 0.90 | 0.77 |
| 重建后（嵌入经人工审阅的 story_json chunks） | **0.80** | **0.833** | **0.733** |

top1 持平；top3 / fact_coverage 小幅下降，原因是重建后嵌入的是 story_json 中审阅过的 chunk（切分和文本与旧 `.txt` 重新切块不同）。**这是预期内的变化，不是紧急回归**；如果之后要追回 top3，属于 corpus / chunk 质量工作（逐案看 `Evaluation/retrieval_eval_report.json` 中哪 1 个 case 从 top3 掉出），不是调检索旋钮。

### 运维提示（不是 P0）
- **一定用 venv 跑脚本**：`.\.venv\Scripts\python.exe scripts\...`（或先激活 `.venv`）。裸 `py scripts\retrieval_eval.py` 用的是系统 Python，会报 `ModuleNotFoundError: langchain_chroma`。本文后面历史部分里的 `py scripts/...` 命令都应按此替换。
- `reset_and_ingest.py` 删除 `data/chroma_db` 时可能报 **WinError 32（文件被占用）**——通常是 API 服务或其它 Python 进程还开着 Chroma。脚本会改用 Chroma API 重置集合，重建照样成功；想彻底删目录就先停掉 `python main.py` 再跑。
- `push_section_metadata.py` 只刷新 `section` / `meta_json`，**不会**刷新顶层 `Author` / `Display_title` / `Summary`。改了作者/标题后请用 `reset_and_ingest.py` 重建。

### Firestone 元数据约定（可选，离线即可做）
- `meta.author` → **`"Firestone Idle"`**（所有 Firestone 页面用同一个字符串）
- `meta.title`（→ Chroma 的 `Display_title`）→ 角色/页面名，一般等于文件名（`Amun`、`Anzo` …）
- `id` → 文件名（替换模板值 `id_01`）；`Is_series` → `false`（角色页不是某个命名系列的章节）
- `summary` → 已有，保留；以后想改再改，**不阻塞**
- 只改 meta 不改 `raw_text` / `chunks`，所以重建时不会被判为 stale
- 具体可复制的步骤见 `P0_CRITICAL_ISSUES_CHECKLIST.md` → “Firestone metadata recipe”。完成后预期：Author / Display_title → **100%**，Summary 仍为 6.4%。

### 现在该做什么

**已完成：** P0.1–P0.3 代码与测试 · 重建 ingest（72/0/0）· `validate_chroma_metadata.py` 报告 · retrieval_eval（venv）

**现在可选（离线）：**
1. 按上面约定补齐 37 个 Firestone 的 `meta.author` / `meta.title`（+ `id` / `Is_series`），再 `reset_and_ingest.py` + `validate_chroma_metadata.py`
2. 跑一次 P0.2 的本地 Ollama 提事实冒烟探针（`Grounded_facts_provider: "local"`），记录 `status=ok kept=N`
3. section 覆盖率 45.8% → **P2 可选**（16 个无标签故事可手工补，或等 enrich）

**仍等 HF 积分：**
- `measure_generation_length.py` 真实测量（按 `has_eval` 拆分 accept rate）
- 经典作品的 Summary 自动 enrich（或离线保持为空——不影响检索和生成）
- HF vs 本地提事实数量/质量对比

**仍未做的 P1：** 嵌入指纹检测（原 P0.4）

详细勾选清单见同目录 `P0_CRITICAL_ISSUES_CHECKLIST.md`。

---

## 一句话诊断

**（审计当时）** Phase 2 中后期：检索稳定，但 grounding 在无评估回退下可被绕过，本地提事实脆弱，重建 ingest 丢掉 story_json 元数据。  
**（本机验证后）** P0.1–P0.3 已落地并通过 163 个测试；重建 ingest 已在真实语料上验证（72/0/0，Author 93.6%）。剩下的是可选的元数据补齐（Firestone 约定）、P0.2 冒烟探针、P0.4 指纹检测，以及下月 HF 复测——都不是 P0。

---

## 已经做得好的地方

1. **设计清晰的 3 步 RAG 管道** — 检索 → 提取有根据的事实（JSON+chunk_id）→ 生成。单向流程，不像多层会丢失事实。

2. **统一的长度目标系统** — 一个 `LengthProfile` 驱动提示、token 预算、接受门槛三个必须一致的东西。解决了"为什么即使 token 预算充足也只输出 450 字"的根本问题。

3. **混合检索 + 重排 + 多样性** — Hybrid BM25+dense RRF 融合 → 跨编码器重排 → 多标题选择。设计合理，top1/top3 指标已稳定（0.80 / 0.90）。

4. **思考模式恢复** — 空草稿重试逻辑：thinking 模式若返回空，自动重试 fast 模式，否则抛出清晰错误。length guard 和代理循环都能优雅回退。

5. **HF API 第一，本地回退** — Step 2 和 Step 3 隔离 VRAM 竞争。Step 2 用 HF API（无本地 VRAM），Step 3 用 Ollama。设计得当。

6. **两个生成后端选项** — Ollama（默认，简单） + vLLM（可并发）。配置驱动，Transformers 也支持。

7. **配置驱动的评估链** — HF → Gemini → 本地 Transformers。对 HF 故障的适应，虽然有缺陷（见下文）。

8. **Qwen3 思考令牌问题已修复** — `/no_think` 后缀 + token 预算升到 3200。诊断过程清晰（日志显示 finish_reason="length"）。

9. **完整的 ingest 脚本套件** — reset、refresh、push metadata 等操作都有脚本。

10. **测试覆盖** — 长度目标、提示合约、代理决策都有单元测试。无 GPU 运行。

---

## 数据与 Ingest 专项审查

> **（2026-09-24 本机验证后）** 本节是**审计当时**的状态，保留作历史记录。现在 `reset_and_ingest.py` 已读取 story_json（本机 72/0/0，Author 93.6%），下文中“ingest 不读 story_json”“Author 应为 0%”“`ingest_manifest.py` 不使用 manifest”等描述已**不再成立**。当前流程见 `docs/DATA_PREP.md`；命令请一律用 `.\.venv\Scripts\python.exe scripts\...`。

### 现在的流程对不对？

**不完全对。文档与代码不同步。**

#### 文档说的流程（DATA_PREP.md）：
```
raw_extracted/*.txt
  ↓ 手动清理 + 分割
stories/*.txt
  ↓ step1_prepare_and_enrich.py
story_json/*.json（包含 author, title, summary, chunks, section 标签）
  ↓ records_to_ingest_manifest.py
ingest_manifest.jsonl
  ↓ ingest_manifest.py
Chroma
```

#### 代码实际做的（src/storyforge/vector_store/ingest_stories.py）：
```
直接从 stories/*.txt 读取
  ↓ _chunk_text()
chunks（无来自 story_json 的元数据）
  ↓ _embed_chunks()
Chroma：
  - Title = 文件名（无后缀）
  - Author = ""（空）
  - Summary = ""（空）
  - metadata 只有这三个 + chunk_id
```

**关键发现：**
- `ingest_stories.py` 不读取 `story_json/` 文件夹
- 不使用 `ingest_manifest.jsonl` 
- `step1_prepare_and_enrich.py` 和 `records_to_ingest_manifest.py` 构建的 `story_json/*.json` **从不进入 ingest 流程**
- Author / Summary / section 标签全部被忽略，Chroma 中默认为空字符串

**这意味着：**
1. `step1_prepare_and_enrich.py` 的所有丰富化工作（HF 摘要、Ollama section 标签）都白做了
2. 评估报告中 `fact_coverage` 指标依赖 `Title` 元数据，但如果标题在 JSON 中被手动修改，Chroma 中仍是文件名 — 不同步
3. Series / volume / chapter 元数据永远无法进入向量库
4. 用户手动调查"为什么没有检索到"时，看到的是 Author="" Summary="" — 没有线索是因为元数据从未写入

#### 数据进入数据库前必须完成的处理步骤（按顺序）

**步骤 0：原始文本准备**
- 从 PDF/EPUB 提取（已完成，输出 `data/raw_extracted/*.txt`）
- 删除古腾堡标题、TOC、页码、跑线头 ✓（必须手动）
- 检查：首段是故事开头，不是"CONTENTS"；名字搜索找到情节句子，不是索引

**步骤 1：按故事分割 + 命名**（手动）
```
一个文件 = 一个故事
filename = Author__Story_Name.txt
例：Lovecraft__The_Call_of_Cthulhu.txt
```
放入 `data/stories/` 下。

**步骤 2：生成 JSON 记录**（脚本 + 手动审查）
```
py scripts/step1_prepare_and_enrich.py
# 输出 story_json/*.json
# 审查 / 手动修正：
#   - meta.author / meta.title（必填）
#   - summary（重写如果模型编造情节）
#   - chunks[].section 标签（如果标签都相同，手动修复）
#   - 删除空块
#   - 检查 Is_series 标志
```

**步骤 3：构建 manifest**（可选，现在无用）
```
py scripts/records_to_ingest_manifest.py
# 输出 ingest/ingest_manifest.jsonl
# （但 ingest_stories.py 不使用这个）
```

**步骤 4：Ingest**
```
py scripts/reset_and_ingest.py  # 或 ingest_manifest.py
# 读取 stories/*.txt（忽略 story_json/*.json 和 manifest.jsonl）
# 输出到 Chroma
```

**当前的关键缺陷：**
- 步骤 2 中手工填写的 author/title/summary 永远进不了 Chroma
- 不存在"将 story_json 字段同步到 Chroma 元数据"的脚本
- `ingest_stories_dir()` 看起来是通用的，但硬编码为"直接从 .txt 读取"
- 没有验证步骤：ingest 后无法检查元数据是否符合预期

### 常见错误 / 最可能踩的坑

| 坑 | 症状 | 根因 |
|----|------|------|
| 使用旧 BGE 向量 | 新查询无关 / 检索混乱 | 2026-09 修复后未运行 `reset_and_ingest.py`；旧/新向量混在一个集合里 |
| Title 元数据为空 | 评估报告 `fact_coverage=0`；用户没看到故事名 | `story_json/*.json` 中 title 修改后未重新 ingest；ingest 只用文件名 |
| Author/Summary 总是空 | 没有视觉确认元数据进入了数据库 | **设计缺陷**：ingest 不读取 story_json |
| ingest_manifest.jsonl 不存在 | 脚本失败或警告 | records_to_ingest_manifest.py 输出到该路径，但 ingest_manifest.py 不验证它存在（脚本走查发现通常工作，但可能丢失） |
| Section 标签都一样（如全是"setup"） | 评估脚本无法按 section 分类 | enrichment 标签错误；无法纠正（ingest 会忽略 story_json）；需手动编辑 .txt 或 story_json 然后重新运行全流程 |
| Chroma_path 指向错误目录 | 新 chunks 进入 A 目录，查询使用 B 目录 | BASE_PATH 在 setup.yaml 中忘记更新，或两个脚本分别读默认值 — 这里存在名为"默认必须一致"的注释但没有断言 |
| 新文件没进 Chroma | 查询不到新故事 | 文件名不是 .txt / 在 stories/ 目录外 / ingest 失败但没有记录（见下文"验证后 ingest"） |

### 推荐的 ingest / re-ingest 操作清单

#### 首次 ingest（新语料库）
```bash
# 1. 准备 stories/ 目录下的 .txt 文件
#    - 一个故事 = 一个文件
#    - 文件名格式：Author__Story_Name.txt

# 2. 可选：生成 JSON 记录（当前忽略，但有文档价值）
py scripts/step1_prepare_and_enrich.py
# → story_json/ 用于视觉审查、手动修正等

# 3. **关键：检查 stories/ 目录中有多少 .txt 文件**
ls -la data/stories/*.txt | wc -l

# 4. Ingest
py scripts/reset_and_ingest.py

# 5. **验证后 ingest（新增）——这一步现在缺失**
py scripts/peek_vector_store.py
# 显示前 5 个 chunk：检查 Title、Author、text 看起来对吗？
# 快速查询一个只有这个故事有的名字
curl -X POST http://localhost:8000/vector_store/query \
  -H "Content-Type: application/json" \
  -d '{"query":"<unique_character_or_event>"}'
# Top result 应该是预期的故事
```

#### 修改现有故事的文本
```bash
# 改过 data/stories/<Title>.txt 中的段落后：
py scripts/refresh_chunk_embeddings.py --glob "Author__*"
# 重新嵌入那些 chunks；保持 chunk_ids 和 metadata 不变
```

#### 仅更新 section 标签（不改文本）
```bash
# 编辑 story_json/<Title>.json，修改 chunks[].section 后：
py scripts/push_section_metadata.py --glob "Author__*"
# 只更新元数据（section 标签）；不重新嵌入
# **注意：此时 Author/Summary 改变不会同步**（设计缺陷）
```

#### 更改嵌入模型
```bash
# 修改 setup.yaml: Vector_store_model = "new-model"
py scripts/reset_and_ingest.py
# 全部重新生成
```

#### 完全清空 + 重新开始
```bash
rm -rf data/chroma_db/
py scripts/reset_and_ingest.py
```

#### **缺失的检查清单（建议添加）**
```bash
# ingest 后验证元数据是否符合预期（新脚本）
py scripts/validate_chroma_metadata.py
# 应输出：
#   - 总 chunks 数
#   - 不同标题数
#   - Author 非空的 chunks 占比（审计当时 0%；本机重建后 93.6%）
#   - 任何名为"Unknown"的标题
#   - 摘要非空的 chunks 占比（审计当时 0%；本机重建后 6.4%，只有 Firestone 故事有 summary）
# ✅ 已实现：scripts/validate_chroma_metadata.py（本机已运行，PROBLEMS: none）
```

---

## 全栈问题清单（按优先级 P0/P1/P2）

### P0 问题（审计当时为关键缺陷；验证状态如下）

#### P0.1: story_json 工作流与 ingest 脱离 — ✅ 已修复（2026-09-24）

**审计纠正：** `ingest_manifest.py` 原本就能吃 manifest；真正缺口是 `reset_and_ingest.py` → `ingest_stories_dir()` 硬编码空 Author/Summary。

**现状：** `ingest_stories_dir()` 会读 `story_json`；raw_text 一致时用已审 chunks/section；过期则用 .txt chunks 并保留故事级元数据；Title 仍为文件名 stem（`Display_title` 存可读标题）。测试：`test_ingest_story_json_metadata.py`。

**你还需做：** 填好 Author/Summary 后跑 `py scripts/reset_and_ingest.py`，再 `peek` / `retrieval_eval`。

#### ~~原问题描述（历史）~~ story_json 与 rebuild ingest 脱离

**症状（修复前）：**
- `step1_prepare_and_enrich.py` 花时间生成 author/title/summary，重建 ingest 后进不了 Chroma
- 用户编辑 `story_json/*.json` 手动修正元数据，但 `reset_and_ingest` 后还是空
- `Title` 元数据只能是文件名；人类可读标题在 story_json 里无人写入向量库

**根因（修复前）：**
- `ingest_stories_dir()` 硬编码读 `.txt` 并写 Author="" / Summary=""
- manifest 路径可用，但文档推荐的全量重建脚本不走它

**证据来自代码：**
```python
# ingest_stories.py::ingest_stories_dir() 第 176-260 行
def ingest_stories_dir(...):
    # 行 194: files = list(_iter_story_files(stories_path))
    # → 只列出 .txt 文件，不读 manifest
    # 行 230: title = f.stem  # 文件名作标题
    # 行 234-242: metadata 硬编码为 {"Title": title, "Author": "", "Summary": "", ...}
    # 从不读取 story_json/<Title>.json 或 manifest 中的值
```

**推荐修复：**
1. 改 `ingest_stories.py::_iter_story_json_records()` 读 `story_json/*.json` 或 `ingest_manifest.jsonl`
2. 或，在 `scripts/ingest_manifest.py` 中补充实现（不是函数，是脚本）以读取 manifest 并调用已修复的 ingest
3. 添加验证：ingest 后 `Title` 应该匹配 `story_json/*.json`，若不匹配则警告
4. 添加新脚本：`sync_metadata_to_chroma.py` — 从 `story_json/` 批量更新已有 collection 的元数据（不重新嵌入）

**验证方法：**
```bash
# 修复前（当前）
grep '"Author"' data/chroma_db/*/chroma.parquet  # 应该全空
# 修复后（目标）
# 某些 chunks 的 Author 应该非空，匹配 story_json 中的 meta.author
```

#### P0.2: 本地 Ollama 回退的 JSON 解析脆弱 — ✅ 已修复（代码，2026-09-24）

**现状：** Ollama schema/JSON format、`salvage_grounded_facts_json`、引用校验、一次压缩重试、`Grounded_facts_provider: "local"`。测试：`test_local_facts_fallback.py`。  
**你还需做：** 本机 Ollama 冒烟（见 checklist）。

#### ~~原问题描述（历史）~~

**症状（修复前）：**
- HF API 返回 402（积分用完）或 502（故障）→ 代码回退到本地 Ollama  
- 本地回退产生格式不正确的 JSON（缺少引号、不完整对象）
- `parse_grounded_facts_json()` 失败 → `facts_count = 0` → 代理循环强制 RE_RETRIEVE
- accept_rate 崩溃

**根因（修复前）：**
- HF `InferenceClient.chat_completion()` 支持 `response_format={"type": "json_object"}` — 强制有效 JSON
- 本地 Ollama 回退不使用该选项，模型可自由格式化（通常有 bug）
- `repair_json()` 函数存在但只尝试简单修复，处理不了严重的格式错误

**证据：**
```python
# extraction.py::extract_grounded_facts() 第 257-308 行
# HF 路径（第 265-272 行）：成功使用 _hf_chat_extract_json_with_retry()
# 回退路径（第 273-283 行）：
grounded_raw = str(facts_llm.invoke(facts_prompt) or "").strip()
# → facts_llm 是 Ollama/vLLM/Transformers，无 response_format
# 结果直接进 parse_grounded_facts_json()（第 286 行），失败概率高

# PROJECT_JOURNEY.md 第 442-456 行：
# "the local extraction fallback then failed its own way: 
# Grounded-facts JSON parse failed at small, early character offsets"
```

**推荐修复：**
1. 给 Ollama/vLLM 生成的回退结果添加 JSON repair：`repair_json()` 调用或正则提取
2. 或，在提示中添加"**output only valid JSON**"示意
3. 或，添加一个"结构化生成"后端（e.g. outlines 库的 grammar-constrained）
4. 最小化：添加日志显示回退路径被触发（当前只在 e.g. 一开始记一次），每次提取失败时重复日志

**验证方法：**
```bash
# 现在
py scripts/debug_hf_grounded_facts_mode.py --show-raw --use-local-fallback
# 观察：大多数"success: true"其实有 JSON 错误（仔细看）

# 修复后
# 相同命令应该显示有效 JSON，或清晰的"parse failed, falling back to empty facts"
```

#### P0.3: 无评估回退路径接受 facts_count=0 — ✅ 已修复（2026-09-24）

**现状：** 无评估时先查 `facts_count <= 0` → `RE_RETRIEVE`；有事实且完整才 ACCEPT；迭代带 `has_eval`；耗尽且无事实 → `max_iterations_no_grounded_facts`。测试在 `test_agentic_loop.py`。

#### ~~原问题描述（历史）~~

**症状（修复前）：**
- HF 评估模型不可用（停机、积分用完、API 错误）→ `has_eval=False`
- `decide_action()` 不检查 `facts_count > 0`，仅检查 completeness
- 代理循环可返回 `accepted: true` 但零事实 → 破坏 grounding 承诺

**根因（修复前）：**
```python
# PROJECT_JOURNEY.md：no-eval 路径仅按 completeness ACCEPT，不要求 facts_count > 0
```

**推荐修复：**
1. `decide_action()` 的无评估路径添加一行：`if facts_count <= 0: return DECISION.RE_RETRIEVE`
2. 或，至少在返回 ACCEPT 时设置一个标志/日志说"accepted without eval, facts may be missing"
3. 测试：mock 掉评估提供者，验证 `facts_count=0` 总是导致 RE_RETRIEVE（不是 ACCEPT）

**验证方法：**
```bash
# 现在（有缺陷）
HF_API_KEY="" py scripts/measure_generation_length.py --mode fast --length long
# 输出 reason 应该是 "accepted without evaluation"，但如果 facts_count=0，不应该被接受

# 修复后
# 同样命令，任何 facts_count=0 的迭代应该 RE_RETRIEVE，不是接受
```

#### P0.4: 嵌入模型不匹配风险 — ⏳ 仍开放（降为 P1）

**现状：** BGE query-only 前缀已修且要求 re-ingest；缺的是 collection 指纹 / 自动检测。不算当前 live P0（若你已按 2026-09 要求重建过库）。

**症状：**
- 用户从旧版本升级，Chroma DB 中有旧向量（带前缀错误）
- 新查询用新嵌入器（无前缀），与旧向量不匹配  
- 检索质量下降，用户困惑
- 没有自动检测

**根因：**
```python
# docs/DATA_PREP.md 第 211-231 行记录了 BGE 前缀修复
# ingest_stories.py 第 42-63 行的注释说明
# 但 ingest_stories_dir() 不检查现有集合的嵌入模型 vs. 配置模型
# 也不警告向量可能不匹配
```

**推荐修复：**
1. `get_or_create_collection()` / `_build_vectorstore()` 添加验证：  
   如果集合已存在，检查其嵌入模型是否与配置匹配（元数据或计算验证）  
   如果不匹配，抛出或警告
2. 或，至少在日志中显示："ingesting with model X into collection with cached model Y"

**验证方法：**
- 在 Chroma collection 中存储嵌入模型名称作为元数据（第一次 ingest 时）
- 重新 ingest 时验证一致性

---

### P1 问题（严重缺陷，影响质量/可观测性）

#### P1.1: Accept rate 测量不可靠（HF 积分掩盖缺陷）

**症状：**
- `scripts/measure_generation_length.py` 显示 accept_rate=0.75
- 但用户检查日志发现大多数迭代是 "completeness OK (no eval provider)"
- 评估时 HF 402 → 代理循环用无评估回退路径
- 无法区分：真正的"接受（有评估）" vs "回退接受（无评估+facts_count可能为0）"

**根因：**
- HF Inference Provider 积分限制（每月配额）
- 代理循环检测到 API 错误 → 使用无评估回退
- 无评估回退对 facts_count 无要求
- 测试套件/用户无法区分两条路径

**证据来自：**
```
docs/PROJECT_JOURNEY.md 第 406-450 行
"expected behavior given the eval outage, not itself a defect, but it means 
this run measured length/accept behavior under the no-eval fallback path, 
not the scored path"
```

**推荐修复：**
1. 保存每个迭代的 `has_real_eval: bool` 标志
2. 报告时分别计算：accept_rate（有评估） vs. accept_rate（无评估回退）
3. 文档：如果 accept_rate（有评估） < 0.60，可能是事实提取问题（见 P0.2）
4. 积分检查脚本：`py scripts/check_hf_credits.py` 显示剩余配额并给出是否足够运行完整测试的建议

#### P1.2: 无 Ingest 验证步骤（无法验证元数据是否进入）

**症状：**
- 用户 ingest 后，无法确认 chunks 是否有正确的元数据
- 想调试"为什么没有检索到"时，不知道是检索问题还是 ingest 问题
- `peek_vector_store.py` 存在但很基础（只显示 5 个 chunk 的文本）
- 没有元数据完整性检查

**根因：**
- Chroma 暴露的 API 可以查询 documents，但不方便验证 metadata
- 没有脚本汇总并报告元数据统计

**推荐修复：**
```python
# 新脚本：scripts/validate_chroma_metadata.py
# 输出：
#   - Total chunks: N
#   - Unique titles: M
#   - Non-empty Author: X / N (%)
#   - Non-empty Summary: Y / N (%)
#   - Non-empty section tags: Z / N (%)
#   - Titles == filenames? (yes/no)
# 建议：
#   - 如果 Author% < 50%, 提示："考虑从 story_json 同步元数据（P0.1 缺失功能）"
```

#### P1.3: Hybrid_bm25_weight 是 no-op 但易被误调整

**症状：**
- 用户读到文档"可调整 Hybrid_bm25_weight 以改进 top-1"
- 用户改 0.3 → 0.5 → 0.1，重新 ingest/测试，看不到差异
- 用户困惑，认为混合检索功能坏了
- 实际上权重被重排器完全覆盖

**根因：**
```python
# retrieval.py 第 286-300 行说得很清楚（代码注释）
# docs/setup.example.yaml 第 146-153 行也说了
# 但配置名字暗示它"可调整"
```

**推荐修复：**
1. `setup.example.yaml` 和代码注释更明显：添加 **粗体** 或 ⚠️ 标记
2. 或，改配置名字为 `Hybrid_bm25_weight_no_op_when_reranking`（太长了，但清楚）
3. 或，在 `retrieve_docs()` 中添加日志：当重排启用时每次都输出 "Reranking enabled: bm25_weight is ignored"

#### P1.4: Streaming 路径跳过 attribution gate（设计缺陷）

**症状：**
- `/orchestration/generate_stream` 端点流式返回 token  
- attribution gate（检查未授权的名字/地点）被跳过
- 用户得到可能有编造的文本
- 文档说"流式跳过 attribution... [SECTION 1]等格式工件（由设计）"但不清楚这是功能还是已知缺陷

**根因：**
- Streaming 必须尽快返回第一个 token，无法等待完整生成后再验证
- Attribution gate 需要完整生成的文本
- 权衡：吞吐量 vs. 安全性

**推荐修复：**
1. 文档明确说"Streaming 不进行 attribution 检查；生产环境应使用非 streaming 路径或添加后处理 attribution"
2. 或，实现"后处理 attribution"：stream 完成后在后台验证并记录违规（不截断文本，但告知用户）
3. 测试：添加单元测试验证流式路径 + 非流式路径生成不同结果当有 attribution 违规时

---

### P2 问题（改进机会，不急迫）

#### P2.1: 元数据一致性（story_json 字段未同步到 Chroma）

**症状：**
- 用户在 `story_json/*.json` 编辑 `Is_series=true` / `series="Grimm Tales"` 等
- Ingest 时被忽略，Chroma 中永远 `Is_series=false`
- 评估脚本 / 过滤查询时用不上这些信息

**根因：**
- Ingest 硬编码为只从 .txt 文件名提取 Title，其他都是默认值

**推荐修复：**
- （取决于 P0.1 修复）若 ingest 改为读 story_json，则一併同步 `Is_series`, `series`, `volume`, `chapter`, `section` 标签

#### P2.2: Re-ingest 脆弱性（Chroma_path / BASE_PATH 配置不一致）

**症状：**
- 用户运行 ingest 脚本多次，偶尔新 chunks 没进去
- 或，查询时新数据找不到
- 根因是两个脚本分别读默认值，DEFAULT 不同（虽然代码注释说"must match"）

**根因：**
```python
# ingest_stories.py 第 188-190 行：
# stories_path = Path(stories_dir) if stories_dir else root / (cfg.get("Story_input") or "data/stories")

# retrieval.py 第 89 行：
# chroma_dir = (base / (cfg.get("Chroma_path") or "data/chroma_db")).resolve()

# 若 cfg 不完整，两个 or 子句可能都生效，但返回的"默认"来自不同的代码位置
```

**推荐修复：**
1. 单一的 DEFAULT 常量文件
2. 或，config.py 中声明所有默认值一次
3. 添加启动检查：确认 BASE_PATH / Chroma_path / Story_input 三者一致

#### P2.3: 没有健康检查端点

**症状：**
- 生成请求开始后，若 Chroma 离线 / Ollama 不响应，请求中途失败
- 用户无法在生成前预检整个系统
- 没有 `/health` 或 `/readiness` 端点

**推荐修复：**
```python
# POST /health
# 返回：
# {
#   "chroma_ok": true/false,
#   "ollama_ok": true/false,
#   "hf_token_set": true/false,
#   "embedding_model_ready": true/false,
#   "message": "..."
# }
```

#### P2.4: 配置值类型转换不一致

**症状：**
- 某些 bool 配置用 `str(...).lower() not in ("false", "0", "no", "")`
- 某些用 `bool(...)`
- 易于混淆

**推荐修复：**
- 集中化配置加载：定义 schema（Pydantic），自动转换类型
- 或至少统一一个辅助函数 `_cfg_bool()`, `_cfg_int()` 等

---

## 不该现在做的事（反模式 / 过早升级）

### ❌ 不要现在就升到 14B 或更大的生成模型

**为什么：**
- 项目约束："do not jump to a bigger generation model until facts-extraction reliability is confirmed"
- 当前状态：facts_count=0 频率未知（HF 积分掩盖），accept_rate 不可靠
- 14B 会消耗 8GB+ VRAM，剩余空间压低其他模型性能
- 问题不是模型大小，是事实可靠性和长度控制

**等待条件：**
- 运行 `py scripts/measure_generation_length.py` 3 次，HF 积分充足（无 402 错误）
- 查看日志：`facts_count > 0` 在 90% 以上迭代
- Accept rate（真实评估）≥ 0.70
- THEN 考虑升级

### ❌ 不要添加更多重排或混合检索旋钮

**为什么：**
- Phase 1 检索调整已停止（top1=0.80 / top3=0.90 稳定）
- 剩余 6 个 top1 miss 中，4 个是"在池中但排名低"（评估案例分析），1 个是真嵌入碰撞
- 再调 `Hybrid_bm25_weight` / `Story_generation_rerank_top_n` 不会帮助
- 所需的修复是"检索池扩大"或"query reformulation"或"corpus 质量"，不是参数调整

### ❌ 不要承诺生产等级的可用性

**当前状态：**
- 本地 16GB GPU 上可工作
- 单用户 / 单生成流
- HF API 故障风险（停机、积分限制）、Ollama 回退脆弱、元数据工作流分裂

**生产需要：**
- 队列系统（后台作业）
- 持久数据库（metadata、生成历史）
- 对象存储（books、输出）
- 完整可观测性（追踪请求、模型版本、长度目标指标）
- 更可靠的回退路径

### ❌ 不要修改 prompts.yaml 中的句子/word 目标而不同时更新 length_profile 预设

**为什么：**
- 提示中的"prefer N-M sentences per section"必须与 `LengthProfile.min_sentences_per_section` 一致
- 否则接受门槛 vs. 模型预期会不同步，重现"为什么 token 充足还是短输出"的问题

---

## 30 天行动计划（周度计划，一人 + 16GB GPU）

### 第 1 周：诊断和数据修复

**优先级：P0 问题理解 + ingest 验证脚本**

- **Day 1-2：** 
  - 运行 `py scripts/measure_generation_length.py --mode fast --length long` 一次（10 个查询）
  - 检查日志中的 `facts_count`, `finish_reason`, `has_eval`, `reasons`
  - 记录输出到 `Evaluation/week1_baseline.log`
  - 目标：确认目前 accept_rate 的真实水平和失败原因

- **Day 3：**
  - 新建脚本 `scripts/validate_chroma_metadata.py`（60 行代码）
  - 输出：chunks 总数、Title/Author/Summary 非空占比、任何异常
  - 运行一次，存结果到 CSV

- **Day 4-5：**
  - 代码审查 `ingest_stories.py` vs. `story_json/` / `ingest_manifest.py`
  - 编写文档（不改代码）：现状 + 为什么脱离 + 修复设计
  - 存档为 `docs/INGEST_WORKFLOW_ANALYSIS.md`

- **Day 6-7：**
  - 写测试：`tests/test_ingest_metadata.py`
    - Ingest 一个故事，验证 Title == filename（当前预期）
    - 验证 Author="" （当前预期）
    - 预留接口以供修复后测试 Author != ""
  - 提交：commit message "Test suite: add ingest metadata validation (pre-fix baseline)"

---

### 第 2 周：P0.2 修复（本地回退 JSON 安全性）

**优先级：防止 facts_count=0 崩溃**

- **Day 8-9：**
  - 复用现有 `repair_json()` 或写改进版本（在 `attribution.py`）
  - 目标：给本地 Ollama/vLLM 输出添加一层安全网
  - 单元测试 5 个格式错误的 JSON 样本

- **Day 10：**
  - 改 `extract_grounded_facts()` 的回退路径
  - HF 失败 → Ollama → 修复 JSON → 解析
  - 添加日志：显示修复尝试（成功/失败）

- **Day 11-12：**
  - 运行 `measure_generation_length.py` 再一次，同样条件（HF 需要可用积分）
  - 目标：确认 JSON 修复后 facts_count 分布改善
  - 对比 week 1 日志

- **Day 13-14：**
  - 测试 + 反馈循环
  - Commit 两个：1）修复代码 2）测试
  - PR 描述：包含 week1 vs. week2 的 facts_count 对比

---

### 第 3 周：P0.3 修复（无评估回退安全性）+ P0.1 规划

**优先级：确保 grounding 承诺**

- **Day 15：**
  - 改 `decide_action()` 或 `agentic_loop.py`
  - 无评估路径：若 `facts_count <= 0`，返回 RE_RETRIEVE（不是 ACCEPT）
  - 添加日志：标记"无评估回退"迭代，便于测量

- **Day 16：**
  - 单元测试：`tests/test_agentic_loop.py` 添加
    - 模拟无评估 + facts_count=0
    - 验证决策是 RE_RETRIEVE
  - 运行全量单元测试，确保无回归

- **Day 17-18：**
  - 开始 P0.1（ingest workflow）的设计
  - 代码审查：`ingest_stories.py` 如何改以读取 `story_json/` 或 `ingest_manifest.jsonl`
  - 选择：
    - 选项 A：ingest 读 story_json（改 `ingest_stories_dir()` 签名）
    - 选项 B：新脚本 `ingest_from_manifest.py`（较小改动）
  - 写设计文档到 `docs/INGEST_WORKFLOW_REDESIGN.md`

- **Day 19-21：**
  - 实现选项 A 或 B（代码）
  - 单元测试：ingest 后验证 Title/Author/Summary 来自 story_json
  - 集成测试：用样本数据（sample/ 下）测试整个流程
  - Commit（小的、可审查的改动），PR 描述指向设计文档

---

### 第 4 周：验证 + 文档 + 可观测性

**优先级：确保修复有效 + 用户可调试**

- **Day 22：**
  - 新脚本 `scripts/validate_chroma_metadata.py` 改进
    - 现在应该显示 Author % > 0（修复后）
    - Summary % > 0（如果 ingest 从 story_json 读取）

- **Day 23-24：**
  - 添加健康检查端点（P2.3）— 快速修复
    - `/health` → 检查 Chroma / Ollama / HF token / embedding 模型
    - 单元测试

- **Day 25-26：**
  - 更新 `docs/DATA_PREP.md`
    - 现在的流程（修复后）：story_json 字段会进 Chroma
    - 新的验证步骤
    - 新的元数据同步脚本（若有的话）

- **Day 27-28：**
  - 运行最后一次 `measure_generation_length.py`（10 个查询，HF 需充足）
  - 对比 week 1 / week 2 / week 4：facts_count / accept_rate / JSON 解析失败率
  - 生成最终报告 `Evaluation/week4_final.log` + 总结文档

- **Day 29-30：**
  - Code review + polish
  - 一个综合 PR：P0.2 + P0.3 + P0.1（可分成多个，但一周内全部 merge）
  - 更新 `docs/README.md` "Current status" 部分，记录修复日期
  - 标记版本 v0.2（或 tag）

**最终检验清单：**
- [ ] `facts_count=0` 在 < 5% 迭代中出现
- [ ] Accept_rate（真实评估）≥ 0.70（目标 0.75+）
- [ ] Chroma 中 Author % > 50 且 == story_json meta.author
- [ ] `/health` 端点返回系统状态
- [ ] `measure_generation_length.py` 日志清晰：`has_eval`, `facts_count`, `finish_reason`
- [ ] 所有新测试通过；无回归

---

## English Summary

**Verification (2026-09-24, on the owner's machine, no HF credits):** P0.1–P0.3 implemented; `.venv` pytest **163 passed**. **P0.1 verified live:** `reset_and_ingest.py` → story_json 72 / stale 0 / none 0, 2138 chunks, `validate_chroma_metadata.py` → `PROBLEMS: none`, Author 93.6%, Display_title 93.6%, Summary 6.4%, section 45.8%. retrieval_eval (venv) after the rebuild: top1 0.80 / top3 0.833 / fact_coverage 0.733 vs Phase 1 0.80 / 0.90 / 0.77 — an expected shift from embedding the reviewed story_json chunks, not a regression emergency. P0.4 is now P1 (no embed fingerprint yet).

**Metadata gaps are content, not code:** the 37 stories without Author/Display_title are Firestone Idle RPG wiki pages → set `meta.author: "Firestone Idle"`, `meta.title` = page name, `id` = stem (replaces shared placeholder `id_01`), `Is_series: false`, then rebuild (recipe in `P0_CRITICAL_ISSUES_CHECKLIST.md`). The 35 classics have Author but no Summary → wait for HF enrich or leave empty. Section coverage 45.8% (16 untagged stories) is P2.

**Ops:** always run scripts with `.\.venv\Scripts\python.exe` (bare `py` lacks `langchain_chroma`). `WinError 32` on deleting `data/chroma_db` is a file lock; the API reset fallback still rebuilds correctly. `push_section_metadata.py` does not refresh top-level Author/Display_title — rebuild instead.

**Optional now (offline):** Firestone metadata fill → rebuild → validate (expect Author 100%); P0.2 Ollama smoke probe with `Grounded_facts_provider: "local"`.
**Waits for HF:** clean `measure_generation_length.py` run split by `has_eval`; classic Summary enrich; HF vs local facts comparison.
**Still open P1:** collection embedding fingerprint (old P0.4); lazy-init of the `chromadb.py` import side-effect.

**Anti-patterns (still):** don't upgrade to 14B yet; don't reopen Phase 1 retrieval knobs; don't treat offline accept_rate as the final truth run.
