# Final Synthesis — Stage 4A: Researcher-Level Alignment & Documentation Audit

2026-10-02（Australia/Sydney）。**Workflow 已更新；文档分类与审计已完成。等待 researcher + ChatGPT review，未开始文档重写。** Stage 4 学习材料已存在，researcher mastery verification 保持 paused；Stage 5 未开始。

## 1. 为什么需要对齐

研究者是第一次系统接触 AI Music 和 representation probing 的 AI undergraduate。能够回忆 A–F 的路径、认出术语并跟随实验，不代表已经能独立解释所有方法细节。此次 review 提供的困难集中在表示的直觉、读出怎样学习、指标看什么、RQ4 实际回答了什么，以及解释边界过多。本轮据这些已报告的困难审计文档，没有重新测试研究者能力。

此前材料保留了正确的科学范围，但学习入口经常过早引入抽象词、精确配置或完整 caveats。问题是解释顺序与掌握要求，不是要降低科学标准。修正目标是让研究者先能用自己的话说清“它在做什么”，再逐步接上必要术语与方法理由。

## 2. Workflow 更新了什么

更新 [docs/project_workflow.md](../project_workflow.md)。它本来就负责协作、learning checkpoints、私人笔记和新会话恢复，因此比冻结实验路线的 roadmap 更适合承载长期学习默认值；没有新建目录或文档层级。

具体更新位于 Core Research Philosophy 的新增 learning-profile 小节、ChatGPT explanation 要求、Understand / Learning Checkpoint、Three-Tier Learning Priority、Personal Notes、New Chat / Session Recovery，以及 Module Completion Checklist。它们共同明确受众、解释顺序、三层负担和验证状态；既有实验决策、冻结规则、角色责任与私人文件政策保留。

新增原则只覆盖学习期待。旧 learning plans / checkpoints 中更高的掌握假设不再自动适用，历史记录中的科学决定和结果不因此改变。创建材料、review 材料、验证研究者理解必须分别报告。

## 3. Researcher learning profile 与当前目标

Workflow 冻结的默认背景是：**“the researcher is an AI undergraduate encountering AI Music and representation probing for the first time.”** 不预设 MIR 熟练度或 probing literature 背景，不以复述成熟研究者的论文措辞证明理解。必要的 regression、Transformer、representation、regularization 和 metrics 可以学习，但从项目直觉开始。

| 层级 | 当前要求 | 学习负担 |
|---|---|---|
| **Tier 1** | 独立解释项目问题、Valence / Arousal、MERT 表示的大意、frozen + linear 的目的、基本实验流程、Ridge、三种数据角色、三种指标、四问推进、主要发现，以及预测成功为何不等于真正理解情绪。 | 将来简化闭卷验收的核心；接受准确的日常语言。 |
| **Tier 2** | 遇到时理解、提示后解释 scaler、alpha、ID 对齐、常数参照、声学表示大意、residual、RMS / BPM、Layer-12 保留、单条提取理由、Test exposure 和声学比较边界。 | 有支持地学习；不用在首次项目介绍里主动讲全。 |
| **Tier 3** | 精确参数 / tie-break、各维特征、逐项系数、counts / shapes、GroupNorm / padding 实现、hashes、CLI、serialization 与验证 machinery。 | 查阅；不能作为掌握证明。若影响决策，只把短理由提升到 Tier 2。 |

## 4. Progressive explanation rule

固定顺序是 **直觉 → 项目含义 → 专业术语 → 必要研究细节 → 可选工程查阅**。先让研究者知道一个东西在干嘛，再连接到 DEAM、两个评分、MERT 向量、Ridge 或某个 RQ，随后命名。

例如 Ridge 的入口应先讲“用输入的数字加权预测评分，并限制某些权重过大”，再介绍 L2 regularization 和 alpha；不把 correlated high-dimensional representation 的措辞放在理解之前。Nuance 只有在实验理由、解释边界或讨论需要时增加。保留科学限制，不要求第一次脱稿介绍就主动列出所有限制。

## 5. Documentation classes 与审计范围

盘点了 **45 份现有 Markdown 文档**：README、Final Report、workflow、roadmap、六份 accepted logs、17 份 Module Stage reports、六份已有 Final Synthesis records、七份私人学习笔记、五份私人 handoffs。另检查 Evidence Map 的 SVG 标题、节点和说明文字；PNG 是相同地图的显示版本。机器可读结果、代码、原始数据和 Obsidian 状态不是本轮语言改写对象。

七份学习笔记逐份审读；公开入口和 Final Report 检查研究故事与具体措辞；科学与工程记录按连贯组检查受众、章节、learning checkpoints 和恢复约定。本轮不是数值重审或逐字校对全部工程历史，不对文档作数字评分。

| 类别 | 实际受众与目标语言 | 默认处理 |
|---|---|---|
| **A — Personal learning notes** | 首次接触 AI Music 的本科生；直觉、具体例子、必要术语，分清 primary / supporting / lookup。 | 主学习入口优先调整；深层材料可保留，逐份判断。 |
| **B — README / map** | 想快速了解本科研究项目的外部读者；专业、直接，研究故事优先。 | 保留有效入口，只简化确实多余的措辞。 |
| **C — Final Research Report** | 本科 mini research report 的读者与 supervisor；保留学术结构、证据和限制，用较直接的 academic English。 | 局部语言简化，保持科学内容。 |
| **D — Logs / protocol / roadmap** | 研究者及未来协作者核查决策与历史；technical but readable。 | 低优先级、选择性处理；细节不因“不背”而删除。 |
| **E — Stage / implementation / verification / handoff records** | 实现、审核及恢复上下文的人；准确与可追溯优先。 | 保留技术记录；不为变容易而重写历史。 |

## 6. 文件与文件组审计

以下处理标签是未来范围建议，不表示本轮已改写。私人路径仅用于定位，不复制笔记全文或建立公开学习入口。

| 文件 / 连贯组 | 受众与当前语言判断 | 抽象程度与简化风险 | 建议 |
|---|---|---|---|
| 七份 `personal_notes/*what_i_should_understand.md` | 目前混合主学习、深层复习和 interview preparation；中文主体不等于负担已经合适。 | 术语和 caveats 常先于直觉。直接删除限制会改变含义，应改顺序和层级。 | 分别见第 7 / 8 节；不整体等量重写。 |
| [README](../../README.md) | 开头说清固定模型与简单读出，随后有四问、回答、图表和来源；对外部读者基本有效。 | 少量 technical terms 服务精确范围，不足以要求整个 landing page 改写。 | **KEEP AS-IS**；第 10 节区分语言与状态。 |
| [Evidence Map SVG](../../outputs/figures/project_evidence_map.svg) / PNG | 快速展示问题推进；短节点、明确 foundation 与四问。 | “Across depth / Selected acoustics”等标签已有问题上下文；复用与 Test 说明不能删。 | **KEEP AS-IS**。 |
| [Final Report](../final_research_report.md) | 结构合适，Method / Results / Discussion 分工清楚；部分段落用较密的抽象名词概括具体动作。 | 可以用直接动词与具体对象简化；不能顺便放宽范围或省略 provenance。 | **LIGHT SIMPLIFICATION**，语言层面中等优先级；见第 9 节。 |
| [Workflow](../project_workflow.md) | 项目协作与学习规范；此前 plain-language 原则缺少明确受众和分层验证。 | 关键缺口是规则，并非要把整份 operating protocol 写成教程。 | 本轮已作定点对齐，其余 **KEEP AS-IS**。 |
| [Post-D roadmap](../research_roadmap_after_module_d.md) | 冻结规划、范围与恢复依据；明确记录当时尚未开始的 E/F。 | 研究范围的抽象表达有记录用途。旧学习要求不再作当前验收标准；历史状态不可倒改。 | **KEEP AS-IS**；未来学习解释服从新 workflow。 |
| [Accepted A/B logs](../research_logs/) | 定义和验证研究输入的科学记录；population、representation 与一致性细节有明确用途。 | Compatibility / padding 等细节应该可查，不能因初学者不背而抹去。 | **DO NOT TOUCH — TECHNICAL RECORD**。 |
| Accepted C/D logs | 原始 endpoint 与完整 depth evidence 的记录，包含决策、图表、结果和解释边界。 | 较技术化适合这个受众；旧“应能解释”段落不是新验收题库。 | **DO NOT TOUCH — TECHNICAL RECORD**。 |
| Accepted E/F logs | 声学定义、比较、关联与 Test history 的正式记忆。 | 压缩 recipe、复用说明或 F 描述性状态容易改变证据含义；保留细节。 | **DO NOT TOUCH — TECHNICAL RECORD**。 |
| 17 份 Module reports：A2–A5、B1–B3、C1–C4、D1、E1–E3、F1–F2 | 概念准备、protocol、实施、失败与验证的阶段记录，不是连续教材。 | A2 的模型调查、C1 的 concepts、C2 的 protocol 仍是历史 / 来源；其余工程密度符合核查用途。 | **DO NOT TOUCH — TECHNICAL RECORD**；可作 reference。 |
| 六份已有 Final Synthesis records：Stages 1–3 design / audit / implementation，Stage 4 design / implementation | 给研究者和审核者记录范围、审批与交付，不是本科核心学习正文。 | Stage 4 design 的旧高掌握目标需由当前 workflow 明确覆盖，而非修改过去已记录的方案。 | **DO NOT TOUCH — TECHNICAL RECORD**。 |
| 五份 `handoffs/`：C→D、D review、post-D→E、F start、Final Synthesis | Private/local 状态快照与恢复助手。都指向 workflow 或正式记录。 | 旧 next-step / learning expectations 不能作为当前状态；技术密度可接受。 | **DO NOT TOUCH — TECHNICAL RECORD**；未来 handoff 必须带新 profile / tiers / pause 状态。 |

## 7. Module A–F 私人笔记专项审计

对应文件名均为 `personal_notes/module_<letter>_what_i_should_understand.md`。表内 Tier 1 是本模块承担的核心直觉，不要求每份笔记覆盖全项目的十一项目标。

| Module | Tier 1：真正需要会讲 | Tier 2：有提示时理解 | Tier 3：查阅 |
|---|---|---|---|
| **A** | 哪些音频配什么静态评分；MERT 把音频变成数字；冻结与时间汇总的大意；这里还没有预测结果。 | Mono / 输入格式处理的目的、full-song 排除理由、13 levels 与 768 数字的含义。 | 兼容版本、权重诊断、exact preprocessing、shape / sample-count 细节。 |
| **B** | 先为每条音频准备一致输入，再训练预测器；数据划分的基本角色。 | ID 避免错配；单条提取保护一致性；一份表示可复用。 | GroupNorm / mask / padding 实现、hooks、cache schema / shapes、完整性与恢复机制。 |
| **C** | 固定 MERT、学习简单读出；回归预测两个数值；Ridge 的权重限制；Train / Validation / Test；三个指标的大意和正面结果。 | Alpha、scaler、Train-mean reference、Layer-12 决策和 held-out 的对应关系。 | 网格 / tie-break、selected alpha 数值、精确 Val/Test MAE / r、solver / gates / 重算过程。 |
| **D** | 为什么读不同深度；Valence 较稳定、Arousal 中间较高后期下降；更深不是总更好。 | 每个 level 分别预测两个目标；共同方法与各自选 alpha；Layer-12 复用。 | 精确最大层差值、每层指标 / alpha、NaN isolation 检查、全部配置和机器验证细节。 |
| **E** | 两种音频描述接受同类预测；声学本身有效，MERT 在此比较更强；差距原因尚未识别。 | 51-D 大致类别、RMS / BPM 的含义、共同框架、预指定 comparator、selected acoustics 的范围。 | 完整维度 recipe、个别 BPM diagnostic、九位小数、各 Stage counters 与 alpha 数值。 |
| **F** | 不重新训练；检查现有结果与 Tempo / Energy 是否一起变化；记住 Energy 模式、Tempo 的有限简单线性证据，关联不证明原因。 | Target / prediction / residual 分工、残差正负、RMS / BPM、Test exposure；为何不能算 performance share 或命名剩余信息。 | 全 40 系数、相关系数的尺度推导、验证路径、gate / provenance 实现。 |

| Module | 当前具体语言 / 负担问题 | 建议处理与程度 |
|---|---|---|
| **A** | §1 的普通“地基”开头有效，但很快进入 compatibility versions 与 `[B,T,768]`；§5 将 shape 和 checkpoint 正确加载并列为脱稿解释重点。Mono 需要先从左右声道合成一个输入的用途解释。 | **LIGHT SIMPLIFICATION**：补直觉入口、标 tier、减轻回顾与 checkpoint；技术主体可作 deeper reference。 |
| **B** | §1、§2.3 和一分钟 recap 都带 GroupNorm / mask / representation-equivalence；§6 第 6 项让深层原因成为解释要求。说明输入一致性的短理由即可承担当前学习任务。 | **KEEP AS-IS，作为 deeper reference**；不整篇重写。在未来主指南给短版 B 作用，详细机制留这里。 |
| **C** | §1 第一段先用“容量受限 / 正则化 / untouched / 线性解码”描述实验；§2.3 和 metrics 段已有可用直觉，但缺一个贯穿输入、评分、预测与误差的具体例子。§5 第 11–12 项还要求回忆 Val/Test 多个数值。 | **LEARNING-LEVEL REVISION**：较明显调整主入口与概念例子；现有表格和协议段移作支持 / 查阅，不把全文变成简写版。 |
| **D** | 大白话开头和“读取器”解释已好；后续小层差值、tie-break、NaN isolation 与八项边界穿插，What I should now be able to explain 还要求辨析六位小数的层间差值。 | **LIGHT SIMPLIFICATION**：保留曲线故事，精确差值与 gate 移 lookup，方法 nuance 提示后展开。 |
| **E** | 开头和两分支比较已经清楚；§3 全 recipe、§5 diagnostics、§10 英文口述稿与 §11 多项方法负担过多。§9 / Quick Revision 的 F 尚未开始是 E closure 的历史状态。 | **LIGHT SIMPLIFICATION**：保留主要解释，recipe / debug 作查阅；口述稿是可选示例。将历史状态与当前学习入口区分，保留当时事实。 |
| **F** | “真实跟不跟它走 → 预测跟不跟它走 → 错误跟不跟它走”是好的直觉，值得保留；但随后多次要求讲 Energy-associated structure / plausible partial explanation，§3 又进入三个 r 不可相减及尺度关系。 | **LEARNING-LEVEL REVISION**：主入口用评分、预测、误差的教学例子接上三问；先说 RQ4 看到了什么，再逐步引入 residual 和必要边界。 |

优先需要明显学习层级调整的是 C/F 的主入口，A/D/E 为选择性轻改，B 保留深层用途。它们不需要六份同长度、同密度的新教程；下一阶段可先在统一主指南里解决 C/F，Module 文件本身的改写另待范围批准。

## 8. Project personal note 专项审计

`personal_notes/project_what_i_should_understand.md` 有完整 mental model、四问推进和来源，适合作 **deeper reference**。问题是它同时承担 13 个概念组、18 项方法理由、九项边界、12 个 RA 问题及闭卷标准，容易被读成一次性掌握清单。§3 的数字维度、§5 的提取 / 评估 nuance、§6 的全部边界和末尾七项验收不能自动成为当前 primary burden。原验收要求已由 workflow 暂停；本轮文件不改。

| 已报告的 friction | 后续主学习层应怎样处理 |
|---|---|
| 13 representation levels 与“13 layers”混淆 | 先画出取不同位置输出的意思，再明确一个 Pre-Transformer 加 12 个 Transformer layers；数字不用当记忆测验。 |
| 768-D 缺少直觉 | 先说一条音频被总结成一串数字；768 是长度，不是 768 种情绪或标签，各维未经验证不能命名为具体概念。 |
| Mono preprocessing 的作用 | 用左右声道合成单声道、满足统一输入格式的例子；算式和重采样细节查阅。 |
| Padding / GroupNorm 太深 | 提示后解释“批处理得到的表示不一致，因此采用单条”；详细传播机制 Tier 3。 |
| Ridge probe 不稳定的直觉 | 先展示音频数字怎样被加权变成评分，再说明限制过大权重，最后命名。 |
| Alpha / Validation selection | 核心先掌握 Validation 用于比较设置；alpha 的强弱和选择细节有提示再展开。 |
| 每个 level 的两个 targets | 同一个输入，分别学习预测 Valence 和 Arousal；不是一个目标有 768 个标签。 |
| MAE / R² / Pearson r | 从错多少、比均值参照改善多少、是否一起升降三个问题开始，用明确标记的教学例子。 |
| MERT > 51-D 无法识别原因 | 先理解两个具体表示的预测比较；提示后说明 selected recipe 不完备、维度与容量不同。 |
| RQ4 target / prediction / residual | 先区分实际评分、模型评分和有方向的差，再连接描述量；residual 为 Tier 2。 |
| RQ4 究竟回答什么 | 核心是已有结果与两种测量怎样共同变化，发现有限 Energy 关联与较少简单线性 Tempo 证据；没有新训练或机制识别。 |
| Boundaries 过多、抽象 | 主线先守住预测不等于理解、关联不证明原因；其余必要限制在相应问题与提示中展开，完整科学范围仍保留在 reference / public report。 |

建议保留当前文件与全部查证入口，后续在同一 `personal_notes/` 目录新增短 primary student learning guide，不把 reference 全文再翻译或压缩一遍。新指南的掌握目标是 Tier 1；Tier 2 用支持框与链接，Tier 3 给查阅位置。

## 9. Final Research Report：语言与科学内容分开审计

**只需局部语言简化，不需要结构重写。** 现有八个部分、Method / Results / Discussion 的职责、tables / figures 与来源分工合理。较难的地方主要在 Introduction / Discussion 把具体动作压成抽象名词组合。正式报告可以比主学习指南更技术化；并非每个专业术语都应删除。

以下是 [当前报告](../final_research_report.md) 的具体模式和可替代的英文表达，**只供 review，不是已实施改写**。行号按本轮未修改文件定位。简化句仍保留研究对象、条件、比较或证据边界；最终实施时还应核对所在整段。

| 位置与当前 style / pattern | 为什么费力 | 较直接的 wording pattern，科学主张保持 |
|---|---|---|
| Introduction，L11：“This separation makes the empirical question one of linear accessibility” | 读者先处理 separation / empirical / accessibility 三个抽象词，才能找回谁固定、谁训练。 | “By keeping MERT fixed and training only a linear readout, we test whether it can predict emotion ratings on samples excluded from fitting and selection.” |
| Discussion，L119：“distributed, target-dependent accessibility across representation depth” | 压缩了多个结果，要求读者先懂 research shorthand。 | “Under this probe and split, emotion ratings can be read at several levels, and the two targets show different patterns across depth. Layer 12 alone does not describe the full pattern.” |
| Discussion，L121：“The observations are conditioned on temporal mean pooling, feature standardization and regularized linear regression.” | Conditioned on 容易让初学者误以为这里做了额外条件统计分析。 | “These results apply to time-averaged representations read by Ridge probes whose inputs were standardized using Train statistics.” |
| Discussion，L123：“The shared evaluation procedure makes the comparison interpretable within the study; it does not transform it into a test of independence from all acoustics.” | 两个抽象动作遮住了具体比较与没有回答的问题。 | “The same split and probing rules help us compare these two representations, although their dimensions and capacity differ. This does not test whether MERT predicts emotion independently of all acoustic information.” |
| Discussion，L125：“RQ4 consequently constrains the stronger RQ3 result without assigning it a causal decomposition.” | Constrains / causal decomposition 过早要求成熟的方法解释语言。 | “RQ4 helps interpret MERT's stronger RQ3 scores. These correlations do not show how much performance was caused by Energy.” |
| Discussion，L129：“their convergence is an integrated interpretation of one study rather than multiple independent confirmations” | Convergence / integrated interpretation 将证据复用的实际事实藏在概括里。 | “The questions use the same data and sometimes reuse predictions. Together they form one study, rather than several independent confirmations.” |
| Limitations，L139：“do not make later Test observations untouched evidence or independent replication” | 单独说证据身份，读者可能不清楚为什么；需要先放回查看 Test 的历史。 | “We kept the analyses fixed and did not tune after seeing Test results. However, Test had already been viewed, so the later associations are descriptive findings, not untouched evidence or independent replication.” |

不得因这些语言建议更改 metrics、RQ 定义、总体与 split、方法选择、limitations、provenance、Test-history disclosures 或既有科学强度。尤其不能把 selected acoustic comparison 改成 beyond-all-acoustics，也不能把 F 描述性证据改成确认性。保留必要限制，用可说明实际动作与原因的句子表达它们。

## 10. README 审计

README 的第一句已从固定模型和简单 linear probe 开始；四问顺序、Question / Answer、结果图表与 read-more 来源让外部读者能快速进入项目。技术 foundation 较密，但可跳读，详细规则另有日志。它已经承担 undergraduate research-project landing page 的作用，不必因私人笔记需要降低负担而同步重写。

**语言建议：KEEP AS-IS，没有必须改的语言问题。** 不建议结构改动、重写整页或添加另一套解释体系。Current Project Status 仍写 Stage 4 next，是 Stage 3 checkpoint 时的状态；现在材料已实现且验收 paused，这个状态差异应在后续获准的状态更新中单独处理，不把它算成学术措辞问题。本轮 README 完全不改。

## 11. 推荐的后续 revision scope

优先建立一个短 primary learning layer，集中解决研究者实际卡住的概念，而不是先修全仓库文字。建议顺序：项目问题和两个评分 → 音频如何变成数字与汇总输入 → 读出怎样学习 → Train / Validation / Test → 三个指标的直觉 → 四问与主要发现 → 足够准确的结论。C/F 的教学例子可以整合进这层，A/B 只讲 foundation 的用途，D/E 保留已有大白话主线。

Primary 层以 Tier 1 为主。Tier 2 在相关位置以提示或短 supporting section 出现，完整细节指向当前 reference；Tier 3 给查阅入口。每个例子明确标作教学例子，不冒充 DEAM 结果。专业术语仍要学，但不能用术语数量、精确小数或主动列完 caveats 衡量掌握。

Module C/F 的局部 learning-level revision、A/D/E 的轻改可以在主指南 review 后按需要决定，不一次要求六份重新实现。Final Report 的七种措辞模式适合另一次局部 academic-English pass；README 没有语言改写必要。公开报告与私人学习材料的任务应区分，避免为了教程语气改动正式证据表达。

## 12. 应保持不动的文件与内容

本轮不改 README、Final Report、Evidence Map、其他 figures、六份 accepted logs、冻结 roadmap、已有 Stage reports、七份私人学习笔记、五份旧 handoffs、代码、configs、数据与科学结果。旧 pending / next-step 状态保留为历史，当前 continuation 依据最新用户指令与已更新 workflow；旧 Stage 4 design / note 的验收要求不能重新启动暂停的验证。

未来语言处理也必须保留全部 scientific evidence、定义、指标、决策理由与 Test history。更好学习入口不会把 engineering / provenance records 变成必须背的教程，也不会反过来删除它们的技术细节。

## 13. Proposed next implementation stage

建议 **Stage 4B — Primary Student Learning Layer Implementation**：在已有 `personal_notes/` 下创建一份私人、Git-ignored 的主学习指南，例如 `project_primary_learning_guide.md`，保留当前总笔记为 deeper reference。先完成 Tier 1 的连贯解释、C/F 所需具体例子、Tier 2 提示与 Tier 3 查阅入口，核对事实与隐私，然后进行材料 review。

这只是待批准方案，本轮没有创建该文件，也没有改写任何学习文档。Simplified closed-book verification 继续 paused，只有后续明确恢复并在材料 review 后，才以自己的话验证 Tier 1、提示辅助 Tier 2。Stage 5、额外实验与公开报告改写不属于建议的第一步。

## 14. Researcher decisions、integrity 与 stop gate

需要 researcher + ChatGPT review 的是后续呈现范围：是否采用“新 primary guide + 保留 deep reference”的方案；是否先只实施主指南、之后再按需要轻改 Module 笔记；Final Report 的局部语言 pass 是否另行授权。推荐前两项采用上述分步范围，报告语言 pass 独立处理。无需重新决定 scientific protocol，也无需为了本次审计开展能力测试。

仓库只修改 `docs/project_workflow.md` 并新增本报告。与实施前基线比较，其余 **106 个 tracked 文件**、七份私人笔记、五份 handoffs 和两份既有 untracked Stage 4 records 的 SHA-256 保持一致；Git index 与 HEAD 保持不变。HEAD 仍为 `9b1d76190d031fecdb15687b79fc86014455411a`。Workflow 改动未 staged，本报告 untracked，未 commit / push。原有四个 untracked Obsidian 文件未读取或修改；私人 notes / handoffs 继续 ignored。

检查限于文档阅读、语言 / 层级审计、链接与 Git / hash 核对，没有运行 experiment、inference、fitting、extraction 或 metric recomputation。

**Stop gate：等待 researcher + ChatGPT review。** 不开始重写，不恢复闭卷验收，不开始 Stage 5。
