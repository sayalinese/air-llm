---
name: chinese-journal-writing
description: Draft, revise, polish, structure, and format Chinese academic journal manuscripts and thesis-style papers using the user's sample Chinese journal DOCX format. Use for Chinese paper writing tasks including title, abstract, keywords, introduction, literature review, methods, results, discussion, conclusion, references, figure/table captions, academic tone polishing, outline expansion, DOCX formatting, and sample-style compliance.
---

# Chinese Journal Writing

## Core Rule

Write and format as a Chinese academic journal manuscript, following the user's sample format. When formatting or creating DOCX output, first read `references/sample-format.md` and use `assets/sample.docx` as the style/template reference when available.

Never invent citations, data, experiments, journal names, DOI numbers, or references. If the user has not supplied sources, mark citation needs with placeholders such as `[需补充文献]` or ask for source material before finalizing references.

## Workflow

1. Identify the task type: draft from outline, revise existing text, polish academic style, generate a section, format DOCX, or prepare a full manuscript.
2. Determine the manuscript field, target journal or school requirements, word count, section scope, citation standard, and whether real references/data have been supplied.
3. Use the mandatory chapter structure:
   - Title (first page, same page as abstract/keywords)
   - Abstract
   - Keywords
   - Optional table/figure lists for long papers
   - `一、绪论` (mandatory first chapter)
   - `二、相关理论基础` (mandatory second chapter)
   - Remaining chapters flexible per content
   - Conclusion and outlook (final chapter)
   - References
4. Draft in formal Chinese academic prose:
   - Prefer objective, evidence-oriented phrasing.
   - Keep paragraphs coherent: topic sentence, evidence/method/detail, interpretation, transition.
   - Avoid slogans, empty claims, exaggerated novelty, and informal wording.
   - Preserve technical terms, abbreviations, variables, and model names exactly unless asked to normalize them.
5. When revising, preserve the user's intended argument and make local improvements unless the user asks for a rewrite.
6. When producing DOCX, apply the sample format and run a visual/document QA workflow if document tooling is available.

## Writing Standards

### Abstract

中文期刊论文摘要须采用严格三段式结构，每段有明确的功能边界和内容要求。

**段落一**（背景与问题引出）：
- 以"随着…"句式起笔，铺设研究领域背景和现实意义
- 明确指出当前研究在方法或模型上存在的缺陷（不超过两项）
- 以"本文以…为研究对象"收束，并概括性提出所构建模型或方法
- **禁止出现**具体模型名（如Gemma、BERT）、技术细节、评估精度数值

**段落二**（方法与实验设计）：
- 开句点明数据来源与规模（如"本文以Aeolus数据集2024年全年约680万条航班记录为基础"）
- 展开核心方法设计，包括但不限于：数据构建方式、特征工程体系、建模策略
- 明确采用的基线对照方式（如"与LSTM、XGBoost、MLP和GCN四种基线模型在统一时间序列切分下进行对比"）
- 模型名称可以出现，如Gemma-4-E4B
- 数据平衡手段应当写入（如"结合加权随机采样处理数据不平衡问题"）

**段落三**（结果与结论）：
- 给出模型的关键量化结果，仅保留论文最核心的指标（如AUC）
- 若是对比实验，指出本文模型在对比中的相对位置（如"在对比模型中表现最佳"）
- 以简洁结论收束，落在方法有效性和应用价值两个落点上

**术语处理规范**：
- 数字单位用中文："680万"而非"680w"
- 通用缩写（如GCN、MLP、AUC）可保留
- 非必要技术细节应删除（如"五折交叉验证"、"对数概率评分"等实现细节不属于摘要范畴）
- 模型简称应当使用完整名称或通用缩写，如"Gemma"或"Gemma-4-E4B"，不缩写成非标准形式

**完整示例**：

摘要

随着航空运输业的持续发展，航班延误造成的经济损失与运力连锁降级问题日趋严重，其中延误传播因同一架飞机多航段连续执飞的耦合效应，成为延误预测中最具挑战性的环节。本文针对现有航班链构建中航司与航班号联合分组破坏物理因果链路、数据划分普遍采用随机切分引入时序信息泄漏的问题，以航班出发延误预测为研究对象，结合物理航班链构建、延误传播衍生特征工程与大规模语言模型参数高效微调，构建了基于物理航班链与语言模型的延误传播分类预测模型。

本文以Aeolus数据集2024年全年约680万条航班记录为基础，提出以飞机尾号为分组键的物理航班链构建方法，按计划出发时间排序形成最大长度为六段的因果航段序列；在此基础上设计六维延误传播衍生特征，联合气象、地理与运营特征形成二十七维输入向量。建模方面，将结构化链数据转化为自然语言文本，基于Gemma-4-E4B进行参数高效LoRA微调与生成式二分类，并与LSTM、XGBoost、MLP和GCN四种基线模型在统一时间序列切分下进行对比，结合加权随机采样处理数据不平衡问题。

实验结果表明，Gemma在测试集上取得AUC 0.71，在对比模型中表现最佳，验证了以自然语言为媒介将结构化航班链信息送入语言模型进行延误分类的可行性。本文通过物理航班链构建和传播特征工程解决了延误传播的因果建模问题，为航班延误智能预测提供了可参考的技术方案。

**Critical**: Title, abstract heading `摘要`, abstract body, and `关键词` MUST appear on the same first page. Do NOT insert page breaks between them.

### Keywords

Use the format `关键词：词1；词2；词3；词4`. Use Chinese semicolons. Prefer 3-6 terms.

### Introduction

Use a funnel structure:

1. Field background and practical significance
2. Existing research progress
3. Current limitations or unresolved problems
4. This paper's research object, method, and contribution

**硬性要求**：第一章必须为 `一、绪论`，第二章必须为 `二、相关理论基础`。无论篇幅长短，不得将文献综述或理论基础合并到绪论中作为子节。后续章节可依内容灵活组织。

### Literature Review

Organize by theme rather than source-by-source summary. Each review paragraph should connect:

1. Research direction
2. Representative method or finding
3. Limitation
4. How it leads to this paper

### Methods

Write methods with reproducibility in mind: data source, inclusion/exclusion rules, preprocessing, variables/features, model or analytical method, evaluation indicators, statistical tests, and software/tooling when known.

### Results And Discussion

Separate fact from interpretation. Present results first, then explain possible mechanisms, compare with existing work if sources are supplied, and state limitations.

### Conclusion

Summarize what was done, what was found, what it means, limitations, and future work. Avoid adding new evidence in the conclusion.

## Formatting

Read `references/sample-format.md` before creating or reformatting a DOCX. Important defaults from the sample:

- A4 portrait.
- Margins: top 2.54 cm, bottom 2.54 cm, left 3.175 cm, right 3.175 cm.
- Chinese main text font: Songti-style Chinese body text when possible; English/numbers use Times New Roman when possible.
- Body paragraphs: first-line indent about 0.85 cm (precisely 0.847 cm), line spacing 1.5x, black text, ordinary paragraph spacing unless journal rules say otherwise.
- Heading 1: `一、...`, centered, Heiti-style, about 15 pt, bold, page break before each H1 after the first.
- Heading 2: `（一）...`, left aligned, Kaiti-style, about 14 pt, first-line indent about 0.99 cm (precisely 0.988 cm), line spacing 1.5x.
- Heading 3: `1....`, left aligned, Heiti-style/bold, about 12 pt, first-line indent about 0.85 cm, line spacing 1.5x.
- Abstract heading: centered `摘要`, Heiti-style, about 14 pt.
- Keywords: `关键词：` label bold/Heiti-style, content follows in the same paragraph.
- Figure captions: `图1 ...`, centered, about 9 pt, Songti-style.
- Table captions: `表1 ...`, centered, about 9 pt, Songti-style, placed ABOVE the table.
- Tables: Use three-line table format (三线表) with thicker top, header-bottom, and bottom borders, thinner internal borders.
- References heading: centered `参考文献`, Heiti-style, about 15 pt, bold, on a new page.
- Reference entries: about 9 pt, Songti-style, single line spacing (1.15x), no first-line indent.
- In-text citations: `[1]` format as superscript markers.

### Three-Line Table Format (三线表)

Chinese academic papers require three-line tables:

- Top border: 1.5 pt thick line
- Header-bottom border: 1.5 pt thick line (separating header from data rows)
- Bottom border: 1.5 pt thick line
- All other borders (left, right, internal): 0.5 pt thin line or omitted
- Header row: bold, about 9 pt font, centered
- Data rows: about 9 pt font, centered alignment preferred

### Title Page Layout

Title, abstract heading, abstract body, and keywords must occupy the FIRST PAGE without page breaks between them. Title font: Fangzheng Xiaobiaosong-style, about 16 pt, bold, centered. If unavailable, fall back to SimHei or a bold Songti variant.

## Writing Style Rules

### Prohibited Expressions

- **Never use** `第一`、`第二`、`第三` (first/second/third) as list markers in body text. Use natural paragraph transitions instead.
- **Never use** `首先`、`其次`、`此外`、`最后` as paragraph openers unless truly necessary for logical flow.
- **Never use** parentheses `（）` to insert auxiliary explanations into sentences. Write the explanation into the sentence naturally or use a separate sentence.
- **Never use** double brackets `[[1]]` for citations—always use single brackets `[1]`.
- **Never use** AI-sounding filler phrases like `可以观察到`、`值得关注的是`、`这表明`、`在此基础上`、`综上所述`. Use concrete, direct language.

### Citation Rules

**硬性位置限制**：引用标记 `[N]` 仅允许出现在文献综述/相关研究章节。方法、实验、结果、讨论、结论等章节禁止插入任何引用标记。若需提及已有工作用于对比或解释，用自然语言表述即可，不加引用编号。

- Do NOT add citations when merely mentioning well-known model/tool names (TextCNN, BERT, RoBERTa, BERTopic, SnowNLP, DeepSeek, etc.) unless discussing their original research contribution.
- Add citations when referencing specific research findings, methods proposed by authors, or evaluation results from prior work.
- Maximum 2 in-text citations per paragraph. If more are needed, consolidate or move some to a separate sentence.
- Never have 3+ consecutive citation markers `[1][2][3]` in a row—this is a citation cluster and must be broken up.
- Each reference number should appear at most 3 times in the entire manuscript.

### Natural Academic Expression

- Favor short, direct sentences over long compound sentences.
- Avoid rhetorical questions in academic writing.
- Use `本文` (this paper) consistently for self-reference, not `我们` (we) or `笔者`.
- Prefer concrete data descriptions: `达到 88.89%` not `取得了较好的效果`.
- Remove filler transitions: go directly from data to interpretation.

## Figure And Table Handling

### In Markdown Source

Use Markdown image syntax for figures:

```
<center>图1　五种模型性能对比</center>

![](../code/结果/论文插图/1_模型总体性能对比.png)
```

For tables, use Markdown table syntax with centered `<center><b>表1　标题</b></center>` caption above the table.

### In DOCX Generation

When converting Markdown to DOCX:
- `![]()` → embed the image centered, width ~5.5 inches
- `<center>图N ...</center>` → centered caption, 9 pt Songti-style
- `<center><b>表N ...</b></center>` → centered bold caption, 9 pt Songti-style
- Markdown tables → three-line table format in DOCX (see Formatting section)

### Figure/Table Numbering

- Figures and tables share a unified numbering sequence within each chapter, or use continuous numbering across the full paper.
- Figure caption: `图1 描述` centered BELOW the figure.
- Table caption: `表1 描述` centered ABOVE the table.
- Cross-reference in text: `如表1所示` or `图2展示了`.

## DOCX Handling

For DOCX creation or editing:

1. Prefer basing the document on `assets/sample.docx` if a sample-style document is requested.
2. Preserve the sample's styles where possible instead of rebuilding formatting from scratch.
3. Apply fonts, paragraph indents, heading styles, captions, page geometry, and references consistently.
4. For existing papers, make local OOXML or style-preserving edits when possible so user-corrected spacing, indentation, and table layout are not disturbed.
5. Apply the formatting script (if available) after any text content changes to ensure consistent font/spacing.
6. If using the documents skill, render the DOCX to page images and inspect layout before delivery.

## Output Discipline

- If asked for text only, output clean manuscript text without meta commentary.
- If asked to revise, provide the revised version first; add a short change note only if useful.
- If information is missing, use clear placeholders rather than inventing facts.
- For references, accept only user-supplied references or verifiable sources. Mark missing citations explicitly.
- When trimming content to meet word count targets, prioritize removing verbose transitions and redundant explanations over data or analysis.
