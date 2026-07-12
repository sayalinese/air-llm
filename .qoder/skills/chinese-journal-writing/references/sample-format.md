# User Sample Chinese Journal Format

Source sample: `assets/sample.docx`, copied from the user's `样本.docx` on 2026-06-30.

Use this file when drafting, revising, or formatting Chinese academic journal manuscripts in the user's preferred style.

## Page Setup

- Paper: A4 portrait.
- Page width: about 21.0 cm.
- Page height: about 29.7 cm.
- Margins:
  - Top: 2.54 cm
  - Bottom: 2.54 cm
  - Left: 3.175 cm
  - Right: 3.175 cm
- Header distance: about 1.50 cm.
- Footer distance: about 1.75 cm.

## Overall Structure

The sample resembles a full Chinese academic paper with the following structure:

1. Chinese title, split across one or more centered title lines when long.
2. `摘要`
3. Abstract paragraphs.
4. `关键词：...`
5. Optional `图目录` and figure/table list for long manuscripts.
6. Main text:
   - `一、绪论`
   - `二、相关理论基础`
   - Further numbered sections
   - `六、结论与展望`
7. `参考文献`

Use this sequence unless the user supplies a journal-specific template.

## Typography

Approximate sample typography:

- Chinese body text: Songti-style Chinese font.
- English letters, numbers, model names, variables, and formulas: Times New Roman where possible.
- Main title: Fangzheng Xiaobiaosong-style or similar Chinese title font, about 16 pt, bold, centered.
- Abstract heading `摘要`: Heiti-style, about 14 pt, centered.
- First-level heading: Heiti-style, about 15 pt, centered.
- Second-level heading: Kaiti-style, about 14 pt, left aligned.
- Third-level heading: Heiti-style/bold style, left aligned.
- Captions and body: inherit body style unless otherwise required.

If a font is unavailable, use practical substitutes:

- Fangzheng Xiaobiaosong -> SimSun, SimHei, or a locally available Chinese title font.
- Songti -> SimSun.
- Heiti -> SimHei.
- Kaiti -> KaiTi.
- English/numbers -> Times New Roman.

## Paragraphs And Indentation

- Body paragraph: first-line indent about 0.847 cm.
- Many body paragraphs have no explicit before/after spacing in the source file.
- Some paragraphs use line setting equivalent to single line (`w:line=240`, auto) where explicitly set.
- Second-level headings: first-line indent about 0.988 cm.
- Third-level headings: first-line indent about 0.85 cm.
- Keywords paragraph: first-line indent about 0.85 cm and starts after the abstract.
- Table/figure list entries: left indent about 1.693 cm.

When creating new content, use consistent first-line indentation and avoid mixing manually inserted spaces with Word paragraph indents.

## Heading Patterns

Use these heading markers:

- Level 1: Chinese numeral plus ideographic comma, e.g. `一、绪论`; centered.
- Level 2: parenthesized Chinese numeral, e.g. `（一）研究背景`; left aligned.
- Level 3: Arabic number plus period, e.g. `1.心源性猝死风险预测研究现状`; left aligned.

Keep heading numbering continuous. Do not mix `1.1` numbering unless the target journal requires it.

## Abstract And Keywords

- Abstract heading: `摘要`, centered, Heiti-style, about 14 pt.
- Abstract body: formal Chinese academic summary of background, objective, methods, results, and conclusion.
- Keywords line: `关键词：心源性猝死；迁移学习；多源特征融合；XGBoost；风险预测`
- Use full-width Chinese punctuation and Chinese semicolons between keywords.

## Figures, Tables, And Captions

- Figure caption pattern: `图1 研究框架图`.
- Table caption pattern: `表1 心源性猝死相关核心心电指标及临床意义`.
- Captions are centered in the sample.
- Long papers may include a figure/table directory with entries such as `图1 研究框架图4`.
- Tables in the sample commonly use 3 columns and should be readable, with clear header rows and wrapped text.

For generated manuscripts, keep captions close to their figure/table. Do not create figure/table numbers that do not correspond to actual visuals or tables.

## References

The sample uses Chinese academic reference entries similar to GB/T 7714 forms, for example:

- Thesis: `作者.题名[D].学校,年份.DOI:...`
- Journal: `作者.题名[J].期刊,年份,卷(期):页码.DOI:...`

Do not fabricate references. If references are missing, use `[需补充参考文献]` or ask the user to provide them.

## Academic Style Preferences

- Use objective and precise wording.
- Prefer phrases such as `本文以...为研究对象`, `构建...模型`, `结果表明`, `由此可知`, `进一步说明`.
- Avoid ungrounded claims such as `具有重大意义` unless explained.
- Avoid casual language, rhetorical questions, and unsupported superlatives.
- For technical work, emphasize data source, preprocessing, feature construction, modeling method, evaluation metrics, and interpretability.

## QA Checklist

Before finalizing a DOCX:

1. Page size and margins match the sample or the user's target journal template.
2. Title, abstract, keywords, headings, body paragraphs, captions, and references use consistent styles.
3. Body first-line indent is paragraph formatting, not manual spaces.
4. Chinese punctuation is consistent.
5. Figure and table numbering is continuous and captions are centered.
6. References are real user-supplied or verified entries; missing references are marked.
7. Rendered pages have no overlapping text, clipped tables, broken glyphs, or awkward blank gaps.
