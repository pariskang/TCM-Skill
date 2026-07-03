# 数据格式说明

## 1. 原始文件(笈成语义化 HTML)

`data/raw/<书名>/index.html`,结构:

```html
<header data-type="book">
  <h1>三指禪</h1>
  <dl class="元資料">
    <div><dt>作者</dt><dd>周學霆</dd></div>
    <div><dt>朝代</dt><dd>清</dd></div>
    <div><dt>年份</dt><dd><data value="1827">公元1827年</data></dd></div>
    <div><dt>底本</dt><dd>道光七年初刻本</dd></div>
    <div><dt>品質</dt><dd>65%</dd></div>
  </dl>
</header>

<h1>卷一</h1>       <!-- 一级篇章 -->
<h2>總論</h2>       <!-- 二级篇章 -->
<p>正文段落…</p>
```

解析规则(`tcm.py` 的 `BookParser`):

- `<h1>`–`<h6>` 维护篇章栈,段落的 `path` = 当前栈内标题按层级拼接;
- `<p> <blockquote> <li> <dd> <pre>` 等块级元素 → 一条段落记录;
- 表格按 `<tr>` 合并为一条,单元格以 ` │ ` 分隔;
- 保留全部原文字符(含 Unicode 扩展区异体字如 𤍠、𤼵)。

## 2. SQLite 数据库 `data/tcm.db`

```sql
books(book PK, author, dynasty, year, year_text, edition, quality,
      n_paras, n_chars)
paras(id INTEGER PK, book, seq, path, text)      -- seq 从 1 起,书内连续
fts   -- FTS5 external-content 表,tokenize='trigram',内容指向 paras.text
```

检索语义:

- ≥3 字查询:`fts MATCH '"<query>"'`(短语匹配)按 rank 排序;
- <3 字或 FTS 无结果:回退 `text LIKE '%q%'` 全扫描;
- 引用定位主键:`(book, seq)`,永远随 build 重建而稳定(解析顺序确定)。

## 3. `data/catalog.json`

全部书目元数据数组,字段同 books 表。体积小,可直接整体读取用于书目分析。

## 4. `data/corpus.jsonl`(可选,`build --jsonl` 生成)

每行一段:

```json
{"book": "傷寒論(宋本)", "seq": 156, "path": ["卷第二", "辨太陽病脉證並治上第五"], "text": "桂枝湯方"}
```

用途:下游 embedding 向量化、词频统计、语料分析。约数百 MB(全量)。

## 5. 目录约定

```
data/                  # 全部生成物,已 .gitignore,不入库
├── jc.data/           # git 浅克隆(fetch 全量模式)
├── raw -> jc.data/pages/book   # 或 --sample 模式下的实体目录
├── tcm.db             # SQLite + FTS5 索引
├── catalog.json
└── corpus.jsonl       # 可选
```

环境变量 `TCM_DATA_DIR` 可重定向数据目录(多项目共享一份索引时有用)。
