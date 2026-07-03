#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tcm.py — 中医古籍语料统一命令行工具(笈成 CC0 语料,1300+ 部典籍)

子命令:
  fetch    从 GitLab (jicheng/jc.data) 拉取语料原始文件
  build    解析 HTML 原始文件,生成 catalog.json / corpus.jsonl / tcm.db(FTS5 索引)
  search   全文检索(繁体查询;>=3 字用 FTS5 trigram,<3 字自动回退 LIKE)
  get      按 书名 + 段落号/篇章 精确取原文(带出处)
  info     查看某书元数据与目录
  list     浏览/筛选书目(按朝代、作者、书名关键词)
  status   检查语料与索引是否就绪

设计约束:仅用 Python 标准库;所有输出面向 LLM 消费(紧凑、含出处、可直接引用)。
数据授权:笈成公有领域典籍以 CC0 释出(https://jicheng.tw/tcm/copyright.html)。
"""

import argparse
import html
import json
import os
import re
import sqlite3
import subprocess
import sys
import unicodedata
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

# ---------------------------------------------------------------------------
# 路径约定:数据目录默认在仓库根的 data/,可用 TCM_DATA_DIR 覆盖
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent


def data_dir() -> Path:
    env = os.environ.get("TCM_DATA_DIR")
    if env:
        return Path(env)
    # skills/tcm-ancient-texts/scripts/ -> 仓库根/data
    for parent in SCRIPT_DIR.parents:
        if (parent / ".git").exists() or (parent / "data").is_dir():
            return parent / "data"
    return SCRIPT_DIR.parents[2] / "data"


DB_NAME = "tcm.db"
RAW_DIR_NAME = "raw"
CATALOG_NAME = "catalog.json"
CORPUS_NAME = "corpus.jsonl"

GITLAB_PROJECT = "jicheng%2Fjc.data"
GITLAB_API = "https://gitlab.com/api/v4/projects/" + GITLAB_PROJECT
GIT_URL = "https://gitlab.com/jicheng/jc.data.git"


def db_path() -> Path:
    return data_dir() / DB_NAME


def raw_dir() -> Path:
    return data_dir() / RAW_DIR_NAME


# ---------------------------------------------------------------------------
# fetch — 拉取语料
# ---------------------------------------------------------------------------

def http_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "tcm-skill/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def api_json(url: str):
    return json.loads(http_get(url).decode("utf-8"))


def list_books_remote():
    """遍历 GitLab keyset 分页,返回全部书名目录。"""
    names = []
    url = (GITLAB_API + "/repository/tree?path=pages/book&per_page=100"
           "&pagination=keyset&order_by=name")
    while url:
        req = urllib.request.Request(url, headers={"User-Agent": "tcm-skill/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.load(r)
            link = r.headers.get("Link", "")
        names += [e["name"] for e in data if e["type"] == "tree"]
        url = None
        for part in link.split(","):
            if 'rel="next"' in part:
                url = part.split("<")[1].split(">")[0]
    return names


def cmd_fetch(args):
    dest = raw_dir()
    dest.mkdir(parents=True, exist_ok=True)

    if args.sample:
        # 快速采样模式:通过 API 抓取前 N 本(或指定书名),用于验证流水线
        if args.books:
            names = args.books
        else:
            print(f"[fetch] 列出远端书目 …", file=sys.stderr)
            names = list_books_remote()[: args.sample]
        for i, name in enumerate(names, 1):
            out = dest / name
            out.mkdir(parents=True, exist_ok=True)
            fp = out / "index.html"
            if fp.exists() and not args.force:
                continue
            path = urllib.parse.quote(f"pages/book/{name}/index.html", safe="")
            url = f"{GITLAB_API}/repository/files/{path}/raw?ref=master"
            try:
                fp.write_bytes(http_get(url))
                print(f"[fetch] ({i}/{len(names)}) {name}", file=sys.stderr)
            except Exception as e:
                print(f"[fetch] 失败 {name}: {e}", file=sys.stderr)
        return

    # 全量模式:浅克隆(最快、可增量 pull)
    clone_dir = data_dir() / "jc.data"
    if (clone_dir / ".git").exists():
        print("[fetch] 已存在克隆,执行 git pull …", file=sys.stderr)
        subprocess.run(["git", "-C", str(clone_dir), "pull", "--ff-only"], check=True)
    else:
        print(f"[fetch] 浅克隆 {GIT_URL} → {clone_dir}(约数百 MB,需几分钟)…",
              file=sys.stderr)
        subprocess.run(
            ["git", "clone", "--depth", "1", GIT_URL, str(clone_dir)], check=True)
    # 将 pages/book 链接为 raw 目录
    src = clone_dir / "pages" / "book"
    if dest.is_symlink() or (dest.exists() and not any(dest.iterdir())):
        try:
            if dest.is_symlink() or dest.is_dir():
                if dest.is_symlink():
                    dest.unlink()
                else:
                    dest.rmdir()
        except OSError:
            pass
    if not dest.exists():
        dest.symlink_to(src, target_is_directory=True)
    print(f"[fetch] 完成。原始文件位于 {src}", file=sys.stderr)


# ---------------------------------------------------------------------------
# 解析器 — 笈成语义化 HTML → 结构化段落
# ---------------------------------------------------------------------------

BLOCK_TAGS = {"p", "blockquote", "li", "caption", "figcaption", "pre", "dd"}
HEADING = re.compile(r"^h([1-6])$")


class BookParser(HTMLParser):
    """解析单本书的 index.html。

    结构约定(见 pages/schema):
      <header data-type="book"> 内含 <h1>书名</h1> 与 <dl class="元資料">
      正文以 h1..h6 分层,<p> 等块级元素为段落。
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.meta = {}
        self.title = None
        self.in_header = False
        self.header_done = False
        self._dt = None
        self._buf = []          # 当前块级元素文本缓冲
        self._capture = 0       # 块级捕获深度
        self._h_level = 0       # 当前所在标题级别(0=非标题)
        self.heading_stack = [] # [(level, text)]
        self.paras = []         # [{"path": [...], "text": ...}]
        self._in_dt = False
        self._in_dd = False
        self._in_table_row = False
        self._row_cells = []

    # -- helpers --
    def _flush_block(self):
        text = "".join(self._buf).strip()
        text = re.sub(r"\s+", " ", text)
        self._buf = []
        if text:
            path = [t for _, t in self.heading_stack]
            self.paras.append({"path": path, "text": text})

    def _set_heading(self, level, text):
        text = re.sub(r"\s+", " ", text.strip())
        if not text:
            return
        while self.heading_stack and self.heading_stack[-1][0] >= level:
            self.heading_stack.pop()
        self.heading_stack.append((level, text))

    # -- HTMLParser hooks --
    def handle_starttag(self, tag, attrs):
        if tag == "header":
            self.in_header = True
            return
        if self.in_header:
            if tag == "dt":
                self._in_dt = True
                self._buf = []
            elif tag == "dd":
                self._in_dd = True
                self._buf = []
            elif tag == "h1":
                self._capture += 1
                self._buf = []
            return
        m = HEADING.match(tag)
        if m:
            self._h_level = int(m.group(1))
            self._buf = []
            return
        if tag in BLOCK_TAGS:
            self._capture += 1
            if self._capture == 1:
                self._buf = []
        elif tag == "tr":
            self._in_table_row = True
            self._row_cells = []
        elif tag in ("td", "th"):
            self._capture += 1
            self._buf = []
        elif tag == "br" and (self._capture or self._h_level):
            self._buf.append(" ")

    def handle_endtag(self, tag):
        if tag == "header":
            self.in_header = False
            self.header_done = True
            return
        if self.in_header:
            if tag == "dt":
                self._in_dt = False
                self._dt = "".join(self._buf).strip()
            elif tag == "dd":
                self._in_dd = False
                if self._dt:
                    self.meta[self._dt] = re.sub(
                        r"\s+", " ", "".join(self._buf).strip())
                self._dt = None
            elif tag == "h1":
                self._capture = max(0, self._capture - 1)
                if not self.title:
                    self.title = "".join(self._buf).strip()
            return
        if HEADING.match(tag) and self._h_level:
            self._set_heading(self._h_level, "".join(self._buf))
            self._h_level = 0
            self._buf = []
            return
        if tag in BLOCK_TAGS:
            if self._capture == 1:
                self._flush_block()
            self._capture = max(0, self._capture - 1)
        elif tag in ("td", "th"):
            cell = re.sub(r"\s+", " ", "".join(self._buf).strip())
            self._row_cells.append(cell)
            self._buf = []
            self._capture = max(0, self._capture - 1)
        elif tag == "tr":
            self._in_table_row = False
            row = " │ ".join(c for c in self._row_cells if c)
            if row:
                path = [t for _, t in self.heading_stack]
                self.paras.append({"path": path, "text": row})
            self._row_cells = []

    def handle_data(self, data):
        if self._capture or self._h_level or self._in_dt or self._in_dd:
            self._buf.append(data)


def parse_book(fp: Path):
    parser = BookParser()
    parser.feed(fp.read_text(encoding="utf-8", errors="replace"))
    year = None
    raw = fp.read_text(encoding="utf-8", errors="replace")
    m = re.search(r'<dt>年份</dt><dd>.*?<data value="(-?\d+)"', raw)
    if m:
        year = int(m.group(1))
    return {
        "book": parser.title or fp.parent.name,
        "author": parser.meta.get("作者", ""),
        "dynasty": parser.meta.get("朝代", ""),
        "year": year,
        "year_text": parser.meta.get("年份", ""),
        "edition": parser.meta.get("底本", ""),
        "quality": parser.meta.get("品質", ""),
        "paras": parser.paras,
    }


# ---------------------------------------------------------------------------
# build — 生成 catalog / corpus / SQLite FTS5 索引
# ---------------------------------------------------------------------------

def cmd_build(args):
    src = raw_dir()
    if not src.is_dir():
        sys.exit("[build] 未找到原始语料,请先运行: tcm.py fetch(或 fetch --sample 20)")
    ddir = data_dir()
    ddir.mkdir(parents=True, exist_ok=True)

    books = sorted(p for p in src.iterdir() if (p / "index.html").is_file())
    if not books:
        sys.exit(f"[build] {src} 下没有书籍目录")
    print(f"[build] 解析 {len(books)} 本书 …", file=sys.stderr)

    dbp = db_path()
    if dbp.exists():
        dbp.unlink()
    db = sqlite3.connect(dbp)
    db.executescript("""
        PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;
        CREATE TABLE books(
            book TEXT PRIMARY KEY, author TEXT, dynasty TEXT,
            year INT, year_text TEXT, edition TEXT, quality TEXT,
            n_paras INT, n_chars INT);
        CREATE TABLE paras(
            id INTEGER PRIMARY KEY, book TEXT, seq INT, path TEXT, text TEXT);
        CREATE INDEX idx_paras_book ON paras(book, seq);
        CREATE VIRTUAL TABLE fts USING fts5(
            text, content='paras', content_rowid='id', tokenize='trigram');
    """)

    catalog = []
    corpus_fp = (ddir / CORPUS_NAME).open("w", encoding="utf-8") \
        if args.jsonl else None
    for i, bdir in enumerate(books, 1):
        try:
            rec = parse_book(bdir / "index.html")
        except Exception as e:
            print(f"[build] 解析失败 {bdir.name}: {e}", file=sys.stderr)
            continue
        name = rec["book"]
        n_chars = sum(len(p["text"]) for p in rec["paras"])
        db.execute(
            "INSERT OR REPLACE INTO books VALUES(?,?,?,?,?,?,?,?,?)",
            (name, rec["author"], rec["dynasty"], rec["year"], rec["year_text"],
             rec["edition"], rec["quality"], len(rec["paras"]), n_chars))
        rows = [(name, seq, " / ".join(p["path"]), p["text"])
                for seq, p in enumerate(rec["paras"], 1)]
        db.executemany(
            "INSERT INTO paras(book,seq,path,text) VALUES(?,?,?,?)", rows)
        if corpus_fp:
            for seq, p in enumerate(rec["paras"], 1):
                corpus_fp.write(json.dumps(
                    {"book": name, "seq": seq, "path": p["path"],
                     "text": p["text"]}, ensure_ascii=False) + "\n")
        catalog.append({k: rec[k] for k in
                        ("book", "author", "dynasty", "year", "year_text",
                         "edition", "quality")} | {"n_paras": len(rec["paras"]),
                                                   "n_chars": n_chars})
        if i % 100 == 0:
            print(f"[build] {i}/{len(books)}", file=sys.stderr)
    if corpus_fp:
        corpus_fp.close()

    print("[build] 建立 FTS5 trigram 索引 …", file=sys.stderr)
    db.execute("INSERT INTO fts(rowid, text) SELECT id, text FROM paras")
    db.commit()
    db.close()

    (ddir / CATALOG_NAME).write_text(
        json.dumps(catalog, ensure_ascii=False, indent=1), encoding="utf-8")
    total_chars = sum(c["n_chars"] for c in catalog)
    print(f"[build] 完成:{len(catalog)} 本书,"
          f"{sum(c['n_paras'] for c in catalog)} 段,约 {total_chars // 10000} 万字。"
          f"\n[build] 索引: {dbp}", file=sys.stderr)


# ---------------------------------------------------------------------------
# 查询辅助
# ---------------------------------------------------------------------------

def open_db() -> sqlite3.Connection:
    dbp = db_path()
    if not dbp.exists():
        sys.exit("[error] 索引未构建。请先运行:\n"
                 f"  python3 {SCRIPT_DIR / 'tcm.py'} fetch   # 或 fetch --sample 20\n"
                 f"  python3 {SCRIPT_DIR / 'tcm.py'} build")
    db = sqlite3.connect(dbp)
    db.row_factory = sqlite3.Row
    return db


def norm_query(q: str) -> str:
    return unicodedata.normalize("NFC", q.strip())


def contains_simplified_hint(q: str) -> bool:
    """粗略检测常见简体字,提示调用方改用繁体。"""
    common_simplified = ("医药证辨论谈脉气热湿风经络脏腑针灸剂张学难传补泻虚实后发这为与"
                         "汤归术呕烦满阴阳当参陈两黄内灵冲数从众问闻页贝车东乐书门马鸟龙")
    trad_only = set("醫藥證辨談脈氣熱濕風經絡臟腑針灸劑張學難傳補瀉虛實後發這為與"
                    "湯歸朮嘔煩滿陰陽當參陳兩黃內靈衝數從眾問聞頁貝車東樂書門馬鳥龍")
    hits = [c for c in q if c in common_simplified and c not in trad_only]
    return bool(hits)


def cite(row) -> str:
    path = row["path"] or "(卷首)"
    return f"《{row['book']}》· {path} · 段{row['seq']}"


def cmd_search(args):
    db = open_db()
    q = norm_query(args.query)
    if contains_simplified_hint(q):
        print("[提示] 查询词疑似含简体字;语料为繁体,请将查询词转换为繁体后重试。",
              file=sys.stderr)
    limit = args.limit
    where_book = ""
    params = []
    if args.book:
        where_book = " AND p.book = ?"

    rows = []
    if len(q) >= 3 and not args.like:
        match = '"' + q.replace('"', '""') + '"'
        sql = ("SELECT p.book, p.seq, p.path, p.text, "
               "snippet(fts, 0, '【', '】', '…', 40) AS snip "
               "FROM fts JOIN paras p ON p.id = fts.rowid "
               "WHERE fts MATCH ?" + where_book +
               " ORDER BY rank LIMIT ?")
        params = [match] + ([args.book] if args.book else []) + [limit]
        rows = db.execute(sql, params).fetchall()
    if not rows:  # 短查询或 FTS 无结果时回退 LIKE(慢但全)
        sql = ("SELECT book, seq, path, text FROM paras "
               "WHERE text LIKE ?" + (" AND book = ?" if args.book else "") +
               " LIMIT ?")
        params = [f"%{q}%"] + ([args.book] if args.book else []) + [limit]
        rows = db.execute(sql, params).fetchall()
        rows = [dict(r) | {"snip": highlight(r["text"], q)} for r in rows]

    if not rows:
        print(f"未命中:{q}\n建议:换用繁体/异体写法、缩短查询词、或拆分为多个关键词分别检索。")
        return
    print(f"# 检索「{q}」 命中 {len(rows)} 条(上限 {limit})\n")
    for r in rows:
        print(f"- {cite(r)}\n  {r['snip']}")
    print("\n(用 get 子命令取完整段落,例:"
          f"tcm.py get \"{rows[0]['book']}\" --seq {rows[0]['seq']})")


def highlight(text: str, q: str, width: int = 80) -> str:
    i = text.find(q)
    if i < 0:
        return text[:width]
    s = max(0, i - width // 3)
    e = min(len(text), i + len(q) + width // 2)
    frag = text[s:e].replace(q, f"【{q}】")
    return ("…" if s else "") + frag + ("…" if e < len(text) else "")


def cmd_get(args):
    db = open_db()
    book = norm_query(args.book)
    row = db.execute("SELECT * FROM books WHERE book = ?", (book,)).fetchone()
    if not row:
        cand = db.execute(
            "SELECT book FROM books WHERE book LIKE ? LIMIT 8",
            (f"%{book}%",)).fetchall()
        hint = "、".join(c["book"] for c in cand) or "(无相近书名)"
        sys.exit(f"[error] 无此书:{book}。相近书名:{hint}")

    if args.seq is not None:
        lo, hi = args.seq, args.seq + max(0, args.context)
        lo = max(1, lo - max(0, args.context))
        rows = db.execute(
            "SELECT * FROM paras WHERE book=? AND seq BETWEEN ? AND ? ORDER BY seq",
            (book, lo, hi)).fetchall()
    elif args.section:
        rows = db.execute(
            "SELECT * FROM paras WHERE book=? AND path LIKE ? ORDER BY seq LIMIT ?",
            (book, f"%{args.section}%", args.limit)).fetchall()
    else:
        rows = db.execute(
            "SELECT * FROM paras WHERE book=? ORDER BY seq LIMIT ?",
            (book, args.limit)).fetchall()
    if not rows:
        sys.exit("[error] 未找到匹配段落(检查 --section 篇名是否为繁体、是否存在)")
    src = f"底本:{row['edition']}" if row["edition"] else ""
    print(f"# 《{book}》 {row['dynasty']}·{row['author']} {row['year_text']} {src}\n")
    cur_path = None
    for r in rows:
        if r["path"] != cur_path:
            cur_path = r["path"]
            print(f"\n## {cur_path or '(卷首)'}")
        print(f"[段{r['seq']}] {r['text']}")


def cmd_info(args):
    db = open_db()
    book = norm_query(args.book)
    row = db.execute("SELECT * FROM books WHERE book = ?", (book,)).fetchone()
    if not row:
        cand = db.execute("SELECT book FROM books WHERE book LIKE ? LIMIT 8",
                          (f"%{book}%",)).fetchall()
        hint = "、".join(c["book"] for c in cand) or "(无相近书名)"
        sys.exit(f"[error] 无此书:{book}。相近书名:{hint}")
    print(f"# 《{row['book']}》")
    for k, label in (("author", "作者"), ("dynasty", "朝代"),
                     ("year_text", "年份"), ("edition", "底本"),
                     ("quality", "录文品质"), ("n_paras", "段落数"),
                     ("n_chars", "字数")):
        v = row[k]
        if v:
            print(f"- {label}: {v}")
    print("\n## 目录")
    toc = db.execute(
        "SELECT path, MIN(seq) AS s, COUNT(*) AS n FROM paras "
        "WHERE book=? GROUP BY path ORDER BY s", (book,)).fetchall()
    for t in toc[: args.limit]:
        print(f"- {t['path'] or '(卷首)'}(段{t['s']}起,{t['n']}段)")
    if len(toc) > args.limit:
        print(f"…共 {len(toc)} 个篇章,--limit 调整显示数")


def cmd_list(args):
    db = open_db()
    sql = "SELECT * FROM books WHERE 1=1"
    params = []
    if args.dynasty:
        sql += " AND dynasty = ?"
        params.append(args.dynasty)
    if args.author:
        sql += " AND author LIKE ?"
        params.append(f"%{args.author}%")
    if args.keyword:
        sql += " AND book LIKE ?"
        params.append(f"%{args.keyword}%")
    sql += " ORDER BY year IS NULL, year LIMIT ?"
    params.append(args.limit)
    rows = db.execute(sql, params).fetchall()
    total = db.execute("SELECT COUNT(*) c FROM books").fetchone()["c"]
    print(f"# 书目(库内共 {total} 本;显示 {len(rows)} 本)\n")
    for r in rows:
        print(f"- 《{r['book']}》 {r['dynasty']}·{r['author']} "
              f"{r['year_text']} [{r['n_chars'] // 1000}k 字]")


def cmd_status(args):
    ddir = data_dir()
    dbp = db_path()
    print(f"数据目录: {ddir}")
    print(f"原始语料: {'就绪' if raw_dir().is_dir() else '缺失(运行 fetch)'}")
    if dbp.exists():
        db = sqlite3.connect(dbp)
        n_books, n_paras = db.execute(
            "SELECT (SELECT COUNT(*) FROM books), (SELECT COUNT(*) FROM paras)"
        ).fetchone()
        print(f"索引: 就绪({n_books} 本书,{n_paras} 段,"
              f"{dbp.stat().st_size // 1048576} MB)")
    else:
        print("索引: 缺失(运行 build)")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("fetch", help="拉取语料(默认全量 git 浅克隆)")
    p.add_argument("--sample", type=int, default=0,
                   help="仅抓取前 N 本用于快速验证")
    p.add_argument("--books", nargs="*", help="指定书名(配合 --sample)")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("build", help="解析并建立索引")
    p.add_argument("--jsonl", action="store_true",
                   help="同时导出 corpus.jsonl(供 embedding 等下游使用)")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("search", help="全文检索(繁体)")
    p.add_argument("query")
    p.add_argument("--book", help="限定在某书内检索")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--like", action="store_true", help="强制 LIKE 精确子串匹配")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("get", help="取原文段落")
    p.add_argument("book")
    p.add_argument("--seq", type=int, help="段落号")
    p.add_argument("--context", type=int, default=0, help="上下各多取 N 段")
    p.add_argument("--section", help="按篇章名(模糊)取整节")
    p.add_argument("--limit", type=int, default=40)
    p.set_defaults(func=cmd_get)

    p = sub.add_parser("info", help="书籍元数据与目录")
    p.add_argument("book")
    p.add_argument("--limit", type=int, default=60)
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("list", help="浏览书目")
    p.add_argument("--dynasty", help="朝代,如 漢/唐/宋/金/元/明/清/民國")
    p.add_argument("--author")
    p.add_argument("--keyword", help="书名关键词")
    p.add_argument("--limit", type=int, default=30)
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("status", help="检查语料与索引状态")
    p.set_defaults(func=cmd_status)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
