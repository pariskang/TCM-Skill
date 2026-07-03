"""语料只读访问层。复用 tcm.py 的路径解析与 SQLite 索引,不重复建库。"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from typing import Iterable, Optional

_SCRIPTS = Path(__file__).resolve().parent.parent   # …/scripts
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import tcm  # noqa: E402  复用 data_dir/db_path,保持单一事实源


class Corpus:
    def __init__(self):
        dbp = tcm.db_path()
        if not dbp.exists():
            raise FileNotFoundError(
                f"索引未构建({dbp})。请先运行 tcm.py fetch && tcm.py build")
        self.db = sqlite3.connect(f"file:{dbp}?mode=ro", uri=True)
        self.db.row_factory = sqlite3.Row

    # -- FTS5 短语召回(trigram),返回 (passage, rank) --
    def fts_search(self, term: str, limit: int = 50, book: Optional[str] = None):
        term = term.strip()
        if len(term) < 3:
            return self.like_search(term, limit, book)
        match = '"' + term.replace('"', '""') + '"'
        sql = ("SELECT p.id, p.book, p.seq, p.path, p.text, rank "
               "FROM fts JOIN paras p ON p.id = fts.rowid "
               "WHERE fts MATCH ?")
        params = [match]
        if book:
            sql += " AND p.book = ?"
            params.append(book)
        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)
        try:
            return [dict(r) for r in self.db.execute(sql, params).fetchall()]
        except sqlite3.OperationalError:
            return self.like_search(term, limit, book)

    def like_search(self, term: str, limit: int = 50, book: Optional[str] = None):
        # 2 字词无法走 trigram FTS,用 LIKE。关键:必须确定性且跨书排序,否则退化为
        # "按 rowid 取最早几行",全量语料下核心 2 字词(腎虛/骨痿/腰痛…)会静默漏检
        # 大量书。用窗口函数按书轮取(每本书先出密度最高的一段,再第二段…),
        # 保证跨书覆盖;书内以出现次数(密度)降序 + 短段优先作为无 BM25 时的相关度代理。
        term = term.strip()
        if not term:
            return []
        dens = "(length(text) - length(replace(text, ?, '')))"
        sql = (f"SELECT id, book, seq, path, text FROM ("
               f"  SELECT id, book, seq, path, text, {dens} AS _hits, "
               f"  ROW_NUMBER() OVER (PARTITION BY book "
               f"    ORDER BY {dens} DESC, length(text) ASC) AS _rn "
               f"  FROM paras WHERE text LIKE ?")
        params = [term, term, f"%{term}%"]
        if book:
            sql += " AND book = ?"
            params.append(book)
        sql += ") ORDER BY _rn ASC, _hits DESC LIMIT ?"
        params.append(limit)
        rows = []
        for r in self.db.execute(sql, params).fetchall():
            d = dict(r)
            d["rank"] = 0.0
            rows.append(d)
        return rows

    def get_passage(self, passage_id: int):
        r = self.db.execute(
            "SELECT id, book, seq, path, text FROM paras WHERE id = ?",
            (passage_id,)).fetchone()
        return dict(r) if r else None

    def context(self, book: str, seq: int, span: int = 1):
        rows = self.db.execute(
            "SELECT seq, text FROM paras WHERE book=? AND seq BETWEEN ? AND ? ORDER BY seq",
            (book, seq - span, seq + span)).fetchall()
        return [dict(r) for r in rows]

    def book_meta(self, book: str):
        r = self.db.execute("SELECT * FROM books WHERE book = ?", (book,)).fetchone()
        return dict(r) if r else {}

    def close(self):
        self.db.close()
