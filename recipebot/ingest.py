"""Build data/recipes.db from the three Recipe Box JSON files.

Run:  python -m recipebot.ingest

Tables
  recipes      one row per usable recipe (stable id = original dataset key)
  ingredients  one row per parsed ingredient name (recipe_id, pos, line, name, qty, unit, ...)
  flags        restriction-group hits (recipe_id, grp, pos, term) used for hard filters
  recipes_fts  SQLite FTS5 index (title, ingredient text, instructions) with BM25 ranking
  meta         build statistics (raw counts, dropped records and why)
"""
from __future__ import annotations

import json
import re
import sqlite3
import time

import pandas as pd

from .config import DB_PATH, RAW_DIR, SOURCES
from .groups import BASE_GROUPS, line_violation
from .normalize import clean_line, is_header, parse_line

BOILERPLATE = re.compile(
    r"^(watch how to make this recipe|photograph(s)? by|photo(s)? by|copyright|©|"
    r"recipe courtesy|reprinted (with|from)|excerpted from|adapted from|cook'?s note|"
    r"from .{0,80} by .{0,80}$)",
    re.I,
)
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def split_steps(instructions: str) -> list[str]:
    """STEP RULE: a step is one non-empty line (paragraph) of the original instructions,
    after removing boilerplate lines (video links, photo credits, copyright, notes)."""
    steps = []
    for ln in (instructions or "").split("\n"):
        ln = ln.strip()
        if ln and not BOILERPLATE.match(ln):
            steps.append(ln)
    return steps


def count_sentences(steps: list[str]) -> int:
    return sum(len([s for s in _SENT.split(st) if s.strip()]) for st in steps)


def build(db_path=DB_PATH) -> dict:
    t0 = time.time()
    rec_rows, ing_rows, flag_rows = [], [], []
    stats = {"raw_records": 0, "dropped_empty_record": 0, "dropped_no_title": 0,
             "dropped_no_ingredients": 0, "dropped_no_instructions": 0, "per_source": {}}

    for src, fname in SOURCES.items():
        data = json.load(open(RAW_DIR / fname, encoding="utf-8"))
        kept = 0
        for key, rec in data.items():
            stats["raw_records"] += 1
            if not rec:
                stats["dropped_empty_record"] += 1
                continue
            title = (rec.get("title") or "").strip()
            if not title:
                stats["dropped_no_title"] += 1
                continue
            raw_ings = rec.get("ingredients") or []
            display = [clean_line(x) for x in raw_ings]
            display = [x for x in display if x]
            if not any(not is_header(x) for x in display):
                stats["dropped_no_ingredients"] += 1
                continue
            instructions = rec.get("instructions") or ""
            steps = split_steps(instructions)
            if not steps:
                stats["dropped_no_instructions"] += 1
                continue

            names = set()
            n_lines = 0
            for pos, line in enumerate(display):
                p = parse_line(line)
                if p is None:
                    continue
                n_lines += 1
                for nm in p.names or [""]:
                    ing_rows.append((key, pos, p.line, nm, p.qty, p.unit, p.dim, p.base_qty))
                    if nm:
                        names.add(nm)
                for g in BASE_GROUPS:
                    hit = line_violation(p.line, g)
                    if hit:
                        flag_rows.append((key, g, pos, hit.lower()))

            rec_rows.append({
                "id": key, "source": src, "title": title,
                "ingredients_raw": json.dumps(raw_ings, ensure_ascii=False),
                "ingredients": json.dumps(display, ensure_ascii=False),
                "instructions": instructions,
                "steps": json.dumps(steps, ensure_ascii=False),
                "n_steps": len(steps), "n_sentences": count_sentences(steps),
                "n_ingredients": n_lines, "n_distinct_ingredients": len(names),
                "picture_link": rec.get("picture_link"),
            })
            kept += 1
        stats["per_source"][src] = {"raw": len(data), "kept": kept}

    stats["usable_recipes"] = len(rec_rows)
    recipes = pd.DataFrame(rec_rows)
    ings = pd.DataFrame(ing_rows, columns=["recipe_id", "pos", "line", "name", "qty", "unit", "dim", "base_qty"])
    flags = pd.DataFrame(flag_rows, columns=["recipe_id", "grp", "pos", "term"])

    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(db_path)
    recipes.to_sql("recipes", con, index=False)
    ings.to_sql("ingredients", con, index=False)
    flags.to_sql("flags", con, index=False)
    con.execute("CREATE INDEX ix_ing_name ON ingredients(name)")
    con.execute("CREATE INDEX ix_ing_rid ON ingredients(recipe_id)")
    con.execute("CREATE INDEX ix_flags ON flags(grp, recipe_id)")
    con.execute("CREATE UNIQUE INDEX ix_rec_id ON recipes(id)")

    con.execute("CREATE VIRTUAL TABLE recipes_fts USING fts5(id UNINDEXED, title, ingredients, instructions, tokenize='porter unicode61')")
    ing_text = ings.groupby("recipe_id")["line"].apply(lambda s: " ; ".join(dict.fromkeys(s)))
    fts = pd.DataFrame({
        "id": recipes["id"], "title": recipes["title"],
        "ingredients": recipes["id"].map(ing_text).fillna(""),
        "instructions": recipes["instructions"],
    })
    con.executemany("INSERT INTO recipes_fts(id, title, ingredients, instructions) VALUES (?,?,?,?)",
                    fts.itertuples(index=False, name=None))
    stats["build_seconds"] = round(time.time() - t0, 1)
    con.execute("CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT)")
    con.execute("INSERT INTO meta VALUES ('build_stats', ?)", (json.dumps(stats),))
    con.commit()
    con.close()
    return stats


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
