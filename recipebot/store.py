"""In-memory access layer over data/recipes.db (pandas + SQLite)."""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field, asdict
from functools import lru_cache

import pandas as pd

from .config import DB_PATH, SOURCE_NAMES
from .groups import GROUPS, members, resolve_group
from .normalize import name_matches, normalize_name, word_pattern

# words that describe a cut/part; "chicken breast" counts as "chicken" for inclusion
CUT_WORDS = {"breast", "thigh", "leg", "wing", "drumstick", "tender", "tenderloin", "fillet",
             "filet", "chop", "loin", "shoulder", "cutlet", "meat", "half", "piece", "quarter",
             "steak", "rib", "strip", "clove", "floret", "stalk", "leaf", "sprig", "wedge",
             "segment", "spear", "kernel", "head", "bulb"}


def inclusion_match(ing_name: str, query: str) -> bool:
    """Inclusion rule = head-noun match after stripping a trailing cut/part word.
    'chicken breast' ~ chicken, 'red onion' ~ onion, 'onion powder' !~ onion."""
    if name_matches(ing_name, query):
        return True
    toks = ing_name.split()
    if len(toks) >= 2 and toks[-1] in CUT_WORDS:
        return name_matches(" ".join(toks[:-1]), query)
    return False


@dataclass
class Filter:
    query: str = ""                                     # dish / keywords (full-text search)
    include_all: list[str] = field(default_factory=list)  # every one of these ingredients
    include_any: list[str] = field(default_factory=list)  # at least one of these
    exclude_ingredients: list[str] = field(default_factory=list)  # hard: never present
    exclude_groups: list[str] = field(default_factory=list)       # hard: dairy, egg, vegan...
    max_steps: int | None = None
    min_steps: int | None = None
    max_ingredients: int | None = None
    sources: list[str] = field(default_factory=list)    # ar / epi / fn

    def describe(self) -> str:
        parts = []
        if self.query:
            parts.append(f'text matches "{self.query}" (title/ingredients)')
        if self.include_all:
            parts.append("contains ALL of: " + ", ".join(self.include_all))
        if self.include_any:
            parts.append("contains ANY of: " + ", ".join(self.include_any))
        if self.exclude_ingredients:
            parts.append("excludes: " + ", ".join(self.exclude_ingredients))
        if self.exclude_groups:
            parts.append("restriction: no " + ", ".join(GROUPS[g]["label"] for g in self.exclude_groups))
        if self.max_steps is not None:
            parts.append(f"≤ {self.max_steps} steps")
        if self.min_steps is not None:
            parts.append(f"≥ {self.min_steps} steps")
        if self.max_ingredients is not None:
            parts.append(f"≤ {self.max_ingredients} ingredient lines")
        if self.sources:
            parts.append("source: " + ", ".join(SOURCE_NAMES.get(s, s) for s in self.sources))
        return "; ".join(parts) if parts else "no filters (whole dataset)"

    def to_dict(self):
        return asdict(self)


class Store:
    def __init__(self, db_path=DB_PATH):
        self.con = sqlite3.connect(db_path, check_same_thread=False)
        self.recipes = pd.read_sql(
            "SELECT id, source, title, n_steps, n_sentences, n_ingredients, n_distinct_ingredients FROM recipes",
            self.con).set_index("id")
        ing = pd.read_sql("SELECT recipe_id, pos, name, qty, unit, dim, base_qty FROM ingredients", self.con)
        ing["name"] = ing["name"].astype("category")
        ing["recipe_id"] = ing["recipe_id"].astype("category")
        self.ing = ing
        self.names = pd.Series(ing["name"].cat.categories)
        flags = pd.read_sql("SELECT recipe_id, grp FROM flags", self.con)
        self.flag_sets = {g: set(df["recipe_id"]) for g, df in flags.groupby("grp")}
        self.build_stats = json.loads(self.con.execute("SELECT value FROM meta WHERE key='build_stats'").fetchone()[0])
        self.total = len(self.recipes)
        self._recipe_names = None

    # ------------------------------------------------------------------ helpers
    @property
    def recipe_names(self) -> dict:
        """recipe_id -> frozenset of normalized names (lazy)."""
        if self._recipe_names is None:
            d: dict[str, set] = {}
            for rid, nm in zip(self.ing["recipe_id"].astype(str).tolist(), self.ing["name"].astype(str).tolist()):
                if nm:
                    d.setdefault(rid, set()).add(nm)
            self._recipe_names = {k: frozenset(v) for k, v in d.items()}
        return self._recipe_names

    @lru_cache(maxsize=512)
    def names_for(self, query: str) -> tuple[str, ...]:
        """All normalized ingredient names that count as `query` under the inclusion rule."""
        q = normalize_name(query) or query.lower().strip()
        return tuple(n for n in self.names if n and inclusion_match(n, q))

    @lru_cache(maxsize=512)
    def recipes_with(self, query: str) -> frozenset:
        names = self.names_for(query)
        return frozenset(self.ing.loc[self.ing["name"].isin(names), "recipe_id"].astype(str).unique())

    @lru_cache(maxsize=256)
    def recipes_mentioning(self, term: str) -> frozenset:
        """Conservative (exclusion) rule: whole-word, plural-tolerant mention in any ingredient line."""
        g = resolve_group(term)
        if g:
            ids = set()
            for m in members(g):
                ids |= self.flag_sets.get(m, set())
            return frozenset(ids)
        pat = word_pattern(normalize_name(term) or term)
        first = (normalize_name(term) or term).split()[-1][:5]
        rows = self.con.execute("SELECT recipe_id, line FROM ingredients WHERE line LIKE ?", (f"%{first}%",)).fetchall()
        ids = {rid for rid, line in rows if pat.search(line)}
        # also catch the raw term (e.g. "peanuts")
        pat2 = word_pattern(term)
        ids |= {rid for rid, line in rows if pat2.search(line)}
        return frozenset(ids)

    def group_ids(self, group: str) -> set:
        ids = set()
        for m in members(group):
            ids |= self.flag_sets.get(m, set())
        return ids

    # ------------------------------------------------------------------ text search
    def fts(self, query: str) -> pd.Series:
        """Full-text search. Returns bm25 score (lower = better) indexed by recipe id.
        Tries: all terms in title/ingredients -> all terms anywhere -> any term."""
        toks = [t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) > 1 and t not in _FTS_STOP]
        if not toks:
            return pd.Series(dtype=float)
        attempts = [
            "{title ingredients} : (" + " AND ".join(f'"{t}"' for t in toks) + ")",
            " AND ".join(f'"{t}"' for t in toks),
            " OR ".join(f'"{t}"' for t in toks),
        ]
        for i, q in enumerate(attempts):
            rows = self.con.execute(
                "SELECT id, bm25(recipes_fts, 0, 10.0, 3.0, 0.5) AS score FROM recipes_fts WHERE recipes_fts MATCH ?",
                (q,)).fetchall()
            if rows:
                s = pd.Series({r[0]: r[1] for r in rows})
                s.attrs["mode"] = ["all terms in title/ingredients", "all terms anywhere (incl. instructions)",
                                   "ANY term (relaxed)"][i]
                return s
        return pd.Series(dtype=float)

    # ------------------------------------------------------------------ filtering
    def apply(self, f: Filter) -> tuple[pd.DataFrame, dict]:
        """Return the COMPLETE filtered recipe table and an audit dict."""
        df = self.recipes
        audit = {"start": len(df)}
        if f.query:
            s = self.fts(f.query)
            df = df.loc[df.index.intersection(s.index)].copy()
            df["score"] = s.reindex(df.index)
            audit["text_mode"] = s.attrs.get("mode", "no match")
            audit["after_text"] = len(df)
        for q in f.include_all:
            df = df[df.index.isin(self.recipes_with(q))]
        if f.include_all:
            audit["after_include_all"] = len(df)
        if f.include_any:
            ids = set()
            for q in f.include_any:
                ids |= self.recipes_with(q)
            df = df[df.index.isin(ids)]
            audit["after_include_any"] = len(df)
        excluded = set()
        for t in f.exclude_ingredients:
            excluded |= self.recipes_mentioning(t)
        for g in f.exclude_groups:
            excluded |= self.group_ids(g)
        if excluded:
            df = df[~df.index.isin(excluded)]
            audit["after_exclusions"] = len(df)
        if f.max_steps is not None:
            df = df[df["n_steps"] <= f.max_steps]
        if f.min_steps is not None:
            df = df[df["n_steps"] >= f.min_steps]
        if f.max_ingredients is not None:
            df = df[df["n_ingredients"] <= f.max_ingredients]
        if f.sources:
            df = df[df["source"].isin(f.sources)]
        audit["final"] = len(df)
        return df, audit

    # ------------------------------------------------------------------ records
    def get(self, rid: str) -> dict | None:
        row = self.con.execute(
            "SELECT id, source, title, ingredients_raw, ingredients, instructions, steps, n_steps, n_ingredients "
            "FROM recipes WHERE id = ?", (rid,)).fetchone()
        if not row:
            return None
        keys = ["id", "source", "title", "ingredients_raw", "ingredients", "instructions", "steps", "n_steps", "n_ingredients"]
        r = dict(zip(keys, row))
        for k in ("ingredients_raw", "ingredients", "steps"):
            r[k] = json.loads(r[k])
        r["source_name"] = SOURCE_NAMES[r["source"]]
        parsed = self.con.execute("SELECT pos, line, name, qty, unit, dim, base_qty FROM ingredients WHERE recipe_id=? ORDER BY pos",
                                  (rid,)).fetchall()
        r["parsed"] = [dict(zip(["pos", "line", "name", "qty", "unit", "dim", "base_qty"], p)) for p in parsed]
        return r

    def evidence_lines(self, rid: str, terms: list[str], groups: list[str]) -> dict:
        """For display: which lines of `rid` hit any exclusion (should be none for eligible recipes)."""
        from .groups import line_violation
        rec = self.get(rid)
        hits = []
        for p in rec["parsed"]:
            for g in groups:
                h = line_violation(p["line"], g)
                if h:
                    hits.append((p["line"], g, h))
            for t in terms:
                if word_pattern(t).search(p["line"]) or word_pattern(normalize_name(t) or t).search(p["line"]):
                    hits.append((p["line"], t, t))
        return {"lines_checked": len({p["pos"] for p in rec["parsed"]}), "hits": hits}


_FTS_STOP = {"a", "an", "the", "recipe", "recipes", "for", "with", "and", "or", "of", "to", "in",
             "some", "find", "me", "show", "dish", "make", "how", "i", "want", "easy", "good", "best"}


@lru_cache(maxsize=1)
def get_store() -> Store:
    return Store()
