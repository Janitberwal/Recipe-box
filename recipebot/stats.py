"""Deterministic analytics over the COMPLETE filtered recipe set (never LLM-estimated)."""
from __future__ import annotations

import pandas as pd

from .config import SOURCE_NAMES
from .normalize import normalize_name
from .store import Filter, Store

STEP_RULE = ("step = one non-empty line/paragraph of the original instructions, after removing "
             "boilerplate lines (e.g. 'Watch how to make this recipe', photo credits); "
             "ties broken by sentence count, then recipe ID")
INGREDIENT_COUNT_RULE = ("ingredient count = number of ingredient lines after removing 'ADVERTISEMENT' "
                         "markers, blank lines and section headers such as 'For the sauce:'")
NORMALIZATION_RULE = ("ingredient names are normalized (quantities/units/preparation words removed, "
                      "plurals singularized, synonyms merged e.g. 'green onion'->'scallion'); "
                      "a recipe 'contains X' if any ingredient's head noun is X "
                      "(e.g. 'red onion' counts as onion; 'onion powder' and 'green onion' do not)")

METRICS = {"steps": ("n_steps", STEP_RULE), "ingredients": ("n_ingredients", INGREDIENT_COUNT_RULE)}


def _base(store: Store, f: Filter):
    df, audit = store.apply(f)
    return df, {
        "filters": f.describe(),
        "dataset_total": store.total,
        "analyzed": len(df),
        "filter_trace": audit,
    }


def count(store: Store, f: Filter) -> dict:
    df, info = _base(store, f)
    n = len(df)
    info.update({
        "type": "count",
        "count": n,
        "denominator": store.total,
        "percentage": round(100.0 * n / store.total, 2) if store.total else 0.0,
        "calculation": f"count of recipes matching all filters / {store.total:,} usable recipes × 100",
        "normalization": NORMALIZATION_RULE,
    })
    # sensitivity: how matching names were expanded
    expansions = {}
    for q in f.include_all + f.include_any:
        names = store.names_for(q)
        counts = store.ing[store.ing["name"].isin(names)].groupby("name", observed=True)["recipe_id"].nunique()
        expansions[q] = counts.sort_values(ascending=False).head(8).to_dict()
    if expansions:
        info["name_expansion_top"] = expansions
    info["sensitivity"] = _broad_count(store, f)
    info["by_source"] = df["source"].map(SOURCE_NAMES).value_counts().to_dict()
    return info


def _broad_count(store: Store, f: Filter) -> dict | None:
    """Same filter but with a BROAD rule: any whole-word mention in the ingredient line
    (so 'garlic powder' counts as garlic). Shows how much the definition matters."""
    if not (f.include_all or f.include_any):
        return None
    df, _ = store.apply(Filter(**{**f.to_dict(), "include_all": [], "include_any": []}))
    ids = set(df.index)
    for q in f.include_all:
        ids &= set(store.recipes_mentioning(q))
    if f.include_any:
        anyids = set()
        for q in f.include_any:
            anyids |= set(store.recipes_mentioning(q))
        ids &= anyids
    return {"broad_rule": "whole-word mention anywhere in an ingredient line (includes powders, green onion, etc.)",
            "broad_count": len(ids),
            "broad_percentage": round(100.0 * len(ids) / store.total, 2)}


def describe(store: Store, f: Filter, metric: str = "ingredients", group_by_source: bool = False) -> dict:
    col, rule = METRICS[metric]
    df, info = _base(store, f)
    s = df[col]
    info.update({
        "type": "describe", "metric": metric, "rule": rule,
        "mean": round(float(s.mean()), 2) if len(s) else None,
        "median": float(s.median()) if len(s) else None,
        "std": round(float(s.std()), 2) if len(s) > 1 else None,
        "min": int(s.min()) if len(s) else None,
        "max": int(s.max()) if len(s) else None,
        "calculation": f"mean/median of `{col}` over all {len(s):,} filtered recipes (pandas)",
    })
    if group_by_source and len(df):
        g = df.groupby("source")[col].agg(["count", "mean", "median"]).round(2)
        g.index = g.index.map(SOURCE_NAMES)
        info["by_source"] = g.reset_index().rename(columns={"source": "source"}).to_dict("records")
    return info


def rank(store: Store, f: Filter, metric: str = "steps", order: str = "asc", k: int = 5) -> dict:
    col, rule = METRICS[metric]
    df, info = _base(store, f)
    asc = order != "desc"
    tie = "n_sentences" if metric == "steps" else "n_distinct_ingredients"
    d = df.reset_index().sort_values([col, tie, "id"], ascending=[asc, asc, True]).head(k)
    info.update({
        "type": "rank", "metric": metric, "order": "fewest first" if asc else "most first", "rule": rule,
        "results": [{"id": r.id, "title": r.title, "source": SOURCE_NAMES[r.source], col: int(getattr(r, col)),
                     "n_sentences": int(r.n_sentences), "n_ingredients": int(r.n_ingredients)}
                    for r in d.itertuples()],
        "calculation": f"sorted all {len(df):,} filtered recipes by `{col}` ({'ascending' if asc else 'descending'}), "
                       f"tie-break `{tie}` then ID",
    })
    if len(df):
        val_at_k = d[col].iloc[-1]
        info["ties_note"] = (f"{int((df[col] == val_at_k).sum()):,} recipes share the value {int(val_at_k)} "
                             f"at position {len(d)}; tie-break decides which are shown")
    return info


STAPLE_NAMES = {"salt", "black pepper", "water"}


def top_ingredients(store: Store, f: Filter, k: int = 10, include_staples: bool = True) -> dict:
    df, info = _base(store, f)
    sub = store.ing[store.ing["recipe_id"].isin(df.index) & (store.ing["name"] != "")]
    counts = sub.groupby("name", observed=True)["recipe_id"].nunique().sort_values(ascending=False)
    if not include_staples:
        counts = counts[~counts.index.isin(STAPLE_NAMES)]
    n = len(df)
    info.update({
        "type": "top_ingredients",
        "results": [{"ingredient": nm, "recipes": int(c), "percent_of_filtered": round(100 * c / n, 1) if n else 0}
                    for nm, c in counts.head(k).items()],
        "calculation": f"for each normalized ingredient name, number of distinct recipes (out of {n:,}) containing it",
        "normalization": NORMALIZATION_RULE,
        "staples_excluded": None if include_staples else sorted(STAPLE_NAMES),
    })
    return info


def ingredient_lookup(store: Store, name: str) -> dict:
    """How a user term is normalized and which dataset names it covers."""
    q = normalize_name(name)
    names = store.names_for(name)
    counts = store.ing[store.ing["name"].isin(names)].groupby("name", observed=True)["recipe_id"].nunique()
    return {"input": name, "normalized": q, "matching_names": counts.sort_values(ascending=False).head(15).to_dict(),
            "recipes": len(store.recipes_with(name))}
