"""Task 2 - rank EXISTING recipes against the ingredients a user has.

Score (shown to the user for every result):
    score = W_MATCH * matched  -  W_EXTRA * extra_required  -  W_STEPS * steps (only if 'fewer steps' preferred)
where
    matched         = user ingredients the recipe uses (inclusion rule, see store.inclusion_match)
    unused          = user ingredients the recipe does not use
    extra_required  = recipe ingredients the user did NOT list, excluding assumed staples
    staples         = salt, black pepper, water (assumed available; listed separately)
Ties: fewer extra ingredients, then fewer steps, then recipe ID.
Hard constraints (exclusions, restriction groups, max steps) are filters, never score terms.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .config import SOURCE_NAMES
from .groups import GROUPS, resolve_group
from .normalize import normalize_name, parse_line
from .store import Filter, Store, inclusion_match

STAPLES = {"salt", "black pepper", "water", "ice"}
W_MATCH, W_EXTRA, W_STEPS = 10.0, 2.0, 0.5
CANDIDATE_CAP = 4000


@dataclass
class PantryItem:
    text: str
    name: str
    qty: float | None = None
    unit: str | None = None
    dim: str = "unknown"
    base_qty: float | None = None


def parse_pantry(items: list[str]) -> list[PantryItem]:
    out = []
    for t in items:
        p = parse_line(t)
        if p is None or not p.names:
            nm = normalize_name(t)
            if nm:
                out.append(PantryItem(text=t, name=nm))
            continue
        has_qty = p.qty is not None and any(ch.isdigit() for ch in t)
        out.append(PantryItem(text=t, name=p.names[0], qty=p.qty if has_qty else None,
                              unit=p.unit if has_qty else None, dim=p.dim if has_qty else "unknown",
                              base_qty=p.base_qty if has_qty else None))
    # de-duplicate by name
    seen, uniq = set(), []
    for it in out:
        if it.name not in seen:
            seen.add(it.name)
            uniq.append(it)
    return uniq


@dataclass
class MatchRequest:
    pantry: list[str]
    exclude_ingredients: list[str] = field(default_factory=list)
    exclude_groups: list[str] = field(default_factory=list)
    max_steps: int | None = None
    max_ingredients: int | None = None
    query: str = ""                    # optional dish constraint (e.g. "soup")
    prefer_fewer_steps: bool = False
    k: int = 5


def _quantity_check(store: Store, rid: str, item: PantryItem) -> str:
    if item.qty is None:
        return "name-level match only (no quantity given)"
    names = set(store.names_for(item.name))
    rows = store.ing[(store.ing["recipe_id"] == rid) & (store.ing["name"].isin(names))]
    need = rows.dropna(subset=["qty"])
    if need.empty:
        return "quantity NOT verified (recipe gives no amount)"
    dims = set(need["dim"])
    have_dim = item.dim
    if len(dims) == 1 and have_dim in dims and have_dim in ("count", "mass", "volume"):
        req = need["base_qty"].sum() if have_dim != "count" else need["qty"].sum()
        have = item.base_qty if have_dim != "count" else item.qty
        unit = {"count": "", "mass": " g", "volume": " ml"}[have_dim]
        ok = have >= req - 1e-9
        return (f"{'sufficient' if ok else 'INSUFFICIENT'}: have {have:g}{unit}, recipe needs {req:g}{unit}"
                + ("" if have_dim == "count" else " (converted)"))
    return (f"quantity NOT verified (recipe uses {', '.join(sorted(d for d in dims if d))}, you gave {have_dim}; "
            "no reliable density/size conversion in the dataset)")


def match(store: Store, req: MatchRequest) -> dict:
    pantry = parse_pantry(req.pantry)
    groups = [resolve_group(g) or g for g in req.exclude_groups]
    # pantry items that themselves violate a hard constraint are ignored (and reported)
    ignored = []
    excl_names = {normalize_name(e) for e in req.exclude_ingredients}
    usable = []
    for it in pantry:
        g_hit = next((g for g in groups if g in GROUPS and _violates_group(it.name, g)), None)
        if it.name in excl_names or any(inclusion_match(it.name, e) for e in excl_names if e) or g_hit:
            ignored.append({"item": it.text, "reason": f"conflicts with restriction '{g_hit or it.name}'"})
        else:
            usable.append(it)

    hard = Filter(query=req.query, exclude_ingredients=req.exclude_ingredients, exclude_groups=groups,
                  max_steps=req.max_steps, max_ingredients=req.max_ingredients)
    eligible, audit = store.apply(hard)
    result = {
        "type": "match",
        "pantry": [vars(p) for p in usable],
        "ignored_pantry": ignored,
        "hard_constraints": hard.describe(),
        "eligible_recipes": len(eligible),
        "scoring": (f"score = {W_MATCH:g}×matched − {W_EXTRA:g}×extra_required"
                    + (f" − {W_STEPS:g}×steps" if req.prefer_fewer_steps else "")
                    + "; ties → fewer extra ingredients, fewer steps, recipe ID. "
                    "Staples (salt, black pepper, water) are assumed available and not counted as extra."),
        "matching_rule": "names normalized; a pantry item matches a recipe ingredient with the same head noun "
                         "(e.g. 'rice' matches 'brown rice' but not 'rice vinegar').",
    }
    ranked = _rank(store, usable, eligible, req)
    result["candidates_considered"] = ranked.attrs.get("candidates", 0)
    if len(ranked):
        result["results"] = [_explain(store, r, usable) for r in ranked.head(req.k).itertuples()]
        result["no_match"] = False
        return result

    # no eligible match: say so, then show closest alternatives CLEARLY marked as violating
    result["no_match"] = True
    result["results"] = []
    soft = Filter(query=req.query)
    allr, _ = store.apply(soft)
    alt = _rank(store, usable, allr, req)
    alts = []
    for r in alt.head(req.k).itertuples():
        e = _explain(store, r, usable)
        e["violations"] = _violations(store, r.Index, req, groups)
        alts.append(e)
    result["alternatives_violating_constraints"] = alts
    return result


def _violates_group(name: str, group: str) -> bool:
    from .groups import line_violation
    return line_violation(name, group) is not None


def _rank(store: Store, pantry: list[PantryItem], eligible: pd.DataFrame, req: MatchRequest) -> pd.DataFrame:
    if not pantry or eligible.empty:
        return pd.DataFrame()
    df = eligible.copy()
    cols = []
    for i, it in enumerate(pantry):
        c = f"pm{i}"
        df[c] = df.index.isin(store.recipes_with(it.name))
        cols.append(c)
    df["matched"] = df[cols].sum(axis=1)
    df = df[df["matched"] > 0]
    df.attrs["candidates"] = len(df)
    if df.empty:
        return df
    # bound work: best by matched, then fewest distinct ingredients
    df = df.sort_values(["matched", "n_distinct_ingredients"], ascending=[False, True]).head(CANDIDATE_CAP)
    covered = set()
    for it in pantry:
        covered |= set(store.names_for(it.name))
    rn = store.recipe_names
    extras = []
    for rid in df.index:
        names = rn.get(rid, frozenset())
        extras.append(sorted(n for n in names if n not in covered and n not in STAPLES))
    df["extra"] = extras
    df["n_extra"] = df["extra"].map(len)
    df["score"] = W_MATCH * df["matched"] - W_EXTRA * df["n_extra"]
    if req.prefer_fewer_steps:
        df["score"] -= W_STEPS * df["n_steps"]
    df = df.reset_index().sort_values(["score", "n_extra", "n_steps", "id"],
                                      ascending=[False, True, True, True]).set_index("id")
    df.attrs["candidates"] = int(df.attrs.get("candidates", len(df)))
    return df


def _explain(store: Store, r, pantry: list[PantryItem]) -> dict:
    rid = r.Index
    matched, unused, qty = [], [], {}
    for i, it in enumerate(pantry):
        if getattr(r, f"pm{i}"):
            matched.append(it.name)
            qty[it.name] = _quantity_check(store, rid, it)
        else:
            unused.append(it.name)
    names = store.recipe_names.get(rid, frozenset())
    staples = sorted(n for n in names if n in STAPLES)
    breakdown = f"{W_MATCH:g}×{len(matched)} matched − {W_EXTRA:g}×{r.n_extra} extra"
    if r.score != W_MATCH * len(matched) - W_EXTRA * r.n_extra:
        breakdown += f" − {W_STEPS:g}×{r.n_steps} steps"
    return {
        "id": rid, "title": r.title, "source": SOURCE_NAMES[r.source], "steps": int(r.n_steps),
        "score": round(float(r.score), 2), "score_breakdown": f"{breakdown} = {r.score:g}",
        "matched": matched, "unused_from_your_list": unused, "extra_required": list(r.extra),
        "staples_assumed": staples, "quantity_check": qty,
    }


def _violations(store: Store, rid: str, req: MatchRequest, groups: list[str]) -> list[str]:
    out = []
    ev = store.evidence_lines(rid, req.exclude_ingredients, groups)
    for line, what, term in ev["hits"]:
        label = GROUPS[what]["label"] if what in GROUPS else what
        out.append(f"contains {label}: '{line}'")
    steps = int(store.recipes.at[rid, "n_steps"])
    if req.max_steps is not None and steps > req.max_steps:
        out.append(f"{steps} steps > max {req.max_steps}")
    n_ing = int(store.recipes.at[rid, "n_ingredients"])
    if req.max_ingredients is not None and n_ing > req.max_ingredients:
        out.append(f"{n_ing} ingredients > max {req.max_ingredients}")
    return out or ["(fails the dish/text constraint)"]
