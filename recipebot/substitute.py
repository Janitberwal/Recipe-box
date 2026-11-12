"""Task 3 - evidence-based ingredient substitution.

Evidence comes from the LOCAL dataset first:
  1. Similar-dish pool: recipes whose titles share the dish words of the original recipe
     (e.g. "banana bread"), most ingredient-similar first (Jaccard on normalized names).
  2. Split the pool into recipes WITH the target ingredient and WITHOUT it.
  3. For each candidate substitute: how many similar recipes WITHOUT the target use it
     (direct support), and how much more common it is without vs. with the target (lift).
  4. "Signal" recipes anywhere in the dataset whose title says e.g. "eggless", "dairy-free",
     "vegan" and that use the candidate (explicit substitution evidence).
Candidate names / ratios / predicted effects come from a curated role table that is
clearly labelled as general culinary knowledge (substitutes_table.py).
Candidates that violate the user's restrictions or do not occur in the dataset are rejected.
"""
from __future__ import annotations

import re

from .groups import GROUPS, line_violation, resolve_group
from .normalize import normalize_name, word_pattern
from .store import Store, inclusion_match
from .substitutes_table import find_entry

TITLE_STOP = {"recipe", "easy", "best", "the", "a", "an", "and", "with", "of", "in", "my", "our", "quick",
              "simple", "homemade", "classic", "ii", "iii", "iv", "i", "s", "mom", "grandma", "perfect",
              "ultimate", "favorite", "delicious", "healthy", "old", "fashioned", "style", "fresh",
              "famous", "world", "for", "on", "to", "from", "or", "no", "chef", "john"}
BAKED_WORDS = {"bread", "cake", "muffin", "muffins", "cookie", "cookies", "brownie", "brownies", "pie",
               "loaf", "scone", "scones", "biscuit", "biscuits", "pancake", "pancakes", "waffle",
               "waffles", "cupcake", "cupcakes", "tart", "cobbler", "crisp", "bars", "roll", "rolls"}
BAKED_INGREDIENTS = {"baking powder", "baking soda", "yeast", "active dry yeast", "flour"}
SIGNAL_WORDS = {
    "egg": ["eggless", "egg-free", "egg free", "vegan", "no egg", "without eggs"],
    "cream": ["dairy-free", "dairy free", "non-dairy", "vegan", "lactose-free", "creamless"],
    "butter": ["dairy-free", "dairy free", "vegan", "butterless", "non-dairy"],
    "milk": ["dairy-free", "dairy free", "non-dairy", "vegan", "lactose-free"],
    "sour cream": ["dairy-free", "vegan", "non-dairy"],
    "cheese": ["dairy-free", "vegan", "cheeseless", "non-dairy"],
    "flour": ["gluten-free", "gluten free", "flourless", "grain-free"],
    "sugar": ["sugar-free", "sugarless", "no sugar", "refined sugar-free"],
    "peanut butter": ["peanut-free", "nut-free"],
    "soy sauce": ["soy-free", "gluten-free"],
}
NOT_TARGET = {
    "cream": {"sour cream", "ice cream", "coconut cream", "cashew cream", "whipped cream", "cream cheese"},
    "butter": {"peanut butter", "almond butter", "apple butter", "cocoa butter", "cashew butter",
               "nut butter", "sunflower butter", "sunflower seed butter", "coconut butter"},
    "milk": {"coconut milk", "almond milk", "soy milk", "rice milk", "oat milk", "cashew milk"},
    "flour": {"rice flour", "almond flour", "coconut flour", "oat flour", "chickpea flour", "corn flour",
              "tapioca flour", "potato flour", "buckwheat flour"},
}
STRENGTH_ORDER = {"strong": 3, "moderate": 2, "weak": 1, "none": 0}


def _title_fts(store: Store, tokens: list[str]) -> set[str]:
    if not tokens:
        return set()
    q = "title : (" + " AND ".join(f'"{t}"' for t in tokens) + ")"
    return {r[0] for r in store.con.execute("SELECT id FROM recipes_fts WHERE recipes_fts MATCH ?", (q,))}


def _signal_ids(store: Store, key: str) -> set[str]:
    ids = set()
    for w in SIGNAL_WORDS.get(key, []):
        toks = re.findall(r"[a-z]+", w)
        q = "title : " + " ".join(f'"{t}"' for t in toks) if len(toks) == 1 else 'title : "' + " ".join(toks) + '"'
        ids |= {r[0] for r in store.con.execute("SELECT id FROM recipes_fts WHERE recipes_fts MATCH ?", (q,))}
    return ids


def _target_names(store: Store, target: str, entry_key, entry) -> set[str]:
    terms = entry["match"] if entry else [target]
    names = set()
    for t in terms:
        names |= set(store.names_for(t))
    names -= NOT_TARGET.get(entry_key, set())
    if entry:
        for c in entry["candidates"]:
            names -= set(store.names_for(c["name"]))
    return names


def substitute(store: Store, rid: str, target: str, restrict_groups: list[str] | None = None,
               exclude_ingredients: list[str] | None = None, dataset_only: bool = True) -> dict:
    """Substitute `target` in recipe `rid`. If restrictions are active, every OTHER ingredient line
    that violates them is also adapted (secondary substitutions) or explicitly removed."""
    restrict_groups = [resolve_group(g) or g for g in (restrict_groups or [])]
    exclude_ingredients = exclude_ingredients or []
    rec = store.get(rid)
    if rec is None:
        return {"type": "substitute", "status": "error", "message": f"Recipe ID {rid} not found."}

    group_target = resolve_group(target)
    if group_target and not find_entry(normalize_name(target) or target)[0]:
        # "make it dairy-free": restriction becomes active, all violating lines are targets
        if group_target not in restrict_groups:
            restrict_groups.append(group_target)
        lines = [p for p in rec["parsed"] if line_violation(p["line"], group_target)]
        if not lines:
            return {"type": "substitute", "status": "not_present", "recipe": _rec_summary(rec), "target": target,
                    "message": f"No {GROUPS[group_target]['label']} ingredient found in {rid}; nothing to substitute."}
        target = lines[0]["name"] or lines[0]["line"]

    out = _analyze(store, rec, target, restrict_groups, exclude_ingredients, dataset_only)
    out["restrictions"] = [GROUPS[g]["label"] for g in restrict_groups if g in GROUPS] + exclude_ingredients
    if "chosen" not in out:
        return out

    # ---- secondary: other lines that still violate restrictions
    handled = set(out["target_pos"])
    secondary = []
    for p in rec["parsed"]:
        if p["pos"] in handled:
            continue
        hit = next((g for g in restrict_groups if g in GROUPS and line_violation(p["line"], g)), None)
        hit = hit or next((e for e in exclude_ingredients if word_pattern(e).search(p["line"])), None)
        if not hit:
            continue
        handled.add(p["pos"])
        sub = _analyze(store, rec, p["name"] or p["line"], restrict_groups, exclude_ingredients, dataset_only,
                       only_pos={p["pos"]})
        if "chosen" in sub:
            secondary.append({"line": p["line"], "pos": p["pos"], "action": "substitute", "chosen": sub["chosen"],
                              "status": sub["status"], "evidence_pool": sub["evidence_pool"],
                              "entry_key": sub.get("entry_key")})
        else:
            secondary.append({"line": p["line"], "pos": p["pos"], "action": "remove",
                              "reason": f"violates restriction ({hit}); no supported substitute in the dataset"})
    out["secondary"] = secondary
    replacements = [(out["target_pos"], out["chosen"], out.get("entry_key"))]
    replacements += [([s["pos"]], s["chosen"], s.get("entry_key")) for s in secondary if s["action"] == "substitute"]
    removals = {s["pos"] for s in secondary if s["action"] == "remove"}
    out["adapted"] = adapt(rec, replacements, removals)
    out["adapted_violations"] = check_adapted(out["adapted"]["ingredients"], restrict_groups, exclude_ingredients)
    if any(s["action"] == "remove" for s in secondary) and out["status"] == "supported":
        out["status"] = "low_confidence"
    out["safety_note"] = ("Based on recipe text only. This is not a medical or allergen-safety assurance: "
                          "check product labels (e.g. hidden dairy/egg in processed foods) and cross-contact.")
    return out


def _rec_summary(rec):
    return {"id": rec["id"], "title": rec["title"], "source": rec["source_name"],
            "ingredients": rec["ingredients"], "steps": rec["steps"]}


GENERIC_TITLE_WORDS = {"cups", "cup", "bites", "bite", "squares", "sticks", "rounds", "mini", "little", "pan",
                       "sheet", "bowl", "bowls", "platter", "jar", "skillet", "baked", "grilled", "roasted",
                       "fried", "slow", "cooker", "instant", "pot", "one", "night", "day", "party", "holiday",
                       "christmas", "thanksgiving", "summer", "winter", "spring", "fall", "italian", "french",
                       "mexican", "southern", "texas", "country", "rich", "creamy", "spicy", "sweet", "hot"}


def _analyze(store, rec, target, restrict_groups, exclude_ingredients, dataset_only, only_pos=None) -> dict:
    rid = rec["id"]
    entry_key, entry = find_entry(normalize_name(target) or target)
    tnames = _target_names(store, target, entry_key, entry)
    if only_pos is not None:
        target_lines = [p for p in rec["parsed"] if p["pos"] in only_pos]
    else:
        target_lines = [p for p in rec["parsed"] if p["name"] in tnames]
    out = {"type": "substitute", "recipe": _rec_summary(rec), "target": target, "entry_key": entry_key}
    if not target_lines:
        out.update(status="not_present",
                   message=f"'{target}' does not appear in the ingredient list of {rid}; nothing to substitute.")
        return out
    out["target_lines"] = [p["line"] for p in target_lines]
    out["target_pos"] = sorted({p["pos"] for p in target_lines})

    rec_names = {p["name"] for p in rec["parsed"] if p["name"]}
    title_toks = [t for t in re.findall(r"[a-z]+", rec["title"].lower()) if t not in TITLE_STOP and len(t) > 2]
    context = "baked" if (set(title_toks) & BAKED_WORDS or len(rec_names & BAKED_INGREDIENTS) >= 2) else "savory"
    out["context"] = context
    roles = (entry["roles"].get(context) or entry["roles"].get("any") or next(iter(entry["roles"].values()))) if entry else []
    out["roles"] = {"roles": roles, "source": "curated culinary-role table (general knowledge, not dataset evidence)"}

    # core-ingredient warning (e.g. replacing eggs in a frittata)
    tw = re.findall(r"[a-z]+", target.lower())
    tgt_words = set(tw) | {w.rstrip("s") for w in tw}
    core = bool(tgt_words & ({t.rstrip("s") for t in title_toks} | set(title_toks)))
    total_count = sum((p["qty"] or 0) for p in target_lines if p["dim"] == "count")
    if entry_key == "egg" and total_count >= 4 and context != "baked":
        core = True
    out["core_ingredient_warning"] = core

    # ---------- similar-dish pool
    dish_toks = [t for t in title_toks if t not in tgt_words and t not in GENERIC_TITLE_WORDS] or \
                [t for t in title_toks if t not in tgt_words]
    tiers = [(dish_toks, "all dish words in title")]
    if len(dish_toks) > 2:
        for j in range(len(dish_toks) - 1, 0, -1):
            tiers.append((dish_toks[j - 1:j + 1], "two consecutive dish words in title"))
    for t in reversed(dish_toks):
        tiers.append(([t], "single dish word in title"))
    pool, tier = set(), "none"
    rn = store.recipe_names
    for toks, label in tiers:
        cand = _title_fts(store, toks) - {rid}
        if context == "baked":   # keep only baked recipes (drops e.g. "banana bread French toast")
            cand = {i for i in cand if len(rn.get(i, frozenset()) & BAKED_INGREDIENTS) >= 2}
        if len(cand) >= 15:
            pool, tier = cand, f"{label}: {' '.join(toks)}"
            break
        if len(cand) > len(pool):
            pool, tier = cand, f"{label}: {' '.join(toks)} (small pool)"
    sims = sorted(((len(rec_names & rn.get(i, frozenset())) / max(1, len(rec_names | rn.get(i, frozenset()))), i)
                   for i in pool), reverse=True)
    pool_ids = [i for _, i in sims[:400]]
    target_ids = set(store.ing.loc[store.ing["name"].isin(tnames), "recipe_id"].astype(str).unique())
    if only_pos is not None and not tnames:
        target_ids = set(store.recipes_mentioning(target))
    with_t = [i for i in pool_ids if i in target_ids]
    without_t = [i for i in pool_ids if i not in target_ids]
    signal = _signal_ids(store, entry_key) - target_ids if entry_key else set()
    out["evidence_pool"] = {"rule": tier, "similar_recipes": len(pool_ids), "with_target": len(with_t),
                            "without_target": len(without_t), "signal_recipes": len(signal),
                            "similarity": "Jaccard overlap of normalized ingredient names (top 400 kept)"}

    cands = [_evaluate(store, c, rec_names, with_t, without_t, signal, restrict_groups, exclude_ingredients,
                       dataset_only, context) for c in (entry["candidates"] if entry else [])]
    out["data_suggested"] = _data_suggested(store, rec_names, with_t, without_t, restrict_groups, exclude_ingredients)
    # ranking: eligible > fits dish type > not already in recipe > not conflicting > not generic > strength > support
    cands.sort(key=lambda c: (c["status"] == "eligible", c["context_fit"], not c["already_in_recipe"],
                              not c["conflicting"], not c["generic"], STRENGTH_ORDER[c["strength"]],
                              c["direct_support"], c["signal_support"]), reverse=True)
    out["candidates"] = cands
    eligible = [c for c in cands if c["status"] == "eligible" and c["strength"] != "none"]
    if not entry:
        out.update(status="unsupported",
                   message=f"No curated substitution knowledge for '{target}', and the dataset alone only shows "
                           "co-occurrence (see data-suggested list), which is not enough to recommend a reliable swap.")
        return out
    if not eligible:
        out.update(status="unsupported",
                   message="No substitute that respects your restrictions has supporting evidence in the dataset. "
                           "A reliable adaptation cannot be recommended from local data.")
        return out
    chosen = eligible[0]
    if core and chosen["strength"] != "strong":
        out.update(status="unsupported", best_attempt=chosen,
                   message=f"'{target}' is a primary/structural ingredient of this dish and the best local evidence "
                           f"({chosen['name']}, {chosen['strength']}) is not strong enough to claim a reliable adaptation.")
        return out
    out["chosen"] = chosen
    out["alternatives"] = eligible[1:3]
    # "supported" requires DIRECT evidence: >=2 similar dishes without the target that use the substitute
    weak = (core or chosen["strength"] == "weak" or chosen["conflicting"] or chosen["generic"]
            or chosen["already_in_recipe"] or chosen["direct_support"] < 2)
    out["status"] = "low_confidence" if weak else "supported"
    return out


def _evaluate(store, c, rec_names, with_t, without_t, signal, groups, excludes, dataset_only, context) -> dict:
    name = c["name"]
    ids = store.recipes_with(name)
    sup_without = [i for i in without_t if i in ids]
    sup_with = [i for i in with_t if i in ids]
    rate_wo = len(sup_without) / len(without_t) if without_t else 0.0
    rate_w = len(sup_with) / len(with_t) if with_t else 0.0
    lift = rate_wo - rate_w
    sig = sorted(signal & ids)
    direct = len(sup_without)
    if (direct >= 5 and lift > 0) or (len(sig) >= 5 and direct >= 2):
        strength = "strong"
    elif direct >= 2 or len(sig) >= 3:
        strength = "moderate"
    elif direct >= 1 or len(sig) >= 1:
        strength = "weak"
    else:
        strength = "none"
    conflicting = direct >= 2 and lift <= 0
    # generic: very common in similar recipes that KEEP the target -> co-occurrence, not replacement
    generic = rate_w >= 0.4
    if generic and strength == "strong":
        strength = "moderate"
    status, reason = "eligible", ""
    if context not in c.get("contexts", [context]):
        status, reason = "unsuitable", f"typically used in {'/'.join(c['contexts'])} dishes, not {context}"
    for g in (groups if status == "eligible" else []):
        if g in GROUPS:
            hit = line_violation(name, g)
            if hit:
                status, reason = "rejected", f"violates restriction: {GROUPS[g]['label']} ('{hit}')"
                break
    if status == "eligible":
        for e in excludes:
            if word_pattern(e).search(name) or inclusion_match(name, normalize_name(e)):
                status, reason = "rejected", f"violates restriction: excludes '{e}'"
                break
    if status == "eligible" and dataset_only and len(ids) == 0:
        status, reason = "rejected", "does not appear in the dataset"
    already = any(inclusion_match(n, normalize_name(name)) for n in rec_names)
    examples = []
    for i in (sup_without[:4] + [s for s in sig if s not in sup_without][:3]):
        line = _line_for(store, i, name)
        examples.append({"id": i, "title": store.recipes.at[i, "title"], "line": line,
                         "kind": "similar dish without target" if i in sup_without else "explicitly egg/dairy/etc.-free title"})
    return {
        "name": name, "ratio": c["ratio"], "predicted_effect": c["effect"], "roles": c.get("roles", []),
        "context_fit": int(context in c.get("contexts", [context])),
        "dataset_recipes_using_it": len(ids), "direct_support": direct, "with_target_count": len(sup_with),
        "rate_without": round(rate_wo, 3), "rate_with": round(rate_w, 3), "lift": round(lift, 3),
        "signal_support": len(sig), "strength": strength, "conflicting": conflicting, "generic": generic,
        "already_in_recipe": already, "status": status, "reason": reason, "examples": examples,
    }


def _line_for(store: Store, rid: str, name: str) -> str:
    names = store.names_for(name)
    for p in store.get(rid)["parsed"]:
        if p["name"] in names:
            return p["line"]
    return ""


def _data_suggested(store, rec_names, with_t, without_t, groups, excludes, k=6) -> list[dict]:
    if len(without_t) < 3:
        return []
    rn = store.recipe_names
    from collections import Counter
    cw = Counter(n for i in without_t for n in rn.get(i, ()))
    cwt = Counter(n for i in with_t for n in rn.get(i, ()))
    rows = []
    for n, c in cw.items():
        if c < 3 or n in rec_names or n in {"salt", "black pepper", "water"}:
            continue
        lift = c / len(without_t) - (cwt.get(n, 0) / len(with_t) if with_t else 0)
        if lift <= 0.03:
            continue
        bad = any(line_violation(n, g) for g in groups if g in GROUPS) or any(word_pattern(e).search(n) for e in excludes)
        rows.append({"name": n, "without_target": c, "rate_without": round(c / len(without_t), 3),
                     "rate_with": round(cwt.get(n, 0) / len(with_t), 3) if with_t else None,
                     "lift": round(lift, 3), "violates_restriction": bool(bad)})
    rows.sort(key=lambda r: r["lift"], reverse=True)
    return rows[:k]


def adapt(rec: dict, replacements: list, removals: set) -> dict:
    """Deterministic adaptation; always valid. replacements = [(positions, chosen, entry_key), ...]."""
    by_pos, first_pos = {}, {}
    for positions, chosen, key in replacements:
        for q in positions:
            by_pos[q] = (tuple(positions), chosen, key)
        first_pos[tuple(positions)] = min(positions)
    new_ing, notes = [], []
    for pos, line in enumerate(rec["ingredients"]):
        if pos in removals:
            notes.append(f"Removed '{line}' (restricted; no supported substitute).")
            continue
        if pos in by_pos:
            positions, chosen, key = by_pos[pos]
            if pos == first_pos[positions]:
                replaced = "; ".join(rec["ingredients"][q] for q in positions)
                new_ing.append(f"{chosen['name']} — {chosen['ratio']}  [replaces: {replaced}]")
            continue
        new_ing.append(line)
    new_steps = list(rec["steps"])
    changed = 0
    for positions, chosen, key in replacements:
        words = {key.split()[-1]} if key else set()
        for q in positions:
            nm = next((p["name"] for p in rec["parsed"] if p["pos"] == q and p["name"]), "")
            if nm:
                words.add(nm)
                words.add(nm.split()[-1])
        pats = [word_pattern(w) for w in sorted(words, key=len, reverse=True) if w]
        for i, st in enumerate(new_steps):
            s2 = st
            for pat in pats:
                s2 = pat.sub(chosen["name"], s2)
            if s2 != st:
                changed += 1
                new_steps[i] = s2
        notes.append(f"Prepare the {chosen['name']} as described ({chosen['ratio']}) and add it where the original "
                     f"recipe adds the {key or 'ingredient'}.")
    for pos in removals:
        nm = next((p["name"] for p in rec["parsed"] if p["pos"] == pos and p["name"]), "")
        if nm:
            pat = word_pattern(nm)
            new_steps = [pat.sub(f"(omitted {nm})", st) for st in new_steps]
    return {"ingredients": new_ing, "steps": new_steps, "notes": notes, "steps_changed": changed}


def check_adapted(lines: list[str], groups: list[str], excludes: list[str]) -> list[str]:
    bad = []
    for line in lines:
        core = line.split("[replaces:")[0]
        if " — " in core:          # substitute line: only the substitute's name is an ingredient
            core = core.split(" — ")[0]
        for g in groups:
            if g in GROUPS and line_violation(core, g):
                bad.append(f"'{core.strip()}' violates {GROUPS[g]['label']}")
        for e in excludes:
            if word_pattern(e).search(core):
                bad.append(f"'{core.strip()}' contains excluded '{e}'")
    return bad
