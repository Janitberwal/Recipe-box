"""Markdown rendering of tool results. Everything shown comes from tool output."""
from __future__ import annotations

import html as html_lib

from .groups import GROUPS

HELP_SHORT = ("Try: *Find a penne pasta recipe* · *How many recipes contain both garlic and onion?* · "
              "*I have eggs, onions, tomatoes and rice…* · *Remove eggs from this banana bread*")

HELP = """**What I can do (all answers are computed from the Recipe Box dataset):**

1. **Search** – by dish, ingredient or description: *Find a penne pasta recipe*, then *I cannot have dairy*.
2. **Statistics** – counts, percentages, common ingredients, average/median ingredient counts, rankings by steps:
   *How many recipes contain both garlic and onion?*, *Among recipes with tomato, which five have the fewest steps?*
3. **Match your ingredients** – *I have eggs, onions, tomatoes and rice. Five recipes, no peanuts, at most 6 steps.*
4. **Substitute** – *Take this banana bread recipe and remove eggs* · *Replace cream in this pasta without dairy*.

Restrictions you state (e.g. *no dairy*) stay active until you lift them (*dairy is fine again*)."""

HIDDEN_NOTE = ("Restriction checks scan every ingredient line for known terms (e.g. dairy: milk, butter, cream, "
               "cheese…, with exceptions like coconut milk / peanut butter). Hidden ingredients inside processed "
               "products cannot be detected from recipe text.")


def _src(s):
    return {"ar": "Allrecipes", "epi": "Epicurious", "fn": "Food Network"}.get(s, s)


def _changes(changes):
    return ("✅ " + "; ".join(changes) + "\n\n") if changes else ""


def _trace(audit: dict) -> str:
    labels = {"start": "all recipes", "after_text": "text match", "after_include_all": "required ingredients",
              "after_include_any": "any-of ingredients", "after_exclusions": "after exclusions", "final": "final"}
    parts = [f"{labels[k]}: {v:,}" for k, v in audit.items() if k in labels]
    return " → ".join(parts)


# ---------------------------------------------------------------- search
def _e(s) -> str:
    return html_lib.escape(str(s), quote=False)


def _tags(*items) -> str:
    return "".join(f"<span class='rtag'>{_e(x)}</span>" for x in items if x)


def _chips(items, cls, limit=None) -> str:
    items = list(items)
    more = ""
    if limit and len(items) > limit:
        more = f"<span class='ichip more'>+{len(items) - limit} more</span>"
        items = items[:limit]
    return "".join(f"<span class='ichip {cls}'>{_e(x)}</span>" for x in items) + more


def _block(s: str) -> str:
    """One HTML block with no blank lines (a blank line would end the HTML block in markdown)."""
    return "\n".join(l for l in s.splitlines() if l.strip())


def search(store, f, df, page, audit, changes, offset) -> str:
    out = [_changes(changes)]
    out.append(_block(f"""<div class='summary'><div><b>🔎 {len(df):,}</b> matching recipes</div>
<div class='summary-sub'>{_e(f.describe())}</div>
<div class='summary-trace'>{_e(_trace(audit))} · text mode: {_e(audit.get('text_mode', 'n/a'))}</div></div>"""))
    if page.empty:
        out.append("**No recipe in the dataset satisfies all of these filters.** "
                   "I did not relax any restriction. You can remove a filter (e.g. *dairy is OK*) to widen the search.")
        return "\n\n".join(out)
    restricted = bool(f.exclude_groups or f.exclude_ingredients)
    what = ", ".join([GROUPS[g]["label"] for g in f.exclude_groups] + f.exclude_ingredients)
    for n, (rid, row) in enumerate(page.iterrows(), start=offset + 1):
        rec = store.get(rid)
        ev_html = ""
        if restricted:
            ev = store.evidence_lines(rid, f.exclude_ingredients, f.exclude_groups)
            if ev["hits"]:
                ev_html = f"<div class='rev bad'>⚠️ conflict found: {_e(ev['hits'])}</div>"
            else:
                ev_html = (f"<div class='rev ok'>✔ Evidence: checked all {ev['lines_checked']} ingredient lines "
                           f"– no {_e(what)} terms found</div>")
        out.append(_block(f"""<div class='rcard'>
<div class='rhead'><span class='rnum'>{n}</span><span class='rtitle'>{_e(row['title'])}</span></div>
<div class='rmeta'>{_tags(_src(row['source']), f"{int(row['n_steps'])} steps", f"{int(row['n_ingredients'])} ingredients")}<code>ID: {_e(rid)}</code></div>
<div class='rings'>{_chips(rec['ingredients'], 'ing', 12)}</div>
{ev_html}
</div>"""))
    if restricted:
        out.append(f"<sub>{HIDDEN_NOTE}</sub>")
    out.append("Ask for *the instructions for #1*, *more*, or add a restriction.")
    return "\n\n".join(out)


# ---------------------------------------------------------------- recipe detail
def recipe(rec, show, groups, excludes, store) -> str:
    out = [f"### {rec['title']}\n`ID: {rec['id']}` · {rec['source_name']} · {rec['n_steps']} steps · "
           f"{rec['n_ingredients']} ingredients"]
    if "ingredients" in show:
        out.append("**Ingredients (original):**\n" + "\n".join(f"- {x}" for x in rec["ingredients"]))
    if "instructions" in show:
        out.append("**Instructions (original):**\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(rec["steps"], 1)))
    if groups or excludes:
        ev = store.evidence_lines(rec["id"], excludes, groups)
        if ev["hits"]:
            out.append("⚠️ **This recipe conflicts with your active restrictions:**\n" +
                       "\n".join(f"- '{l}' → {GROUPS[w]['label'] if w in GROUPS else w}" for l, w, _ in ev["hits"]) +
                       "\n\nAsk me to adapt it (e.g. *make it dairy-free*).")
        else:
            out.append("✔ No ingredient line matches your active restrictions.")
    return "\n\n".join(out)


def unsupported(attr, fields, focus) -> str:
    return (f"**I can't answer that from the dataset.** {fields}\n\n"
            f"So I cannot give {attr or 'that information'} without guessing, and I won't estimate it. "
            "What I *can* do: show the instructions text (which sometimes mentions times such as "
            "\"bake 30 minutes\"), count ingredients or steps, or compare recipes on those.")


# ---------------------------------------------------------------- stats
def stats(res, changes) -> str:
    out = [_changes(changes)]
    t = res.get("type")
    if t == "lookup":
        for it in res["items"]:
            out.append(f"**'{it['input']}'** normalizes to **'{it['normalized']}'** and appears in "
                       f"**{it['recipes']:,}** recipes. Dataset names counted as it: " +
                       ", ".join(f"{k} ({v:,})" for k, v in it["matching_names"].items()))
        return "\n\n".join(out)
    if t == "count":
        out.append(f"### {res['count']:,} recipes ({res['percentage']}% of the dataset)")
        out.append(f"- **Filters:** {res['filters']}\n- **Denominator:** {res['denominator']:,} usable recipes "
                   f"(all three files: Allrecipes, Epicurious, Food Network; records with no title, ingredients or "
                   f"instructions excluded)\n- **Calculation:** {res['calculation']}\n"
                   f"- **By source:** " + ", ".join(f"{k}: {v:,}" for k, v in res["by_source"].items()))
        out.append(f"**Ingredient-name normalization:** {res['normalization']}.")
        if res.get("name_expansion_top"):
            for q, d in res["name_expansion_top"].items():
                out.append(f"- '{q}' counted these dataset names (top, by recipes): " +
                           ", ".join(f"{k} ({v:,})" for k, v in d.items()))
        if res.get("sensitivity"):
            sv = res["sensitivity"]
            out.append(f"**Sensitivity check:** with a broader rule ({sv['broad_rule']}) the count would be "
                       f"{sv['broad_count']:,} ({sv['broad_percentage']}%). The headline number uses the stricter "
                       "head-noun rule above.")
        return "\n\n".join(out)
    if t == "describe":
        out.append(f"### {res['metric'].title()} per recipe: mean {res['mean']}, median {res['median']}")
        out.append(f"- **Filters:** {res['filters']}\n- **Recipes analyzed:** {res['analyzed']:,}\n"
                   f"- min {res['min']}, max {res['max']}, std {res['std']}\n- **Rule:** {res['rule']}\n"
                   f"- **Calculation:** {res['calculation']}")
        if res.get("by_source"):
            out.append("| Source | Recipes | Mean | Median |\n|---|---|---|---|\n" + "\n".join(
                f"| {r['source']} | {int(r['count']):,} | {r['mean']} | {r['median']} |" for r in res["by_source"]))
        return "\n\n".join(out)
    if t == "rank":
        col = "n_steps" if res["metric"] == "steps" else "n_ingredients"
        out.append(f"### Ranking by {res['metric']} ({res['order']})")
        out.append(f"- **Filters:** {res['filters']}\n- **Recipes analyzed:** {res['analyzed']:,} (complete filtered set)\n"
                   f"- **Rule:** {res['rule']}\n- **Calculation:** {res['calculation']}")
        if not res["results"]:
            out.append("No recipes match these filters.")
            return "\n\n".join(out)
        out.append(f"| # | Recipe | ID | {res['metric'].title()} | Sentences | Ingredients |\n|---|---|---|---|---|---|\n" +
                   "\n".join(f"| {i} | {r['title']} | `{r['id']}` | {r[col]} | {r['n_sentences']} | {r['n_ingredients']} |"
                             for i, r in enumerate(res["results"], 1)))
        if res.get("ties_note"):
            out.append(f"<sub>Note: {res['ties_note']}. Step counts depend on how each source formats "
                       "instructions (some write everything in one paragraph).</sub>")
        return "\n\n".join(out)
    if t == "top_ingredients":
        out.append(f"### Most common ingredients")
        out.append(f"- **Filters:** {res['filters']}\n- **Recipes analyzed:** {res['analyzed']:,}\n"
                   f"- **Calculation:** {res['calculation']}" +
                   (f"\n- Staples excluded: {', '.join(res['staples_excluded'])}" if res.get("staples_excluded") else ""))
        out.append("| # | Ingredient | Recipes | % of filtered |\n|---|---|---|---|\n" + "\n".join(
            f"| {i} | {r['ingredient']} | {r['recipes']:,} | {r['percent_of_filtered']}% |"
            for i, r in enumerate(res["results"], 1)))
        out.append(f"<sub>{res['normalization']}</sub>")
        return "\n\n".join(out)
    return "No result."


# ---------------------------------------------------------------- match
def _match_card(i, r, violating=False, n_pantry=1) -> str:
    dash = "<span class='ichip none'>—</span>"
    pct = int(100 * len(r["matched"]) / max(1, n_pantry))
    q = [f"{k}: {v}" for k, v in r["quantity_check"].items() if not v.startswith("name-level")]
    staples = f"<span class='rnote'>staples assumed: {_e(', '.join(r['staples_assumed']))}</span>" if r["staples_assumed"] else ""
    viol = ("<div class='rev bad'>⛔ Violates: " + _e("; ".join(r["violations"])) + "</div>") if violating else ""
    qty = ("<div class='rnote'>⚖️ " + _e("; ".join(q)) + "</div>") if q else ""
    return _block(f"""<div class='rcard{' violating' if violating else ''}'>
<div class='rhead'><span class='rnum'>{i}</span><span class='rtitle'>{_e(r['title'])}</span><span class='rscore'>score {r['score']:g}</span></div>
<div class='rmeta'>{_tags(r['source'], f"{r['steps']} steps")}<code>ID: {_e(r['id'])}</code></div>
<div class='mbar'><div class='mfill' style='width:{pct}%'></div></div>
<div class='mrow'><span class='mlab'>✅ Matched</span>{_chips(r['matched'], 'have') or dash}</div>
<div class='mrow'><span class='mlab'>➖ Missing</span>{_chips(r['unused_from_your_list'], 'miss') or dash}<span class='rnote'>your items it doesn't use</span></div>
<div class='mrow'><span class='mlab'>🛒 Extra required</span>{_chips(r['extra_required'], 'extra', 10) or dash}{staples}</div>
<div class='rwhy'>Why #{i}: {_e(r['score_breakdown'])}</div>
{qty}{viol}
</div>""")


def match(res, changes) -> str:
    out = [_changes(changes)]
    pantry = "".join(f"<span class='ichip have'>{_e(p['text'])}</span>" for p in res["pantry"]) or "—"
    out.append(_block(f"""<div class='summary'><div><b>🧺 Your ingredients</b> {pantry}</div>
<div class='summary-sub'>Hard constraints: {_e(res['hard_constraints'])}</div>
<div class='summary-trace'>Eligible recipes: {res['eligible_recipes']:,} · using ≥1 of your ingredients: {res['candidates_considered']:,}</div></div>"""))
    if res["ignored_pantry"]:
        out.append("⚠️ Ignored from your list: " + "; ".join(f"{x['item']} ({x['reason']})" for x in res["ignored_pantry"]))
    if res["no_match"]:
        out.append("### ❌ No eligible recipe found\nNo recipe in the dataset satisfies all hard constraints while "
                   "using any of your ingredients. I did **not** relax any constraint.")
        alts = res.get("alternatives_violating_constraints") or []
        if alts:
            out.append("**Closest alternatives – each VIOLATES at least one of your constraints (shown for "
                       "reference only):**")
            out += [_match_card(i, r, violating=True, n_pantry=len(res['pantry'])) for i, r in enumerate(alts, 1)]
        return "\n\n".join(out)
    out += [_match_card(i, r, n_pantry=len(res['pantry'])) for i, r in enumerate(res["results"], 1)]
    out.append(f"<sub>**Ranking:** {res['scoring']} **Matching:** {res['matching_rule']} "
               "Quantities are compared only when both sides use the same kind of unit (count/mass/volume); "
               "otherwise only names were matched and sufficiency is not verified.</sub>")
    return "\n\n".join(out)


# ---------------------------------------------------------------- substitution
_STRENGTH_ICON = {"strong": "🟢 strong", "moderate": "🟡 moderate", "weak": "🟠 weak", "none": "⚪ none"}


def _cand_line(c) -> str:
    flags = []
    if c["conflicting"]:
        flags.append("conflicting: about as common in recipes that keep the original")
    if c.get("generic"):
        flags.append("generic: common in these dishes anyway → indirect")
    if c["already_in_recipe"]:
        flags.append("already in this recipe")
    return (f"| {c['name']} | {_STRENGTH_ICON[c['strength']]} | {c['direct_support']} | {c['signal_support']} | "
            f"{c['dataset_recipes_using_it']:,} | {c['status']}{(': ' + c['reason']) if c['reason'] else ''}"
            f"{(' · ' + '; '.join(flags)) if flags else ''} |")


def substitution(res, changes, picked_from, rewritten) -> str:
    out = [_changes(changes)]
    if res.get("status") == "error":
        return res["message"]
    r = res["recipe"]
    if picked_from:
        out.append(f"I used the best-matching recipe for “{picked_from}”: **{r['title']}** (`{r['id']}`). "
                   "Say *use #2* or give an ID to pick another.")
    else:
        out.append(f"**Recipe:** {r['title']} (`{r['id']}`, {r['source']})")
    if res["status"] == "not_present":
        out.append(res["message"])
        return "\n\n".join(out)
    out.append(f"**Replacing:** {'; '.join(res['target_lines'])}  \n**Your restrictions:** "
               f"{', '.join(res.get('restrictions') or []) or 'none'}  \n**Dish type:** {res['context']}  \n"
               f"**Role of this ingredient** _(general culinary knowledge, not dataset evidence)_: "
               f"{', '.join(res['roles']['roles']) or 'unknown'}")
    ep = res["evidence_pool"]
    out.append(f"**Local evidence pool:** {ep['similar_recipes']} similar recipes ({ep['rule']}); "
               f"{ep['with_target']} keep the ingredient, {ep['without_target']} don't. Plus {ep['signal_recipes']:,} "
               "recipes anywhere in the dataset whose title says e.g. *eggless / dairy-free / vegan*.")
    if res.get("candidates"):
        out.append("| Candidate | Evidence | Similar recipes without it using this | ‘free-from’ titled recipes using this | "
                   "Dataset recipes using it | Status |\n|---|---|---|---|---|---|\n" +
                   "\n".join(_cand_line(c) for c in res["candidates"]))
    if res.get("data_suggested"):
        out.append("<sub>Data-only signal (ingredients over-represented when the original is absent – co-occurrence, "
                   "not proof of substitution): " + ", ".join(
                       f"{d['name']} (+{d['lift'] * 100:.0f} pts)" for d in res["data_suggested"]) + "</sub>")

    if res["status"] == "unsupported":
        out.append(f"### ⚠️ No reliable adaptation\n{res['message']}")
        if res.get("best_attempt"):
            b = res["best_attempt"]
            out.append(f"Closest idea (not recommended as reliable): **{b['name']}** – {b['ratio']}. Predicted effect: "
                       f"{b['predicted_effect']}.")
        return "\n\n".join(out)

    c = res["chosen"]
    label = ("✅ Supported by local evidence" if res["status"] == "supported" else
             "⚠️ Low-confidence adaptation (evidence is indirect)" if c["direct_support"] < 2 else
             "⚠️ Low-confidence adaptation")
    out.append(f"### {label}: use **{c['name']}**\n- Ratio _(general guideline)_: {c['ratio']}\n"
               f"- **Predicted** effect (not verified): {c['predicted_effect']}\n"
               f"- Evidence strength: {_STRENGTH_ICON[c['strength']]} – {c['direct_support']} similar recipes without "
               f"the original use it; {c['signal_support']} ‘free-from’ recipes use it.")
    if c["examples"]:
        out.append("**Supporting recipes (local dataset):**\n" + "\n".join(
            f"- `{e['id']}` {e['title']} – “{e['line']}” _({e['kind']})_" for e in c["examples"]))
    if c["direct_support"] == 0:
        out.append("_Evidence is indirect: no closely similar recipe makes this exact swap; support comes from "
                   "‘free-from’ recipes of other dishes._")
    if res.get("alternatives"):
        out.append("**Alternatives:** " + "; ".join(f"{a['name']} ({a['strength']}; {a['predicted_effect']})"
                                                     for a in res["alternatives"]))
    for s in res.get("secondary", []):
        if s["action"] == "substitute":
            out.append(f"➕ Also restricted: “{s['line']}” → **{s['chosen']['name']}** ({s['chosen']['ratio']}; "
                       f"{s['chosen']['strength']} evidence; predicted: {s['chosen']['predicted_effect']})")
        else:
            out.append(f"➖ Also restricted: “{s['line']}” → **removed** ({s['reason']})")
    ad = res["adapted"]
    steps = rewritten or ad["steps"]
    src = ("instructions re-phrased by the LLM from the adapted ingredient list, then checked against your "
           "restrictions" if rewritten else "original instructions with the ingredient swapped")
    ing_html = "".join(
        f"<li class='swap'>🔄 {_e(x)}</li>" if "[replaces:" in x else f"<li>{_e(x)}</li>" for x in ad["ingredients"])
    step_html = "".join(f"<li>{_e(s)}</li>" for s in steps)
    notes = f"<div class='anote'>📝 {_e(' '.join(ad['notes']))}</div>" if ad["notes"] else ""
    out.append(_block(f"""<div class='adapted'>
<div class='ahead'>🍽️ Adapted recipe · <span>{_e(r['title'])}</span></div>
<div class='acols'><div><div class='asub'>Ingredients</div><ul class='aing'>{ing_html}</ul></div>
<div><div class='asub'>Instructions <span class='asrc'>({_e(src)})</span></div><ol class='asteps'>{step_html}</ol></div></div>
{notes}
</div>"""))
    if res.get("adapted_violations"):
        out.append("⛔ **Remaining restriction conflicts:** " + "; ".join(res["adapted_violations"]))
    else:
        out.append("✔ Adapted ingredient list checked: no line matches your restrictions.")
    out.append(f"<sub>{res['safety_note']}</sub>")
    return "\n\n".join(out)
