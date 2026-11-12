"""Reproducible evaluation.

    python eval/run_eval.py              # tool-level cases (deterministic, no API key needed)
    python eval/run_eval.py --llm        # + end-to-end conversations through Gemini + one-shot LLM baseline

Each case runs a structured action through the Agent (same code path the chat uses) and checks the
answer with INDEPENDENT code that reads the raw JSON files directly (not the SQLite build), so the
system is not grading itself with its own logic.
Writes eval/results.md and eval/results.json.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from recipebot.agent import Agent  # noqa: E402
from recipebot.store import get_store  # noqa: E402

RAW = ROOT / "data" / "raw"

# ----------------------------------------------------------------- independent raw-data oracle
_raw = None


def raw():
    """{id: (title, [ingredient lines], instructions)} for usable records, straight from the JSON files."""
    global _raw
    if _raw is None:
        _raw = {}
        for f in ("recipes_raw_nosource_ar.json", "recipes_raw_nosource_epi.json", "recipes_raw_nosource_fn.json"):
            for k, v in json.load(open(RAW / f, encoding="utf-8")).items():
                if not v or not (v.get("title") or "").strip():
                    continue
                ings = [re.sub(r"\s+", " ", x.replace("ADVERTISEMENT", "")).strip() for x in v.get("ingredients") or []]
                ings = [x for x in ings if x and not (x.endswith(":") and len(x) < 60)]
                if not ings:
                    continue
                _raw[k] = (v["title"].strip(), ings, v.get("instructions") or "")
    return _raw


def mentions(lines, word):
    pat = re.compile(rf"(?<![a-z]){re.escape(word)}(?:s|es)?(?![a-z])", re.I)
    return any(pat.search(l) for l in lines)


DAIRY_WORDS = ["milk", "butter", "cream", "cheese", "yogurt", "parmesan", "mozzarella", "cheddar", "ricotta",
               "ghee", "buttermilk", "half-and-half", "feta"]
DAIRY_OK = re.compile(r"coconut milk|coconut cream|almond milk|soy milk|peanut butter|almond butter|cream of tartar|"
                      r"butternut|non-dairy|dairy-free|apple butter|cocoa butter|rice milk|oat milk|butter beans?|"
                      r"butter lettuce|cashew butter", re.I)


def has_dairy(lines):
    return any(mentions([DAIRY_OK.sub(" ", l)], w) for l in lines for w in DAIRY_WORDS)


def raw_steps(instr):
    bp = re.compile(r"^(watch how to make this recipe|photograph(s)? by|photo(s)? by|copyright|©|recipe courtesy|"
                    r"reprinted (with|from)|excerpted from|adapted from|cook'?s note|from .{0,80} by .{0,80}$)", re.I)
    return [l.strip() for l in instr.split("\n") if l.strip() and not bp.match(l.strip())]


# ----------------------------------------------------------------- cases
CASES = []


def case(cid, task, desc, conversation):
    def deco(fn):
        CASES.append({"id": cid, "task": task, "desc": desc, "conversation": conversation, "check": fn})
        return fn
    return deco


@case("T1-search-refine", "Task 1", "Search penne, then 'I cannot have dairy' refines the SAME search",
      [("Find a penne pasta recipe", {"intent": "search", "search_mode": "new", "query": "penne pasta"}),
       ("I cannot have dairy", {"intent": "update_constraints", "add_exclude_groups": ["dairy"]})])
def _(replies, ag):
    r1, r2 = replies
    ids1, ids2 = r1.data["ids"], r2.data["ids"]
    R = raw()
    checks = {
        "turn1 returns results": len(ids1) > 0,
        "turn1 every result mentions penne": all("penne" in (R[i][0] + " ".join(R[i][1])).lower() for i in ids1),
        "turn2 keeps the penne query": r2.data["filter"]["query"] == "penne pasta",
        "turn2 every result still penne": all("penne" in (R[i][0] + " ".join(R[i][1])).lower() for i in ids2),
        "turn2 no dairy word in any ingredient line (independent check)": all(not has_dairy(R[i][1]) for i in ids2),
        "turn2 result set shrank": r2.data["audit"]["final"] < r1.data["audit"]["final"],
    }
    return checks, f"{r1.data['audit']['final']} → {r2.data['audit']['final']} recipes"


@case("T1-count", "Task 1", "How many recipes contain both garlic and onion? Percentage?",
      [("How many recipes contain both garlic and onion? What percentage of the dataset is that?",
        {"intent": "stats", "stats_op": "count", "include_all": ["garlic", "onion"]})])
def _(replies, ag):
    d = replies[0].data
    R = raw()
    # independent recomputation of the BROAD rule straight from raw JSON
    usable = {k for k, v in R.items() if raw_steps(v[2])}
    broad = sum(1 for k in usable if mentions(R[k][1], "garlic") and mentions(R[k][1], "onion"))
    checks = {
        "denominator == usable recipes in raw files": d["denominator"] == len(usable),
        "broad count matches independent raw recomputation": d["sensitivity"]["broad_count"] == broad,
        "strict count <= broad count": d["count"] <= d["sensitivity"]["broad_count"],
        "percentage = count/denominator": abs(d["percentage"] - round(100 * d["count"] / d["denominator"], 2)) < 1e-9,
        "answer states filters, denominator and normalization": all(
            s in replies[0].markdown for s in ("Filters", "Denominator", "normalization")),
    }
    return checks, f"strict {d['count']:,} ({d['percentage']}%), broad {broad:,} of {len(usable):,}"


@case("T1-rank", "Task 1", "Among recipes containing tomato, five with the fewest instruction steps",
      [("Among recipes containing tomato, which five have the fewest instruction steps?",
        {"intent": "stats", "stats_op": "rank", "include_all": ["tomato"], "metric": "steps", "order": "asc", "k": 5})])
def _(replies, ag):
    d = replies[0].data
    R = raw()
    res = d["results"]
    steps = [len(raw_steps(R[r["id"]][2])) for r in res]
    tomato_recipes = [k for k, v in R.items() if mentions(v[1], "tomato") and raw_steps(v[2])]
    global_min = min(len(raw_steps(R[k][2])) for k in tomato_recipes)
    checks = {
        "5 results with IDs": len(res) == 5 and all(r["id"] in R for r in res),
        "every result has a tomato ingredient line": all(mentions(R[r["id"]][1], "tomato") for r in res),
        "step counts re-derived from raw text agree": steps == [r["n_steps"] for r in res],
        "non-decreasing order": steps == sorted(steps),
        "first result equals global minimum over ALL tomato recipes": steps[0] == global_min,
        "analysed the complete filtered set (>1000 recipes)": d["analyzed"] > 1000,
    }
    return checks, f"steps {steps}; analysed {d['analyzed']:,}"


@case("T1-describe", "Task 1", "Average and median ingredient count by source",
      [("What is the average and median number of ingredients per recipe, by source?",
        {"intent": "stats", "stats_op": "describe", "metric": "ingredients", "group_by_source": True})])
def _(replies, ag):
    d = replies[0].data
    R = raw()
    counts = [len(v[1]) for v in R.values() if raw_steps(v[2])]
    checks = {
        "median matches independent recomputation": d["median"] == statistics.median(counts),
        "mean matches within 0.05": abs(d["mean"] - statistics.mean(counts)) < 0.05,
        "analysed every usable recipe": d["analyzed"] == len(counts),
    }
    return checks, f"mean {d['mean']} (oracle {statistics.mean(counts):.2f}), median {d['median']}"


@case("T1-unsupported", "Task 1", "Question about calories / cooking time must be refused, not estimated",
      [("How many calories are in this recipe and how long does it take?",
        {"intent": "unsupported_attribute", "unsupported_attribute": "calories and cooking time"})])
def _(replies, ag):
    md = replies[0].markdown
    checks = {"says it cannot answer": "can't answer" in md, "explains what the dataset lacks": "nutrition" in md,
              "no number of calories given": not re.search(r"\d+\s*(k?cal|calories)", md)}
    return checks, "refused with explanation"


@case("T2-match-followup", "Task 2", "Eggs/onions/tomatoes/rice, no peanuts, ≤6 steps; then 'Exclude eggs too'",
      [("I have eggs, onions, tomatoes, and rice. Suggest five recipes using as many of these as possible, with no "
        "peanuts and at most six instruction steps. Prefer recipes that need the fewest additional ingredients.",
        {"intent": "match", "pantry_mode": "new", "pantry": ["eggs", "onions", "tomatoes", "rice"],
         "add_exclude_groups": ["peanut"], "max_steps": 6, "k": 5}),
       ("Exclude eggs too", {"intent": "update_constraints", "add_exclude_groups": ["egg"]})])
def _(replies, ag):
    R = raw()
    r1, r2 = replies[0].data["results"], replies[1].data["results"]

    def ok(rs, no_egg=False):
        out = True
        for r in rs:
            lines = R[r["id"]][1]
            out &= not mentions(lines, "peanut")
            out &= len(raw_steps(R[r["id"]][2])) <= 6
            if no_egg:
                out &= not mentions(lines, "egg")
        return out
    checks = {
        "5 results each turn": len(r1) == 5 and len(r2) == 5,
        "turn1: no peanuts, ≤6 steps (raw check)": ok(r1),
        "turn1: scores non-increasing": [x["score"] for x in r1] == sorted([x["score"] for x in r1], reverse=True),
        "turn1: matched items really appear in raw lines": all(
            all(mentions(R[r["id"]][1], m.split()[-1]) for m in r["matched"]) for r in r1),
        "turn2: no egg in any result (raw check)": ok(r2, no_egg=True),
        "turn2: eggs reported as ignored": any("egg" in x["item"] for x in replies[1].data["ignored_pantry"]),
        "explanation shown (score breakdown)": all("matched" in r["score_breakdown"] for r in r1),
    }
    return checks, "; ".join(f"{r['title']} ({r['score']:g})" for r in r1[:3])


@case("T2-no-match", "Task 2", "Impossible constraints → explicit 'no match' + clearly-marked violating alternatives",
      [("I have tofu and kale. Find a lasagna with at most 1 step, no dairy and no gluten.",
        {"intent": "match", "pantry_mode": "new", "pantry": ["tofu", "kale"], "query": "lasagna",
         "add_exclude_groups": ["dairy", "gluten"], "max_steps": 1})])
def _(replies, ag):
    d = replies[0].data
    alts = d.get("alternatives_violating_constraints", [])
    checks = {"no_match flagged": d["no_match"] is True, "no eligible results returned": d["results"] == [],
              "alternatives each list a violation": bool(alts) and all(a["violations"] for a in alts),
              "markdown says constraints not relaxed": "did **not** relax" in replies[0].markdown}
    return checks, f"{len(alts)} alternatives, e.g. {alts[0]['violations'][:1] if alts else '-'}"


@case("T2-quantity", "Task 2", "Quantities: '2 eggs' compared by count; '500 g rice' not convertible from cups",
      [("I have 2 eggs and 500 g rice", {"intent": "match", "pantry_mode": "new", "pantry": ["2 eggs", "500 g rice"], "k": 5})])
def _(replies, ag):
    rs = replies[0].data["results"]
    q = [v for r in rs for v in r["quantity_check"].values()]
    checks = {"egg quantity verified as sufficient/insufficient": any(v.startswith(("sufficient", "INSUFFICIENT")) for v in q),
              "unverifiable quantities reported as NOT verified": any("NOT verified" in v for v in q) or
              all(v.startswith(("sufficient", "INSUFFICIENT")) for v in q)}
    return checks, q[:3]


@case("T3-banana-bread-eggs", "Task 3", "Banana bread without eggs, dataset ingredients only",
      [("Take this banana bread recipe and remove eggs. Use only ingredients that appear in the dataset.",
        {"intent": "substitute", "target_ingredient": "eggs", "recipe_ref": "banana bread"})])
def _(replies, ag):
    d = replies[0].data
    R = raw()
    ch = d.get("chosen", {})
    ex = ch.get("examples", [])
    checks = {
        "a substitute was chosen": bool(ch),
        "substitute appears in the dataset": ch.get("dataset_recipes_using_it", 0) > 0,
        "adapted ingredient list has no egg line": not any(
            mentions([l.split(" — ")[0] if " — " in l else l], "egg") for l in d["adapted"]["ingredients"]),
        "cited recipes exist and do NOT contain egg (raw check)": bool(ex) and all(
            e["id"] in R and not mentions(R[e["id"]][1], "egg") for e in ex),
        "cited recipes contain the substitute (raw check)": all(
            mentions(R[e["id"]][1], ch["name"].split()[-1]) for e in ex),
        "effects labelled as predicted": "Predicted" in replies[0].markdown,
        "revised instructions returned": len(d["adapted"]["steps"]) > 0,
    }
    return checks, f"{d['recipe']['title']} → {ch.get('name')} ({ch.get('strength')}); status {d['status']}"


@case("T3-pasta-cream-dairy", "Task 3", "Replace cream in a pasta recipe without dairy (secondary dairy lines too)",
      [("Find a penne recipe with heavy cream", {"intent": "search", "search_mode": "new", "query": "penne",
                                                  "include_all": ["heavy cream", "parmesan cheese"]}),
       ("Replace cream in this pasta recipe without dairy. Prefer examples from recipes already in the database.",
        {"intent": "substitute", "target_ingredient": "cream", "add_exclude_groups": ["dairy"]})])
def _(replies, ag):
    d = replies[1].data
    lines = [l.split("[replaces:")[0] for l in d["adapted"]["ingredients"]]
    lines = [l.split(" — ")[0] if " — " in l else l for l in lines]
    checks = {
        "chosen substitute is not dairy": not has_dairy([d["chosen"]["name"]]),
        "dairy candidates were rejected with reason": any(c["status"] == "rejected" and "dairy" in c["reason"]
                                                          for c in d["candidates"]),
        "NO dairy anywhere in adapted ingredient list (independent check)": not has_dairy(lines),
        "other dairy lines (e.g. cheese) handled": len(d.get("secondary", [])) >= 1,
        "safety note present": "not a medical" in replies[1].markdown,
    }
    return checks, f"{d['recipe']['title']}: cream → {d['chosen']['name']}; secondary " + \
        ", ".join(f"{s['line']}→{s.get('chosen', {}).get('name', 'removed')}" for s in d.get("secondary", []))


@case("T3-weak-evidence", "Task 3", "Remove eggs from a frittata → evidence too weak; must refuse a confident swap",
      [("Show me a frittata", {"intent": "search", "search_mode": "new", "query": "frittata"}),
       ("Remove the eggs from it", {"intent": "substitute", "target_ingredient": "eggs"})])
def _(replies, ag):
    d = replies[1].data
    checks = {"status is unsupported (no confident adaptation)": d["status"] == "unsupported",
              "core-ingredient warning raised": d.get("core_ingredient_warning") is True,
              "baked-only substitutes marked unsuitable": any(c["status"] == "unsuitable" for c in d["candidates"]),
              "message explains why": "No reliable adaptation" in replies[1].markdown}
    return checks, d.get("message", "")[:120]


@case("T3-not-present", "Task 3", "Ask to replace an ingredient the recipe does not contain",
      [("Find a penne pasta recipe", {"intent": "search", "search_mode": "new", "query": "penne pasta salad asparagus"}),
       ("Replace the chocolate", {"intent": "substitute", "target_ingredient": "chocolate"})])
def _(replies, ag):
    d = replies[1].data
    return {"reports not present": d["status"] == "not_present"}, d.get("message", "")


# ----------------------------------------------------------------- runner
def run_tool_cases(store):
    results = []
    for c in CASES:
        ag = Agent(store)
        t = time.time()
        replies = [ag.handle(u, action=a) for u, a in c["conversation"]]
        try:
            checks, summary = c["check"](replies, ag)
        except Exception as e:
            checks, summary = {"check raised": False}, repr(e)
        results.append({"id": c["id"], "task": c["task"], "desc": c["desc"], "checks": checks,
                        "passed": all(checks.values()), "summary": str(summary), "seconds": round(time.time() - t, 1),
                        "transcript": [(u, r.markdown) for (u, _), r in zip(c["conversation"], replies)]})
        print(("PASS " if results[-1]["passed"] else "FAIL ") + c["id"], "-", summary)
        for k, v in checks.items():
            if not v:
                print("   ✗", k)
    return results


# parsing eval (LLM): user text -> expected key fields
PARSE_CASES = [
    ("Find a penne pasta recipe", {"intent": "search"}),
    ("I cannot have dairy", {"intent": "update_constraints", "add_exclude_groups": ["dairy"]}),
    ("How many recipes contain both garlic and onion? What percentage of the dataset is that?",
     {"intent": "stats", "stats_op": "count"}),
    ("Among recipes containing tomato, which five have the fewest instruction steps?",
     {"intent": "stats", "stats_op": "rank", "order": "asc"}),
    ("I have eggs, onions, tomatoes, and rice. Suggest five recipes, no peanuts, at most six steps.",
     {"intent": "match", "max_steps": 6, "add_exclude_groups": ["peanut"]}),
    ("Exclude eggs too", {"intent": "update_constraints"}),
    ("Take this banana bread recipe and remove eggs.", {"intent": "substitute", "target_ingredient": "egg"}),
    ("How many calories does it have?", {"intent": "unsupported_attribute"}),
    ("show me the instructions for the second one", {"intent": "show_recipe", "recipe_ref": "2"}),
    ("What are the 10 most common ingredients in Epicurious recipes?", {"intent": "stats", "stats_op": "top_ingredients"}),
]


def _field_ok(got, exp):
    if isinstance(exp, list):
        g = [str(x).lower() for x in (got or [])]
        return all(any(e in x for x in g) for e in exp)
    if isinstance(exp, str) and isinstance(got, str):
        return exp.lower() in got.lower()
    return got == exp


def run_llm(store):
    from recipebot import llm
    out = {"parse": [], "baseline": [], "conversations": []}
    ag = Agent(store)
    for text, exp in PARSE_CASES:
        try:
            a = ag.parse(text)
        except Exception as e:
            a = {"error": str(e)}
        ok = all(_field_ok(a.get(k), v) for k, v in exp.items())
        out["parse"].append({"text": text, "expected": exp, "got": a, "ok": ok})
        print(("PARSE OK  " if ok else "PARSE BAD ") + text)
        time.sleep(7)

    # one-shot LLM baseline vs. grounded system
    R = raw()
    usable = {k for k, v in R.items() if raw_steps(v[2])}
    qs = [
        "In the Recipe Box dataset (eightportions.com, ~125k recipes from Allrecipes, Epicurious, Food Network), how many "
        "recipes contain both garlic and onion, and what percentage of the dataset is that? Give a single number.",
        "In the Recipe Box dataset, among recipes containing tomato, list the five with the fewest instruction steps. "
        "Give each recipe's dataset ID and title.",
        "Give me the calories per serving and total cooking time of a classic banana bread recipe from the Recipe Box dataset.",
    ]
    for q in qs:
        try:
            ans = llm.generate_text("Answer the question directly.", q, temperature=0.0)
        except Exception as e:
            ans = f"(error: {e})"
        ids = re.findall(r"[A-Za-z0-9./]{25,35}", ans)
        out["baseline"].append({"question": q, "answer": ans, "ids_found": ids,
                                "ids_that_exist": [i for i in ids if i in R]})
        time.sleep(7)
    out["oracle"] = {"usable": len(usable),
                     "garlic_onion_broad": sum(1 for k in usable if mentions(R[k][1], "garlic") and mentions(R[k][1], "onion"))}

    # end-to-end conversations through the real parser
    convs = [
        ["Find a penne pasta recipe", "I cannot have dairy", "Show me the instructions for the first one"],
        ["How many recipes contain both garlic and onion? What percentage of the dataset is that?",
         "Among recipes containing tomato, which five have the fewest instruction steps?"],
        ["I have eggs, onions, tomatoes, and rice. Suggest five recipes using as many of these as possible, with no "
         "peanuts and at most six instruction steps. Prefer recipes that need the fewest additional ingredients.",
         "Exclude eggs too"],
        ["Take a banana bread recipe and remove eggs. Use only ingredients that appear in the dataset. What "
         "replacements have evidence from related baked recipes, and how might the texture change?"],
        ["Find a creamy penne pasta with heavy cream",
         "Replace cream in this pasta recipe without dairy. Prefer examples from recipes already in the database."],
        ["I have tofu and kale. Find me a lasagna with at most 1 step, no dairy and no gluten."],
        ["Show me a frittata", "Remove the eggs from it", "How many calories does it have?"],
    ]
    for conv in convs:
        ag = Agent(store)
        turns = []
        for u in conv:
            r = ag.handle(u)
            turns.append({"user": u, "action": r.action, "reply": r.markdown})
            time.sleep(7)
        out["conversations"].append(turns)
        print("CONV done:", conv[0][:60])
    return out


def baseline_verdicts(llm_out, tool):
    """Automatic comparison of the one-shot answers with the oracle / grounded system."""
    o = llm_out["oracle"]
    strict = next((r["summary"] for r in tool if r["id"] == "T1-count"), "")
    b = llm_out["baseline"]
    out = []
    nums = [int(x.replace(",", "")) for x in re.findall(r"\d[\d,]{2,}", b[0]["answer"])
            if int(x.replace(",", "")) > 500]
    if nums:
        off = min(abs(n - o["garlic_onion_broad"]) / o["garlic_onion_broad"] for n in nums)
        out.append(f"❌ Ungrounded number(s) {nums}. Computed from the data: {strict}. The one-shot value is off by "
                   f"{off * 100:.0f}% even against the broadest matching rule, and gives no denominator or normalization.")
    else:
        out.append("⚠️ No usable number given. The grounded system computes it exactly (see T1-count).")
    ids, real = b[1]["ids_found"], b[1]["ids_that_exist"]
    claimed = re.findall(r"ID:?\**\s*`?([A-Za-z0-9./]*\d[A-Za-z0-9./]*)`?", b[1]["answer"])
    R = raw()
    claimed_real = [c for c in claimed if c in R]
    rank = next((r for r in tool if r["id"] == "T1-rank"), None)
    if not ids and not claimed:
        out.append("❌ Could not name a single recipe or ID (it answered with generic code/advice): a one-shot LLM "
                   "has no access to the dataset. The grounded system ranks ALL tomato recipes and returns 5 real "
                   f"IDs whose step counts are re-verified from raw text (T1-rank: {rank['summary'] if rank else ''}).")
    elif not real and not claimed_real:
        out.append(f"❌ None of the cited IDs exist in the dataset (cited: {', '.join(claimed[:5]) or 'none'}; real "
                   "IDs are 30-character keys). The grounded system ranks ALL tomato recipes and returns real IDs "
                   f"whose step counts are re-verified from raw text (T1-rank: {rank['summary'] if rank else ''}).")
    else:
        out.append(f"⚠️ {len(real) + len(claimed_real)} cited IDs exist in the dataset.")
    fabricated = bool(re.search(r"\d+\s*(?:to|-|–)?\s*\d*\s*(?:k?cal|calories)", b[2]["answer"], re.I))
    out.append("❌ Fabricates calories/cooking time that the dataset does not contain. The grounded system refuses "
               "and names the missing fields (T1-unsupported)." if fabricated else
               "✅ The one-shot model also declined to give numbers.")
    return out


def write_report(tool, llm_out):
    lines = ["# Evaluation results", "",
             "Generated by `python eval/run_eval.py" + (" --llm" if llm_out else "") + "`. "
             "Checks use an independent oracle that re-reads the raw JSON files.", ""]
    passed = sum(r["passed"] for r in tool)
    lines += [f"## Tool-level cases: {passed}/{len(tool)} passed", "",
              "| Case | Task | Description | Result | Summary |", "|---|---|---|---|---|"]
    for r in tool:
        lines.append(f"| {r['id']} | {r['task']} | {r['desc']} | {'✅ pass' if r['passed'] else '❌ fail'} | "
                     f"{r['summary'].replace('|', '/')} |")
    lines += ["", "### Individual checks", ""]
    for r in tool:
        lines.append(f"**{r['id']}**")
        lines += [f"- {'✅' if v else '❌'} {k}" for k, v in r["checks"].items()]
        lines.append("")
    if llm_out:
        ok = sum(p["ok"] for p in llm_out["parse"])
        lines += [f"## LLM intent parsing: {ok}/{len(llm_out['parse'])} correct", "",
                  "| Message | Expected | Parsed intent | OK |", "|---|---|---|---|"]
        for p in llm_out["parse"]:
            lines.append(f"| {p['text'][:70]} | `{json.dumps(p['expected'])}` | `{p['got'].get('intent')}` | "
                         f"{'✅' if p['ok'] else '❌'} |")
        lines += ["", "## One-shot LLM baseline vs. grounded system", "",
                  f"Oracle: {llm_out['oracle']['usable']:,} usable recipes; garlic+onion (broad whole-word rule) = "
                  f"{llm_out['oracle']['garlic_onion_broad']:,}.", ""]
        for b, v in zip(llm_out["baseline"], baseline_verdicts(llm_out, tool)):
            lines += [f"**Q:** {b['question']}", "", "**One-shot LLM answer:**", "", "> " + b["answer"].replace("\n", "\n> "),
                      "", f"**Verdict:** {v}", ""]
        lines += ["## End-to-end conversations (real LLM parsing)", ""]
        for conv in llm_out["conversations"]:
            for t in conv:
                lines += [f"**User:** {t['user']}", "", f"<details><summary>parsed action</summary>\n\n```json\n"
                          f"{json.dumps(t['action'], indent=1)}\n```\n</details>", "", t["reply"], "", "---", ""]
    lines += ["## Tool-level transcripts", ""]
    for r in tool:
        lines.append(f"### {r['id']}: {r['desc']}")
        for u, md in r["transcript"]:
            lines += [f"**User:** {u}", "", md, ""]
        lines.append("---")
    (ROOT / "eval" / "results.md").write_text("\n".join(lines), encoding="utf-8")
    (ROOT / "eval" / "results.json").write_text(json.dumps({"tool": tool, "llm": llm_out},
                                                           indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--report-only", action="store_true", help="rebuild results.md from results.json (no API calls)")
    args = ap.parse_args()
    if args.report_only:
        saved = json.loads((ROOT / "eval" / "results.json").read_text(encoding="utf-8"))
        write_report(saved["tool"], saved["llm"])
        print("Report regenerated: eval/results.md")
        sys.exit(0)
    store = get_store()
    tool = run_tool_cases(store)
    llm_out = run_llm(store) if args.llm else None
    if llm_out is None and (ROOT / "eval" / "results.json").exists():
        # keep the last LLM section (baseline + transcripts) when re-running only the deterministic cases
        llm_out = json.loads((ROOT / "eval" / "results.json").read_text(encoding="utf-8")).get("llm")
    write_report(tool, llm_out)
    print(f"\n{sum(r['passed'] for r in tool)}/{len(tool)} tool-level cases passed. Report: eval/results.md")
