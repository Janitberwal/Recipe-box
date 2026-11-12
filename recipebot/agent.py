"""Conversation agent: LLM parses each turn into a structured action; deterministic tools
execute it against the dataset; results are rendered from tool output only."""
from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field, asdict

from . import llm, render, stats
from .groups import GROUPS, resolve_group
from .matcher import MatchRequest, match
from .store import Filter, Store
from .substitute import substitute

UNSUPPORTED_FIELDS = ("The Recipe Box dataset only contains a title, an ingredient list, instructions and a "
                      "picture link for each recipe. It has no structured fields for nutrition (calories, macros), "
                      "cooking/prep time, servings, ratings, cuisine, cost or difficulty.")

PARSE_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": ["search", "update_constraints", "show_recipe", "stats", "match",
                                              "substitute", "more", "unsupported_attribute", "reset", "help", "other"]},
        "search_mode": {"type": "string", "enum": ["new", "refine"]},
        "query": {"type": "string"},
        "include_all": {"type": "array", "items": {"type": "string"}},
        "include_any": {"type": "array", "items": {"type": "string"}},
        "add_exclude_ingredients": {"type": "array", "items": {"type": "string"}},
        "add_exclude_groups": {"type": "array", "items": {"type": "string"}},
        "remove_restrictions": {"type": "array", "items": {"type": "string"}},
        "clear_restrictions": {"type": "boolean"},
        "max_steps": {"type": "integer"},
        "min_steps": {"type": "integer"},
        "max_ingredients": {"type": "integer"},
        "clear_limits": {"type": "boolean"},
        "sources": {"type": "array", "items": {"type": "string", "enum": ["ar", "epi", "fn"]}},
        "stats_op": {"type": "string", "enum": ["count", "describe", "rank", "top_ingredients", "lookup"]},
        "metric": {"type": "string", "enum": ["steps", "ingredients"]},
        "order": {"type": "string", "enum": ["asc", "desc"]},
        "k": {"type": "integer"},
        "group_by_source": {"type": "boolean"},
        "use_active_search": {"type": "boolean"},
        "include_staples": {"type": "boolean"},
        "pantry_mode": {"type": "string", "enum": ["new", "add", "remove", "keep"]},
        "pantry": {"type": "array", "items": {"type": "string"}},
        "prefer_fewer_steps": {"type": "boolean"},
        "recipe_ref": {"type": "string"},
        "target_ingredient": {"type": "string"},
        "show": {"type": "array", "items": {"type": "string", "enum": ["ingredients", "instructions"]}},
        "unsupported_attribute": {"type": "string"},
        "reply": {"type": "string"},
    },
    "required": ["intent"],
}

SYSTEM = """You convert ONE user message of a recipe chatbot into a JSON action. You never answer questions
yourself and never invent recipes, IDs, counts or facts - the system computes everything from a dataset.

Intents:
- search: find recipes by dish name, ingredients, or a natural-language request. search_mode "new" for a new
  topic, "refine" when the user narrows the current search (e.g. "only ones with chicken", "make it spicy").
  Put dish/keywords in `query` (short, e.g. "penne pasta", "chocolate chip cookies"). Put required ingredients in
  include_all. Translate vague requests into concrete keywords (e.g. "something cozy for winter" -> "stew").
- update_constraints: the user ONLY states/changes a restriction or limit that should apply to whatever is
  active (e.g. "I cannot have dairy", "exclude eggs too", "at most 4 steps", "nuts are fine again").
  Diet/allergy words go to add_exclude_groups using these names only: dairy, egg, peanut, tree_nut, gluten,
  shellfish, fish, soy, meat, vegetarian, vegan. Other specific ingredients go to add_exclude_ingredients.
  Lifting a restriction -> remove_restrictions.
- show_recipe: user wants ingredients/instructions/details of a recipe. recipe_ref = the recipe ID if given,
  else the result number from the numbered list ("the second one" -> "2"), else the dish title.
- stats: counts, percentages, most common ingredients, average/median ingredient or step counts, rankings by
  steps or ingredients. stats_op: count | describe (average/median/min/max) | rank | top_ingredients | lookup
  (how an ingredient name is normalized). Use include_all for "contains both X and Y", include_any for "X or Y".
  metric: steps|ingredients. order asc = fewest. use_active_search=true only if the user refers to the current
  results ("of those", "among these").
- match: user lists ingredients they HAVE and wants recipes using them. pantry = items with quantities if given
  ("2 eggs", "500 g rice"). pantry_mode new|add|remove. Hard constraints (no peanuts, max steps) also go in
  add_exclude_groups / add_exclude_ingredients / max_steps. prefer_fewer_steps if they prefer fewer steps.
  k = number of results requested.
- substitute: adapt a recipe by replacing/removing an ingredient. target_ingredient = what to replace (e.g. "eggs",
  "cream"); restrictions (e.g. "without dairy") -> add_exclude_groups. recipe_ref as in show_recipe ("this" ->
  leave empty to use the current recipe; a dish name like "banana bread" -> recipe_ref "banana bread").
- more: show more results of the current list.
- unsupported_attribute: questions about data the dataset does not have (calories, nutrition, cooking time,
  servings, ratings, price, cuisine of origin). Set unsupported_attribute.
- reset: start over. help: what can you do. other: chit-chat (put a one-sentence reply in `reply`).
Only fill fields that the message actually mentions. Use the conversation context to resolve references.
A newly stated restriction is ALWAYS an exclusion, never an include."""


EXAMPLES = """

EXAMPLES (message -> action):
"Find a penne pasta recipe" -> {"intent":"search","search_mode":"new","query":"penne pasta"}
"I cannot have dairy" -> {"intent":"update_constraints","add_exclude_groups":["dairy"]}
"How many recipes contain both garlic and onion? What percentage of the dataset is that?" ->
  {"intent":"stats","stats_op":"count","include_all":["garlic","onion"]}
"How many recipes use chicken or turkey?" -> {"intent":"stats","stats_op":"count","include_any":["chicken","turkey"]}
"Among recipes containing tomato, which five have the fewest instruction steps?" ->
  {"intent":"stats","stats_op":"rank","include_all":["tomato"],"metric":"steps","order":"asc","k":5}
"What is the average and median number of ingredients in Epicurious cake recipes?" ->
  {"intent":"stats","stats_op":"describe","metric":"ingredients","query":"cake","sources":["epi"]}
"What are the 10 most common ingredients in vegan recipes?" ->
  {"intent":"stats","stats_op":"top_ingredients","k":10,"add_exclude_groups":["vegan"]}
"I have eggs, onions, tomatoes, and rice. Suggest five recipes using as many of these as possible, with no peanuts
 and at most six instruction steps. Prefer recipes that need the fewest additional ingredients." ->
  {"intent":"match","pantry_mode":"new","pantry":["eggs","onions","tomatoes","rice"],"add_exclude_groups":["peanut"],
   "max_steps":6,"k":5}
"Exclude eggs too" -> {"intent":"update_constraints","add_exclude_groups":["egg"]}
"I also have 2 cups of spinach" -> {"intent":"match","pantry_mode":"add","pantry":["2 cups spinach"]}
"show the instructions for #2" -> {"intent":"show_recipe","recipe_ref":"2","show":["instructions"]}
"Take this banana bread recipe and remove eggs" (no banana bread in context) ->
  {"intent":"substitute","target_ingredient":"eggs","recipe_ref":"banana bread"}
"Replace cream in this pasta recipe without dairy" (a recipe is in focus) ->
  {"intent":"substitute","target_ingredient":"cream","add_exclude_groups":["dairy"]}
"How long does it take to cook?" -> {"intent":"unsupported_attribute","unsupported_attribute":"cooking time"}
ALWAYS put every ingredient the user names into include_all / include_any / pantry - never drop them."""

_CONTAINS_RE = re.compile(r"\b(?:contain|contains|containing|with|use|uses|using|have|has)\s+(?:both\s+)?"
                          r"([a-z][a-z \-]+?)(?:\s+(and|or)\s+([a-z][a-z \-]+?))?(?=\?|,|\.|$|\s+(?:what|which|and what|in|among))",
                          re.I)


def sanitize_action(raw: dict) -> dict:
    """Validate the LLM output against PARSE_SCHEMA: drop unknown keys, coerce types, check enums."""
    if not isinstance(raw, dict):
        return {"intent": "other"}
    props = PARSE_SCHEMA["properties"]
    out = {}
    for k, v in raw.items():
        spec = props.get(k)
        if spec is None or v is None or v == "" or v == []:
            continue
        t = spec["type"]
        try:
            if t == "string":
                v = str(v)
                if "enum" in spec and v not in spec["enum"]:
                    continue
            elif t == "integer":
                v = int(v)
            elif t == "boolean":
                v = v if isinstance(v, bool) else str(v).lower() in ("true", "1", "yes")
            elif t == "array":
                v = [str(x) for x in (v if isinstance(v, list) else [v]) if str(x).strip()]
                enum = spec["items"].get("enum")
                if enum:
                    v = [x for x in v if x in enum]
                if not v:
                    continue
        except (TypeError, ValueError):
            continue
        out[k] = v
    if out.get("intent") not in props["intent"]["enum"]:
        out["intent"] = "other"
    return out


def repair_action(a: dict, text: str) -> dict:
    """Deterministic safety net: if a stats/search action lost the ingredients the user clearly named
    ("contain both X and Y"), recover them from the text. Recorded in the action for transparency."""
    if a.get("intent") == "stats" and not (a.get("include_all") or a.get("include_any") or a.get("query")):
        m = _CONTAINS_RE.search(text)
        if m:
            first, op, second = m.group(1).strip(), (m.group(2) or "").lower(), (m.group(3) or "").strip()
            items = [x for x in (first, second) if x and x.lower() not in {"the", "a", "recipes"}]
            if items:
                a["include_any" if op == "or" else "include_all"] = items
                a["_repaired"] = "ingredients recovered from the message text by a deterministic rule"
    return a


@dataclass
class Session:
    exclude_groups: list[str] = field(default_factory=list)
    exclude_ingredients: list[str] = field(default_factory=list)
    search: dict = field(default_factory=dict)        # Filter fields of the active search (no restrictions)
    match: dict = field(default_factory=dict)         # pantry & limits of the active match request
    active_task: str = ""                             # search | match | stats | substitute
    last_results: list[dict] = field(default_factory=list)   # [{id,title}]
    last_offset: int = 0
    focus_recipe: str = ""
    history: list[dict] = field(default_factory=list)

    def restriction_text(self) -> str:
        parts = [GROUPS[g]["label"] for g in self.exclude_groups] + self.exclude_ingredients
        return ", ".join(parts) if parts else "none"


@dataclass
class Reply:
    markdown: str
    action: dict
    data: dict | None = None
    error: str | None = None


class Agent:
    def __init__(self, store: Store):
        self.store = store
        self.s = Session()

    # ------------------------------------------------------------ parsing
    def _context_for_llm(self) -> str:
        ctx = {
            "active_task": self.s.active_task,
            "active_search": self.s.search,
            "active_match": self.s.match,
            "restrictions": {"groups": self.s.exclude_groups, "ingredients": self.s.exclude_ingredients},
            "focus_recipe": self.s.focus_recipe,
            "numbered_results": [f"{i + 1}. {r['title']} [{r['id']}]" for i, r in enumerate(self.s.last_results[:10])],
            "recent_turns": self.s.history[-6:],
        }
        return json.dumps(ctx, ensure_ascii=False)

    def parse(self, text: str) -> dict:
        prompt = f"CONTEXT:\n{self._context_for_llm()}\n\nUSER MESSAGE:\n{text}\n\nReturn the JSON action."
        # NB: schema-constrained mode made the model drop optional fields, so we use plain JSON mode
        # and validate the result against PARSE_SCHEMA in Python instead.
        action = llm.generate_json(SYSTEM + EXAMPLES + "\nReturn ONLY a JSON object with the fields above.", prompt)
        return repair_action(sanitize_action(action), text)

    # ------------------------------------------------------------ main entry
    def handle(self, text: str, action: dict | None = None) -> Reply:
        try:
            action = action if action is not None else self.parse(text)
        except llm.LLMError as e:
            if getattr(e, "rate_limited", False):
                msg = ("⏳ **The chef is catching their breath.** The free Gemini tier allows only a few requests "
                       "per minute and every model is busy right now. Please wait about 30 seconds and send your "
                       "message again. Nothing was guessed; no answer was produced.")
            else:
                msg = "⚠️ **I couldn't reach the language model to understand your message.** Please try again."
            return Reply(markdown=msg + f"\n\n<sub>Details: {str(e)[:300]}</sub>", action={}, error=str(e))
        try:
            reply = self.execute(action)
        except Exception as e:  # never crash the UI
            import traceback
            traceback.print_exc()
            reply = Reply(markdown=f"⚠️ Something went wrong while computing the answer: `{e}`", action=action, error=str(e))
        self.s.history.append({"user": text, "bot_action": action.get("intent"),
                               "bot_summary": reply.markdown[:200]})
        return reply

    # ------------------------------------------------------------ execution
    def _apply_restrictions(self, a: dict) -> list[str]:
        changes = []
        if a.get("clear_restrictions"):
            if self.s.exclude_groups or self.s.exclude_ingredients:
                changes.append("cleared all restrictions")
            self.s.exclude_groups, self.s.exclude_ingredients = [], []
        for g in a.get("add_exclude_groups") or []:
            rg = resolve_group(g)
            if rg and rg not in self.s.exclude_groups:
                self.s.exclude_groups.append(rg)
                changes.append(f"added restriction: no {GROUPS[rg]['label']}")
            elif not rg and g.lower() not in self.s.exclude_ingredients:
                self.s.exclude_ingredients.append(g.lower())
                changes.append(f"added restriction: no {g}")
        for t in a.get("add_exclude_ingredients") or []:
            rg = resolve_group(t)
            if rg:
                if rg not in self.s.exclude_groups:
                    self.s.exclude_groups.append(rg)
                    changes.append(f"added restriction: no {GROUPS[rg]['label']}")
            elif t.lower() not in self.s.exclude_ingredients:
                self.s.exclude_ingredients.append(t.lower())
                changes.append(f"added restriction: no {t}")
        for t in a.get("remove_restrictions") or []:
            rg = resolve_group(t)
            if rg and rg in self.s.exclude_groups:
                self.s.exclude_groups.remove(rg)
                changes.append(f"removed restriction: {GROUPS[rg]['label']}")
            elif t.lower() in self.s.exclude_ingredients:
                self.s.exclude_ingredients.remove(t.lower())
                changes.append(f"removed restriction: {t}")
        return changes

    def _apply_limits(self, target: dict, a: dict):
        if a.get("clear_limits"):
            for k in ("max_steps", "min_steps", "max_ingredients"):
                target.pop(k, None)
        for k in ("max_steps", "min_steps", "max_ingredients"):
            if a.get(k) is not None and a.get(k, 0) > 0:
                target[k] = int(a[k])

    def execute(self, a: dict) -> Reply:
        intent = a.get("intent", "other")
        # a filter stated inside a stats question applies to that question only (unless it is about the
        # active search); everywhere else a stated restriction persists for the session
        changes = [] if (intent == "stats" and not a.get("use_active_search")) else self._apply_restrictions(a)

        if intent == "reset":
            self.s = Session()
            return Reply("Started a new session. All searches, ingredient lists and restrictions are cleared.", a)
        if intent == "help":
            return Reply(render.HELP, a)
        if intent == "unsupported_attribute":
            return Reply(render.unsupported(a.get("unsupported_attribute", ""), UNSUPPORTED_FIELDS,
                                            self.s.focus_recipe), a)
        if intent == "other":
            msg = a.get("reply") or "I can search recipes, compute statistics, match your ingredients, or adapt a recipe."
            return Reply(msg + "\n\n" + render.HELP_SHORT, a)

        if intent == "update_constraints":
            task = self.s.active_task or "search"
            if task == "match" and self.s.match:
                self._apply_limits(self.s.match, a)
                return self._run_match(a, changes)
            if task == "substitute" and self.s.focus_recipe and self.s.match.get("_last_target"):
                return self._run_substitute(a, changes, self.s.focus_recipe, self.s.match["_last_target"])
            self._apply_limits(self.s.search, a)
            if not self.s.search:
                return Reply("Noted. Active restrictions: **" + self.s.restriction_text() +
                             "**. They will apply to your next search or ingredient match.", a)
            return self._run_search(a, changes)

        if intent == "search":
            if a.get("search_mode") == "refine" and self.s.search:
                sf = self.s.search
                if a.get("query"):
                    sf["query"] = (sf.get("query", "") + " " + a["query"]).strip()
                for k in ("include_all", "include_any"):
                    if a.get(k):
                        sf[k] = list(dict.fromkeys(sf.get(k, []) + a[k]))
            else:
                self.s.search = {k: a[k] for k in ("query", "include_all", "include_any", "sources") if a.get(k)}
            self._apply_limits(self.s.search, a)
            return self._run_search(a, changes)

        if intent == "more":
            if self.s.active_task == "search":
                self.s.last_offset += 5
                return self._run_search(a, changes, offset=self.s.last_offset)
            if self.s.active_task == "match":
                self.s.match["k"] = self.s.match.get("k", 5) + 5
                return self._run_match(a, changes)
            return Reply("There is no active result list to extend.", a)

        if intent == "show_recipe":
            rid = self._resolve(a.get("recipe_ref", ""))
            if not rid:
                return Reply("I couldn't tell which recipe you mean. Please give the result number or recipe ID.", a)
            self.s.focus_recipe = rid
            rec = self.store.get(rid)
            return Reply(render.recipe(rec, a.get("show") or ["ingredients", "instructions"],
                                       self.s.exclude_groups, self.s.exclude_ingredients, self.store), a,
                         data={"recipe": rid})

        if intent == "stats":
            return self._run_stats(a, changes)

        if intent == "match":
            mode = a.get("pantry_mode") or ("new" if a.get("pantry") else "keep")
            m = self.s.match if (mode != "new" and self.s.match) else {"pantry": []}
            if mode == "new":
                m = {"pantry": list(a.get("pantry") or [])}
            elif mode == "add":
                m["pantry"] = list(dict.fromkeys(m.get("pantry", []) + (a.get("pantry") or [])))
            elif mode == "remove":
                rem = {p.lower() for p in a.get("pantry") or []}
                m["pantry"] = [p for p in m.get("pantry", []) if p.lower() not in rem]
            if a.get("prefer_fewer_steps") is not None:
                m["prefer_fewer_steps"] = bool(a["prefer_fewer_steps"])
            if a.get("k"):
                m["k"] = max(1, min(int(a["k"]), 20))
            if a.get("query") and mode == "new":
                m["query"] = a["query"]
            self._apply_limits(m, a)
            self.s.match = m
            return self._run_match(a, changes)

        if intent == "substitute":
            target = a.get("target_ingredient") or ""
            if not target:
                return Reply("Which ingredient should I replace or remove?", a)
            ref = a.get("recipe_ref", "")
            rid = self._resolve(ref, must_contain=target) if ref else self.s.focus_recipe
            if not rid:
                return Reply("Which recipe should I adapt? Give a result number, recipe ID or dish name.", a)
            return self._run_substitute(a, changes, rid, target, picked_from=ref if ref and ref != rid else "")

        return Reply("Sorry, I did not understand that. " + render.HELP_SHORT, a)

    # ------------------------------------------------------------ runners
    def _filter(self, base: dict, with_restrictions=True) -> Filter:
        f = Filter(**{k: v for k, v in base.items() if k in Filter.__dataclass_fields__})
        if with_restrictions:
            f.exclude_groups = list(dict.fromkeys(f.exclude_groups + self.s.exclude_groups))
            f.exclude_ingredients = list(dict.fromkeys(f.exclude_ingredients + self.s.exclude_ingredients))
        return f

    def _run_search(self, a, changes, offset=0) -> Reply:
        self.s.active_task = "search"
        if offset == 0:
            self.s.last_offset = 0
        f = self._filter(self.s.search)
        df, audit = self.store.apply(f)
        if "score" in df:
            df = df.sort_values(["score", "n_steps"])
        page = df.iloc[offset:offset + 5]
        self.s.last_results = [{"id": i, "title": t} for i, t in zip(df.index[:50], df["title"][:50])]
        if offset == 0 and len(page):
            self.s.focus_recipe = page.index[0]
        md = render.search(self.store, f, df, page, audit, changes, offset)
        return Reply(md, a, data={"filter": f.to_dict(), "audit": audit, "ids": list(page.index)})

    def _run_stats(self, a, changes) -> Reply:
        base = {k: a[k] for k in ("query", "include_all", "include_any", "sources") if a.get(k)}
        self._apply_limits(base, a)
        if a.get("use_active_search") and self.s.search:
            merged = copy.deepcopy(self.s.search)
            for k, v in base.items():
                merged[k] = (merged.get(k, []) + v) if isinstance(v, list) else v
            f = self._filter(merged, with_restrictions=True)
        else:
            f = self._filter(base, with_restrictions=False)
            f.exclude_groups = [resolve_group(g) for g in (a.get("add_exclude_groups") or []) if resolve_group(g)]
            f.exclude_ingredients = [t for t in (a.get("add_exclude_ingredients") or []) if not resolve_group(t)]
            f.exclude_groups += [resolve_group(t) for t in (a.get("add_exclude_ingredients") or []) if resolve_group(t)]
        op = a.get("stats_op") or "count"
        metric = a.get("metric") or ("steps" if op == "rank" else "ingredients")
        if op == "count":
            res = stats.count(self.store, f)
        elif op == "describe":
            res = stats.describe(self.store, f, metric, bool(a.get("group_by_source")))
        elif op == "rank":
            res = stats.rank(self.store, f, metric, a.get("order") or "asc", max(1, min(int(a.get("k") or 5), 25)))
            self.s.last_results = [{"id": r["id"], "title": r["title"]} for r in res["results"]]
            if res["results"]:
                self.s.focus_recipe = res["results"][0]["id"]
        elif op == "top_ingredients":
            res = stats.top_ingredients(self.store, f, max(1, min(int(a.get("k") or 10), 30)),
                                        a.get("include_staples", True) is not False)
        else:
            names = (a.get("include_all") or []) + (a.get("include_any") or []) or [a.get("query", "")]
            res = {"type": "lookup", "items": [stats.ingredient_lookup(self.store, n) for n in names if n]}
        self.s.active_task = self.s.active_task if op == "lookup" else "stats"
        return Reply(render.stats(res, changes), a, data=res)

    def _run_match(self, a, changes) -> Reply:
        self.s.active_task = "match"
        m = self.s.match
        if not m.get("pantry"):
            return Reply("Tell me which ingredients you have (e.g. *I have eggs, onions, tomatoes and rice*).", a)
        req = MatchRequest(pantry=m["pantry"], exclude_ingredients=list(self.s.exclude_ingredients),
                           exclude_groups=list(self.s.exclude_groups), max_steps=m.get("max_steps"),
                           max_ingredients=m.get("max_ingredients"), query=m.get("query", ""),
                           prefer_fewer_steps=m.get("prefer_fewer_steps", False), k=m.get("k", 5))
        res = match(self.store, req)
        self.s.last_results = [{"id": r["id"], "title": r["title"]} for r in res.get("results", [])]
        if self.s.last_results:
            self.s.focus_recipe = self.s.last_results[0]["id"]
        return Reply(render.match(res, changes), a, data=res)

    def _run_substitute(self, a, changes, rid, target, picked_from="") -> Reply:
        self.s.active_task = "substitute"
        self.s.focus_recipe = rid
        self.s.match["_last_target"] = target
        res = substitute(self.store, rid, target, list(self.s.exclude_groups), list(self.s.exclude_ingredients))
        rewritten = None
        if res.get("adapted") and llm.available():
            rewritten = _llm_rewrite(res, self.s)
        return Reply(render.substitution(res, changes, picked_from, rewritten), a, data=res)

    # ------------------------------------------------------------ references
    def _resolve(self, ref: str, must_contain: str = "") -> str | None:
        ref = (ref or "").strip()
        if not ref:
            return self.s.focus_recipe or None
        if self.store.get(ref):
            return ref
        m = re.fullmatch(r"#?(\d{1,2})", ref)
        if m and self.s.last_results:
            i = int(m.group(1)) - 1
            if 0 <= i < len(self.s.last_results):
                return self.s.last_results[i]["id"]
        for r in self.s.last_results:
            if r["title"].lower() == ref.lower():
                return r["id"]
        # title search: best-ranked recipe whose TITLE matches, preferring ones containing the target
        f = Filter(query=ref)
        df, _ = self.store.apply(f)
        if df.empty:
            return None
        df = df.sort_values("score")
        if must_contain:
            tgt = resolve_group(must_contain)
            ids = self.store.group_ids(tgt) if tgt else self.store.recipes_mentioning(must_contain)
            hit = df[df.index.isin(ids)]
            if len(hit):
                return hit.index[0]
        return df.index[0]


REWRITE_SYSTEM = """You rewrite recipe instructions after an ingredient substitution.
Rules: use ONLY the ingredients in the adapted ingredient list; do not add new ingredients; keep the same
order and number of steps where possible; mention the substitute where the original ingredient was used,
including any preparation the ratio requires (e.g. mixing flaxseed with water). Return JSON {"steps": [..]}."""


def _llm_rewrite(res: dict, s: Session) -> list[str] | None:
    from .groups import line_violation
    try:
        prompt = json.dumps({"original_steps": res["recipe"]["steps"],
                             "adapted_ingredients": res["adapted"]["ingredients"],
                             "notes": res["adapted"]["notes"]}, ensure_ascii=False)
        out = llm.generate_json(REWRITE_SYSTEM, prompt,
                                {"type": "object", "properties": {"steps": {"type": "array", "items": {"type": "string"}}},
                                 "required": ["steps"]})
        steps = [x for x in out.get("steps", []) if isinstance(x, str) and x.strip()]
        if not steps:
            return None
        # validation: rewritten steps must not reintroduce restricted ingredients
        from .normalize import word_pattern
        target_pat = word_pattern(res["entry_key"] or res["target"]) if (res.get("entry_key") or res.get("target")) else None
        for st in steps:
            for g in s.exclude_groups:
                if line_violation(st, g):
                    return None
            if target_pat and target_pat.search(st.split("(")[0]) and res["entry_key"] not in ("cream",):
                return None
        return steps
    except Exception:
        return None
