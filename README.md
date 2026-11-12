# Recipe Box Assistant: a grounded recipe chatbot

A conversational chatbot over the [Recipe Box dataset](https://eightportions.com/datasets/Recipes/)
(all three files: Allrecipes, Epicurious and Food Network). It does three things:

1. **Search & factual Q&A.** Finds recipes and answers statistics questions.
2. **Ingredient matching.** Ranks existing recipes against the ingredients you have.
3. **Evidence-based substitution.** Adapts a recipe, citing recipes in the dataset as evidence.

**Design principle:** the LLM (Gemini) only *interprets* the user's message, turning it into a
structured JSON action. Every recipe, ID, count, average, ranking and piece of evidence is computed
by deterministic Python code (pandas + SQLite) over the complete dataset. The answer text is built
from that tool output, so the LLM has no opportunity to invent facts.

```
user message ──► Gemini (intent → JSON action) ──► session state (restrictions, active search, pantry, focus recipe)
                                                        │
                       ┌────────────────┬───────────────┼─────────────────┬──────────────────┐
                       ▼                ▼               ▼                 ▼                  ▼
                 search (FTS5      stats (pandas,   matcher (scored   substitute (local   unsupported →
                 BM25 + filters)   full filtered    ranking, hard     evidence + curated   explicit refusal
                                   set)             constraints)      role table)
                       └────────────────┴───────────────┴─────────────────┘
                                         ▼
                        markdown rendered ONLY from tool output (+ "How this was computed" panel)
```

---

## 1. Setup

Requires Python 3.10+ (tested on 3.13, Windows).

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# 1) Data: download https://eightportions.com/recipes_raw.zip and unzip into data/raw/
#    (recipes_raw_nosource_ar.json, recipes_raw_nosource_epi.json, recipes_raw_nosource_fn.json)
# 2) Build the database (about 6 minutes; creates data/recipes.db)
python -m recipebot.ingest

# 3) API key: copy .env.example to .env and paste a free Google AI Studio key
#    (https://aistudio.google.com/apikey)

# 4) Run
streamlit run app.py              # web UI (recommended)
python cli.py                     # or the command-line chat
```

Tests and evaluation:

```bash
python -m pytest -q tests                 # unit tests: normalization, matching, restriction groups
python eval/run_eval.py                   # deterministic evaluation (no API key needed)
python eval/run_eval.py --llm             # + LLM parsing eval, one-shot baseline, end-to-end transcripts
```

## 2. Project layout

| Path | Purpose |
|---|---|
| `recipebot/ingest.py` | Loads all 3 JSON files, cleans them, parses ingredients and builds SQLite (`recipes`, `ingredients`, `flags`, FTS5 index, `meta`) |
| `recipebot/normalize.py` | Ingredient-line parser (quantity, unit, name) and name normalization |
| `recipebot/groups.py` | Restriction groups (dairy, egg, peanut, tree nut, gluten, shellfish, fish, soy, meat, vegetarian, vegan) |
| `recipebot/store.py` | In-memory pandas tables, the shared `Filter` spec, full-text search, inclusion and exclusion rules |
| `recipebot/stats.py` | Counts, percentages, mean/median, rankings, top ingredients |
| `recipebot/matcher.py` | Task 2 scoring and ranking |
| `recipebot/substitute.py`, `substitutes_table.py` | Task 3 evidence engine, plus a curated role table (labeled as general knowledge) |
| `recipebot/agent.py` | Session state, LLM intent parsing, action execution |
| `recipebot/render.py` | Markdown answers built only from tool output |
| `app.py` / `cli.py` | Streamlit UI / CLI |
| `eval/run_eval.py`, `eval/results.md` | Reproducible evaluation and report |
| `tests/` | Unit tests |

## 3. Data handling

* **Stable ID**: the original dataset key (e.g. `rmK12Uau.ntP510KeImX506H6Mr6jTu`). Keys are unique across
  all three files (verified), so no prefix is needed; the `source` column records the origin.
* **Originals preserved**: `ingredients_raw` (exact original list), `ingredients` (the same list with only
  `ADVERTISEMENT` markers removed) and `instructions` (exact original text) are stored for every recipe.
* **Usable recipes**: 125,164 raw records → **122,929 usable**. Dropped: 517 empty records, 52 without a title,
  1,625 without ingredients, 41 without instructions. Every percentage uses 122,929 as its denominator,
  and the answer says so.

## 4. Key decisions

### Retrieval
* **SQLite FTS5 with BM25** over title (weight 10), ingredient text (3) and instructions (0.5), with
  Porter stemming. Matching runs in three tiers, and the answer reports which tier was used:
  all terms in title/ingredients → all terms anywhere → any term.
* Natural-language requests are turned into keywords plus structured filters by the LLM
  (e.g. "something cozy for winter" becomes `stew`). I chose this over embedding search to keep retrieval
  explainable and the install light (no PyTorch). The trade-off: highly abstract requests rely on the
  LLM choosing good keywords.
* **Search vs. aggregates**: search returns the top 5 by BM25. Every statistic runs on the **complete**
  filtered set (`Store.apply` returns all matching rows), never on the displayed results.

### Ingredient normalization
Each ingredient line is parsed into a quantity, a unit and a canonical name. The parser:
* removes quantities, ranges, metric equivalents ("3 tablespoons/45 ml"), package sizes
  ("2 (10.75 ounce) cans" becomes 21.5 oz), preparation words (chopped, minced, fresh…) and size words;
* takes the first real alternative ("butter or margarine" → butter);
* singularizes the head noun;
* applies a small, explicit synonym table (e.g. green onion → scallion, all-purpose flour → flour,
  kosher salt → salt, heavy whipping cream → heavy cream);
* splits "salt and pepper" into two ingredients.

**Different products stay different:** `garlic powder` ≠ `garlic`, `tomato paste` ≠ `tomato`,
`scallion` ≠ `onion`, `rice vinegar` ≠ `rice`.

* **Inclusion rule** ("recipe contains X"): head-noun match. `red onion` counts as onion, and
  `chicken breast` counts as chicken (a trailing cut/part word is stripped). `onion powder` does not count.
* **Exclusion rule** (hard constraints): deliberately broader. Any whole-word mention in the ingredient line
  counts, plus restriction-group term lists with explicit exceptions (coconut milk is not dairy, peanut
  butter is not dairy, eggplant is not egg, nutmeg is not a tree nut). This over-excludes on purpose, because
  a missed allergen is worse than a missed recipe.
* Count answers report the strict count, which names were merged, and a **sensitivity count** under the
  broad rule. For example, garlic + onion is 16,090 recipes (13.09%) strict and 20,240 (16.46%) broad.
* About 0.6% of lines produce no name. These are mostly notes ("*available at specialty stores").

### Step counting
A step is one non-empty line/paragraph of the original instructions, after removing boilerplate
lines ("Watch how to make this recipe.", "Photograph by …", copyright and credits). Ties are broken by
sentence count, then by ID.

Known limitation: some sources write the whole method as one paragraph, so their step counts are low.
Rankings report how many recipes tie at the cut-off, which makes this visible.

### Ranking (Task 2)
`score = 10 × matched − 2 × extra_required (− 0.5 × steps if fewer steps is preferred)`.
* **matched**: your ingredients that the recipe uses.
* **extra_required**: recipe ingredients you didn't list. Salt, black pepper and water are assumed staples
  and listed separately.
* **Not used from your list**: your ingredients that the recipe doesn't use.
* Ties are broken by fewer extras, then fewer steps, then ID.

Every result shows its score breakdown. Hard constraints (exclusions, restriction groups, maximum
steps/ingredients, dish keywords) are **filters, never score terms**, so they can't be traded off.
If nothing is eligible, the answer says **"No eligible recipe found"** and then shows the closest
alternatives, each labeled with the constraint it violates and the offending ingredient line.
If a pantry item conflicts with a restriction (e.g. eggs after "exclude eggs too"), it is ignored and
the answer says so.

**Quantities:** compared only when both sides are in the same dimension: count vs count (eggs),
mass vs mass, or volume vs volume (converted to g or ml). Otherwise the answer says
"quantity NOT verified" and gives the reason (e.g. grams vs cups would need a density the dataset
doesn't provide).

### Substitution evidence (Task 3)
1. **Similar-dish pool**: recipes whose titles share the dish words (e.g. *banana bread*), ranked by
   ingredient Jaccard similarity (top 400). Baked dishes only match other baked recipes.
2. The pool is split into recipes **with** and **without** the target ingredient.
3. **Candidates** come from a curated role table: roles, typical ratios and *predicted* effects. This table
   is **general culinary knowledge** and is labeled as such in every answer. For each candidate the
   engine measures:
   * *direct support*: similar recipes without the target that use this candidate;
   * *lift*: how much more common the candidate is when the target is absent;
   * *signal support*: recipes anywhere in the dataset with *eggless / dairy-free / vegan …* titles
     that use it.
4. **Evidence strength** is strong, moderate, weak or none. Candidates get flags for these cases:
   * **conflicting**: about as common in recipes that keep the original;
   * **generic**: common in the dish anyway (e.g. olive oil in pasta), so the evidence is indirect;
   * **already in recipe**: e.g. adding more banana to banana bread is circular.
5. **Restrictions are hard.** Candidates that violate them are rejected and the reason is shown.
   With "dataset only", candidates that never appear in the dataset are rejected. When a restriction is
   active, **every other violating line is adapted too** (e.g. Parmesan in a dairy-free pasta →
   nutritional yeast) or explicitly removed. The final ingredient list is re-checked.
6. **Refusal**: if the target is the core of the dish (eggs in a frittata) and the evidence isn't strong,
   or if no candidate has evidence, the answer is "No reliable adaptation" rather than an invented swap.
7. **Output**: the adapted ingredient list, revised instructions (a deterministic swap; optionally
   re-phrased by the LLM, then validated against the restrictions and the target, with fallback if the
   check fails), the supporting recipe IDs with the actual matching lines, alternatives, predicted
   effects, and a safety note ("recipe text only, not a medical assurance").

### LLM usage (intent parsing only)
* **Models:** `gemini-3.5-flash` with low thinking (about 2–4 s per turn) as the primary model. It falls back
  automatically to `gemini-3.5-flash-lite` and then `gemini-3.8-flash` on overload (503), timeout (504),
  retired model (404) or rate limits. Each request has a 60 s timeout, so the UI can't hang. All of this is
  configurable in `.env`.
* **Plain JSON mode, validated in Python:** schema-constrained output made the model silently drop
  optional fields (e.g. "no peanuts" and "max 6 steps" disappeared). So the model returns plain JSON, guided
  by worked examples in the prompt, and `sanitize_action` validates it against the schema: unknown keys
  dropped, enums checked, types coerced.
* **Deterministic safety net:** if a statistics question loses its ingredients
  ("contain both X and Y"), `repair_action` recovers them from the text, and the action records
  `_repaired`. The parsed action is always visible under "How this was computed".

### Uncertainty and unsupported questions
The dataset has no nutrition, time, servings, rating or cuisine fields. Questions about these get an
explicit refusal that says what is missing. Nothing is estimated.

### External search
Not used. The assignment makes it optional. The system prefers to state when local evidence is weak
or missing rather than mix in web sources.

### Conversation state
The session keeps:
* **restrictions**, which persist across searches until lifted;
* the **active search filter**;
* the **pantry and match limits**;
* the **last numbered results**;
* the **focus recipe**.

Follow-ups like "I cannot have dairy" or "exclude eggs too" re-run whatever task is active.
"The second one" resolves against the last result list. A filter stated inside a statistics question
("how many vegan recipes…") applies only to that question.

## 5. Evaluation
See **[eval/results.md](eval/results.md)**. In brief:
* **Tool-level cases** run through the same agent code path as the chat. They are checked by an
  **independent oracle** that re-reads the raw JSON files with separate, simple code.
* **LLM parsing accuracy** on the example prompts.
* A **one-shot LLM baseline**: Gemini answering dataset questions directly, compared with the
  grounded system.
* **End-to-end transcripts** covering multi-turn follow-ups, an aggregate query, a no-match case, a
  substitution with weak evidence (refused), and an unsupported-attribute refusal.

**Headline results:** 12/12 tool-level cases pass against the oracle, 10/10 messages are parsed correctly,
and 41/41 unit tests pass.

**One-shot LLM vs. grounded system:**

| Question | One-shot Gemini | This system |
|---|---|---|
| Recipes with garlic + onion | 36,499 / **68,452** / 36,251 across three runs (wrong and inconsistent; no denominator) | **16,090 (13.09%)** strict, 20,240 broad, of 122,929; the broad count matches the oracle exactly, every run |
| 5 tomato recipes with fewest steps | Invented short IDs (13861, 201103…; **0 exist**) and a claimed minimum of 2 steps in one run; only generic pandas code in another | 5 real IDs with 1 step each (the true minimum); step counts re-derived from raw text; 1,727 ties disclosed |
| Banana bread calories and time | Fabricated "200–250 kcal, about 1 h" | Refuses and names the missing fields |

### Requirement checklist
| Requirement | Where |
|---|---|
| All 3 JSON files, stable IDs, original title/ingredients/instructions | `ingest.py` (`id`, `ingredients_raw`, `instructions`) |
| Runnable chatbot with follow-ups | `app.py` (Streamlit), `cli.py`; session state in `agent.py` |
| Search by dish, ingredient or natural language; titles + IDs; ingredients/instructions on request | `store.fts`, `Agent._run_search`, `show_recipe` |
| A new restriction applies to the active search | `update_constraints` re-runs the active task (eval `T1-search-refine`) |
| Counts, common ingredients, average/median, ranking by steps; states filters, N, method | `stats.py`, `render.stats` (eval `T1-count`, `T1-rank`, `T1-describe`) |
| Aggregates on the complete filtered set | `Store.apply` returns every matching row |
| No fabricated attributes; say what is missing | `unsupported_attribute` intent (eval `T1-unsupported`) |
| Pantry with optional quantities, hard constraints, preferences | `matcher.py` (`MatchRequest`, `_quantity_check`) |
| Justified normalization; different products stay different | `normalize.py`, `store.inclusion_match` |
| Explainable ranking; matched / missing / extra required | `matcher._explain`, `render.match` |
| Never relax silently; no-match + clearly marked violating alternatives | `matcher.match` (eval `T2-no-match`) |
| Quantity use, or "not verified" | `_quantity_check` (eval `T2-quantity`) |
| "Exclude eggs too" updates results | eval `T2-match-followup` |
| Substitutes respecting restrictions, with ingredient role | `substitute.py`, `substitutes_table.py` |
| Local dataset first; cite IDs; flag indirect evidence | evidence pool, signal recipes, `examples`, "evidence is indirect" note |
| Updated ingredient list + revised instructions; predicted vs verified effects | `adapt`, `render.substitution` |
| Hard restrictions, no medical claims, refuse when unsupported | `check_adapted`, safety note, `unsupported` status (eval `T3-weak-evidence`) |
| 5+ test conversations, eval set, what worked/failed, one-shot comparison | `eval/run_eval.py`, `eval/results.md`, this section |

### What failed during development (and how it was fixed)
| Failure observed | Why | Fix |
|---|---|---|
| Dairy-free creamy penne still contained "1 cup grated Parmesan cheese" | Only the named target (cream) was substituted | Secondary pass: every remaining line that violates a restriction is substituted or removed, then the list is re-validated |
| Banana bread without eggs suggested "more banana" | The candidate was already in the recipe and about as common in egg-containing banana breads (circular, conflicting evidence) | Flags for already-in-recipe, conflicting and generic, ranked below independent evidence |
| Egg substitute for "Zucchini Scallion Frittata **Cups**" used a pool of unrelated "cups" recipes | The fallback used the last title word | Generic title words skipped; the pool tries word pairs before single words |
| Applesauce/vinegar proposed for a savory frittata | Candidates weren't tied to the dish type | Candidates carry contexts (baked/savory); mismatches are marked *unsuitable* |
| "1 1/2 cups 1/4-inch diced red onion", "3 tablespoons/45 ml tomato paste" → no name | Size and metric-equivalent tokens confused the parser | Size and equivalent stripping; unparsed lines fell from 7,258 to under 1,000 |
| "2 ribs celery" flagged as meat | "ribs" was in the meat term list | Replaced with "spare ribs", "back ribs" and similar, plus a celery exception |
| "Juice of 1 lemon" normalized to "juice" | Phrase order | A rule maps it to "lemon juice" |
| "How many recipes contain both garlic and onion?" answered **122,929 (100%)** in the UI | The LLM's JSON-schema mode dropped `include_all`, so the count ran with no filter | Switched to plain JSON with worked examples, added Python-side validation and a deterministic repair rule, and the answer always prints its filters, so the error would have been visible |
| `gemini-2.5-flash` returned 404 for a new API key; `gemini-3.8-flash` timed out or was overloaded | Model retired for new users; heavy demand | `gemini-3.5-flash` (low thinking) plus a fallback chain and request timeouts |
| Raw CSS appeared as text on the page | Streamlit markdown ends an HTML block at a blank line, so the style block "leaked" | CSS moved to `assets/style.css`, injected with comments and blank lines stripped; all HTML snippets are rendered blank-line-free |
| Rate-limit errors were shown raw to the user | Free tier allows only a few requests per minute | Longer back-off (6 s, 12 s) before fallback, and a friendly "wait ~30 s" message with details collapsed |
| Matching took 76 s | pandas groupby building per-recipe name sets | A plain-Python index, now 2–3 s |

## 6. Known limitations
* Step counts depend on each source's paragraphing (see above).
* Restriction detection works on recipe text. It can't see hidden ingredients inside processed products.
* Natural-language search depends on the LLM's keyword extraction; there is no embedding search.
* Quantity checks don't convert mass to volume (no density data).
* Substitution evidence is co-occurrence in similar recipes, not tested outcomes. Effects are labeled
  as predictions.
