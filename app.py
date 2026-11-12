"""Streamlit chat UI.  Run:  streamlit run app.py"""
import json
import re
from pathlib import Path

import streamlit as st

from recipebot import llm
from recipebot.agent import Agent
from recipebot.config import DB_PATH, GEMINI_MODEL
from recipebot.groups import GROUPS

st.set_page_config(page_title="Recipe Box Kitchen", page_icon="🍅", layout="wide",
                   initial_sidebar_state="expanded")


def inject_css():
    css = (Path(__file__).parent / "assets" / "style.css").read_text(encoding="utf-8")
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)                 # drop comments
    css = "\n".join(l for l in css.splitlines() if l.strip())        # NO blank lines: a blank line would end
    st.markdown(f"<style>\n{css}\n</style>", unsafe_allow_html=True)  # the HTML block and leak CSS as text


def html(s: str, where=st):
    """Render an HTML snippet. Blank lines removed so markdown never breaks the block."""
    where.markdown("\n".join(l for l in s.splitlines() if l.strip()), unsafe_allow_html=True)


inject_css()


@st.cache_resource(show_spinner="Warming up the kitchen: loading 122,929 recipes (first run only)…")
def load_store():
    from recipebot.store import get_store
    s = get_store()
    _ = s.recipe_names  # warm index
    return s


if not DB_PATH.exists():
    st.error("Database not found. Build it first:  `python -m recipebot.ingest`")
    st.stop()

store = load_store()
if "agent" not in st.session_state:
    st.session_state.agent = Agent(store)
    st.session_state.messages = []
agent: Agent = st.session_state.agent

TASKS = [
    ("c1", "🔎", "Find & ask", "Search by dish or ingredient and get exact, computed statistics.",
     [("Find a penne pasta recipe", "🍝  Find a penne pasta recipe"),
      ("I cannot have dairy", "🥛  …then: I cannot have dairy"),
      ("How many recipes contain both garlic and onion? What percentage of the dataset is that?",
       "🧄  Garlic + onion: how many?"),
      ("Among recipes containing tomato, which five have the fewest instruction steps?",
       "🍅  Tomato recipes with the fewest steps")]),
    ("c2", "🧺", "Cook from your pantry", "Rank real recipes by what's in your kitchen – hard limits always respected.",
     [("I have eggs, onions, tomatoes, and rice. Suggest five recipes using as many of these as possible, with no "
       "peanuts and at most six instruction steps. Prefer recipes that need the fewest additional ingredients.",
       "🥚  I have eggs, onions, tomatoes & rice"),
      ("Exclude eggs too", "🚫  …then: exclude eggs too"),
      ("I have tofu and kale. Find me a lasagna with at most 1 step, no dairy and no gluten.",
       "🥬  Tofu & kale lasagna (no-match test)")]),
    ("c3", "🔄", "Smart swaps", "Adapt a recipe using evidence from similar recipes in the dataset.",
     [("Take a banana bread recipe and remove eggs. Use only ingredients that appear in the dataset. "
       "What replacements have evidence from related baked recipes, and how might the texture change?",
       "🍌  Banana bread without eggs"),
      ("Find a creamy penne pasta with heavy cream", "🧈  Find a creamy penne with cream"),
      ("Replace cream in this pasta recipe without dairy. Prefer examples from recipes already in the database.",
       "🌱  …then: replace the cream, no dairy")]),
]


def ask(text):
    st.session_state.pending = text


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    ok = llm.available()
    s = agent.s
    html(f"""
<div class="side-brand"><div class="logo">🍅</div>
<div><div class="name">Your kitchen</div><div class="tag">what I'm keeping in mind</div></div></div>
<span class="status"><span class="dot" style="background:{'#2B8A3E' if ok else '#E03131'}"></span>
{('Chef is ready · ' + GEMINI_MODEL) if ok else 'No Gemini key – add it to .env'}</span>
""")
    chips = [GROUPS[g]["label"] for g in s.exclude_groups] + s.exclude_ingredients
    html("<div class='side-h'>🚫 Restrictions</div><div class='side-box'>" +
         (" ".join(f"<span class='chip no'>no {c}</span>" for c in chips) or
          "<span class='chip off'>none yet – try “I can't have dairy”</span>") + "</div>")
    pantry = s.match.get("pantry") or []
    limits = []
    if s.match.get("max_steps"):
        limits.append(f"≤ {s.match['max_steps']} steps")
    if s.match.get("k"):
        limits.append(f"top {s.match['k']}")
    html("<div class='side-h'>🧺 In your pantry</div><div class='side-box'>" +
         (" ".join(f"<span class='chip have'>{p}</span>" for p in pantry) or
          "<span class='chip off'>tell me what you have</span>") +
         ("<br>" + " ".join(f"<span class='chip off'>{x}</span>" for x in limits) if limits else "") + "</div>")
    if s.search.get("query"):
        html(f"<div class='side-h'>🔎 Active search</div><div class='side-box'><span class='chip off'>"
             f"{s.search['query']}</span></div>")
    if s.focus_recipe and s.focus_recipe in store.recipes.index:
        r = store.recipes.loc[s.focus_recipe]
        html(f"<div class='side-h'>🔪 On the cutting board</div><div class='side-box recipe-card'><b>{r['title']}</b>"
             f"<div class='meta'>{int(r['n_steps'])} steps · {int(r['n_ingredients'])} ingredients</div>"
             f"<div class='meta'><code>{s.focus_recipe}</code></div></div>")
    st.write("")
    if st.button("🧽  Clean the counter · new chat", use_container_width=True):
        st.session_state.agent = Agent(store)
        st.session_state.messages = []
        st.rerun()
    html(f"<div class='side-foot'>📚 {store.total:,} recipes from Allrecipes, Epicurious &amp; Food Network.<br>"
         "Every number and recipe ID is computed from the dataset – the AI only reads your question.</div>")

# ------------------------------------------------------------------ hero
html(f"""
<div class="hero">
<div class="deco">🍝🥖🍅</div>
<h1>Recipe Box Kitchen</h1>
<p>Your grounded cooking companion – search {store.total:,} real recipes, crunch the numbers,
cook from your pantry and swap ingredients with evidence.</p>
<div class="pills"><span class="pill">🍳 {store.total:,} recipes</span><span class="pill">📊 exact statistics</span>
<span class="pill">🛡️ allergy-aware</span><span class="pill">🔗 cited recipe IDs</span></div>
</div>""")

if not llm.available():
    st.warning("No `GEMINI_API_KEY` found in `.env` – add your Google AI Studio key and restart the app.")

# ------------------------------------------------------------------ task cards (welcome screen)
if not st.session_state.messages:
    cols = st.columns(3, gap="medium")
    for col, (cls, emo, title, sub, exs) in zip(cols, TASKS):
        with col:
            with st.container(key=f"card-{cls}"):
                html(f"<div class='card-head'><div class='card-icon {cls}'>{emo}</div>"
                     f"<div class='card-title'>{title}</div></div><div class='card-sub'>{sub}</div>")
                for full, label in exs:
                    st.button(label, key=f"ex-{label}", on_click=ask, args=(full,), use_container_width=True)
    html("<div class='hint'>…or just type below. Follow-ups like <i>“I can't have dairy”</i>, "
         "<i>“show #2”</i> or <i>“exclude eggs too”</i> keep the context.</div>")

# ------------------------------------------------------------------ chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"], avatar="🙂" if m["role"] == "user" else "🧑‍🍳"):
        st.markdown(m["content"], unsafe_allow_html=True)
        if m.get("action"):
            with st.expander("🔍 How this was computed"):
                st.markdown("**Your message, as understood (LLM → structured action):**")
                st.json(m["action"])
                if m.get("data"):
                    st.markdown("**Raw tool output (computed from the dataset):**")
                    st.json(m["data"], expanded=False)

if st.session_state.messages:
    html("<div class='side-h'>Quick follow-ups</div>")
    qcols = st.columns(4)
    quick = [("📖  Instructions for #1", "Show the instructions for #1"), ("🥛  I cannot have dairy", "I cannot have dairy"),
             ("➕  Show more", "Show more"), ("🔥  Calories?", "How many calories does it have?")]
    for c, (label, q) in zip(qcols, quick):
        c.button(label, key=f"q-{q}", on_click=ask, args=(q,), use_container_width=True)

prompt = st.chat_input("What are we cooking today?  e.g. “Find a penne pasta recipe”")
if st.session_state.get("pending"):
    prompt = st.session_state.pop("pending")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🙂"):
        st.markdown(prompt)
    with st.chat_message("assistant", avatar="🧑‍🍳"):
        with st.spinner("Chopping through 122,929 recipes…"):
            reply = agent.handle(prompt)
        st.markdown(reply.markdown, unsafe_allow_html=True)
    data = reply.data
    try:
        data = json.loads(json.dumps(data, default=str)) if data else None
    except Exception:
        data = None
    st.session_state.messages.append({"role": "assistant", "content": reply.markdown, "action": reply.action,
                                      "data": data})
    st.rerun()
