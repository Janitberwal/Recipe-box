"""Ingredient-line parsing and ingredient-name normalization.

Design (documented in README):
* Each raw ingredient line -> cleaned display line, optional quantity + unit, and one
  or more canonical ingredient *names*.
* Normalization removes quantities, units, preparation words and size words,
  singularizes the head noun and applies a small, explicit synonym table.
* Product distinctions that change a match are kept: "garlic powder" is not "garlic",
  "tomato paste" is not "tomato", "green onion" (-> scallion) is not "onion".
* Inclusion matching is head-noun based: query "onion" matches "red onion" and
  "onion" but not "onion powder" (head = powder).
* Exclusion matching (hard constraints) is deliberately broader / conservative:
  any whole-word occurrence in the line counts (see `line_mentions`).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from fractions import Fraction

UNICODE_FRACTIONS = {
    "½": " 1/2", "⅓": " 1/3", "⅔": " 2/3", "¼": " 1/4", "¾": " 3/4", "⅕": " 1/5",
    "⅛": " 1/8", "⅜": " 3/8", "⅝": " 5/8", "⅞": " 7/8", "⅙": " 1/6",
}

# unit word -> (canonical unit, dimension, factor to base unit)
# base units: volume = ml, mass = g, count = each
_U = {}
def _add(names, canon, dim, factor):
    for n in names:
        _U[n] = (canon, dim, factor)

_add(["teaspoon", "teaspoons", "tsp", "tsps", "t"], "tsp", "volume", 4.93)
_add(["tablespoon", "tablespoons", "tbsp", "tbsps", "tbs", "tbl", "tbsp.", "tb"], "tbsp", "volume", 14.79)
_add(["cup", "cups", "c"], "cup", "volume", 236.6)
_add(["pint", "pints", "pt"], "pint", "volume", 473.2)
_add(["quart", "quarts", "qt"], "quart", "volume", 946.4)
_add(["gallon", "gallons", "gal"], "gallon", "volume", 3785.0)
_add(["ml", "milliliter", "milliliters", "millilitre", "millilitres"], "ml", "volume", 1.0)
_add(["l", "liter", "liters", "litre", "litres"], "l", "volume", 1000.0)
_add(["ounce", "ounces", "oz"], "oz", "mass", 28.35)
_add(["pound", "pounds", "lb", "lbs"], "lb", "mass", 453.6)
_add(["g", "gram", "grams", "gr"], "g", "mass", 1.0)
_add(["kg", "kilogram", "kilograms"], "kg", "mass", 1000.0)
# count-like containers / pieces. Dimension "package" means not comparable with mass/volume.
_add(["can", "cans", "jar", "jars", "package", "packages", "pkg", "packet", "packets",
      "box", "boxes", "bag", "bags", "bottle", "bottles", "container", "containers",
      "envelope", "envelopes", "carton", "cartons", "tub", "tubs"], "package", "package", 1.0)
_add(["clove", "cloves"], "clove", "piece", 1.0)
_add(["slice", "slices"], "slice", "piece", 1.0)
_add(["stalk", "stalks", "rib", "ribs"], "stalk", "piece", 1.0)
_add(["sprig", "sprigs"], "sprig", "piece", 1.0)
_add(["bunch", "bunches"], "bunch", "piece", 1.0)
_add(["head", "heads"], "head", "piece", 1.0)
_add(["stick", "sticks"], "stick", "piece", 1.0)
_add(["pinch", "pinches", "dash", "dashes", "splash"], "pinch", "volume", 0.3)
_add(["handful", "handfuls"], "handful", "piece", 1.0)
_add(["fillet", "fillets", "filet", "filets"], "fillet", "piece", 1.0)
_add(["ear", "ears"], "ear", "piece", 1.0)
_add(["leaf", "leaves"], "leaf", "piece", 1.0)
_add(["piece", "pieces"], "piece", "piece", 1.0)
_add(["drop", "drops"], "drop", "volume", 0.05)
_add(["sheet", "sheets"], "sheet", "piece", 1.0)
_add(["strip", "strips"], "strip", "piece", 1.0)
UNITS = _U

# Units that, when they appear without anything after them, ARE the ingredient (e.g. "4 cloves").
_UNIT_CAN_BE_NAME = {"clove", "leaf", "stick"}

DESCRIPTORS = set("""
fresh freshly chopped finely coarsely roughly minced diced sliced thinly thickly grated
shredded crushed peeled seeded deseeded cored halved quartered cubed cut trimmed rinsed
drained washed dried large small medium extra-large jumbo whole beaten lightly softened
melted room temperature cold warm boneless skinless organic packed firmly loosely heaping
level about approximately plus more additional optional divided frozen thawed canned
cooked uncooked raw ripe very unsalted salted good-quality quality high-quality
store-bought storebought homemade prepared toasted lean plain pure natural sifted mashed pitted
stemmed deveined shelled hulled zested juiced squeezed freshly-squeezed fresh-squeezed torn
crumbled cooled chilled inch inches thick thin pieces piece chunks chunk cubes cube strips
wedges rounds dice into a an the some few such as if desired needed taste garnish
serving well generous scant splash finely-chopped thinly-sliced snipped bite-size
bite-sized size sized julienned slivered sliver chilled reserved rinsed patted dry-packed
best quality half halves fat-free low-fat lowfat reduced-fat light nonfat non-fat regular
""".split())
# NB: "ground", "hot", "dry" are NOT generic descriptors (ground beef, hot sauce, dry wine)
# but "ground" is removed in front of spices (see _SPICES).

_SPICES = set("""
black pepper white pepper cinnamon nutmeg cumin coriander ginger cloves clove allspice cardamom
turmeric paprika cayenne mustard fennel sage thyme mace anise pepper chipotle chile chili
red pepper
""".split())

_STOP_TAIL = re.compile(
    r"\b(for (?:garnish|serving|dusting|brushing|frying|greasing|the pan|coating|drizzling|sprinkling|topping)\b.*"
    r"|to taste.*|as needed.*|if desired.*|or to taste.*|or more.*|or as needed.*"
    r"|at room temperature.*|such as .*|preferably .*|available .*|see note.*|optional.*)$"
)

# canonical synonym table (applied to the whole normalized name)
SYNONYMS = {
    "green onion": "scallion", "spring onion": "scallion", "scallion": "scallion",
    "garbanzo bean": "chickpea", "garbanzo": "chickpea", "chick pea": "chickpea",
    "confectioner sugar": "powdered sugar", "confectioners sugar": "powdered sugar",
    "confectioners' sugar": "powdered sugar", "icing sugar": "powdered sugar",
    "white sugar": "sugar", "granulated sugar": "sugar", "granulated white sugar": "sugar",
    "caster sugar": "sugar", "superfine sugar": "sugar", "cane sugar": "sugar",
    "light brown sugar": "brown sugar", "dark brown sugar": "brown sugar",
    "all-purpose flour": "flour", "all purpose flour": "flour", "plain flour": "flour",
    "unbleached all-purpose flour": "flour", "white flour": "flour", "ap flour": "flour",
    "kosher salt": "salt", "sea salt": "salt", "table salt": "salt", "coarse salt": "salt",
    "fine salt": "salt", "flaky salt": "salt", "coarse kosher salt": "salt", "fine sea salt": "salt",
    "flaky sea salt": "salt", "coarse sea salt": "salt",
    "ground black pepper": "black pepper", "pepper": "black pepper",
    "cracked black pepper": "black pepper", "black peppercorn": "black peppercorn",
    "extra-virgin olive oil": "olive oil", "extra virgin olive oil": "olive oil",
    "virgin olive oil": "olive oil",
    "heavy whipping cream": "heavy cream", "whipping cream": "heavy cream",
    "double cream": "heavy cream", "heavy cream": "heavy cream",
    "half and half": "half-and-half", "half & half": "half-and-half",
    "cilantro leaf": "cilantro", "fresh cilantro": "cilantro", "coriander leaf": "cilantro",
    "parsley leaf": "parsley", "flat-leaf parsley": "parsley", "italian parsley": "parsley",
    "flat leaf parsley": "parsley", "curly parsley": "parsley",
    "basil leaf": "basil", "mint leaf": "mint", "thyme leaf": "thyme", "sage leaf": "sage",
    "garlic clove": "garlic", "clove garlic": "garlic", "clove of garlic": "garlic",
    "egg white": "egg white", "egg yolk": "egg yolk", "yolk": "egg yolk",
    "crushed red pepper": "red pepper flakes", "red pepper flake": "red pepper flakes",
    "crushed red pepper flake": "red pepper flakes", "hot red pepper flake": "red pepper flakes",
    "hot red-pepper flake": "red pepper flakes", "dried red pepper flake": "red pepper flakes",
    "red-pepper flake": "red pepper flakes",
    "cornflour": "cornstarch", "corn starch": "cornstarch",
    "bicarbonate of soda": "baking soda", "bicarb soda": "baking soda",
    "zucchini squash": "zucchini", "courgette": "zucchini", "aubergine": "eggplant",
    "capsicum": "bell pepper", "red bell pepper": "red bell pepper",
    "canola oil": "canola oil", "rapeseed oil": "canola oil",
    "plain yogurt": "yogurt", "plain greek yogurt": "greek yogurt", "yoghurt": "yogurt",
    "greek-style yogurt": "greek yogurt",
    "ground beef chuck": "ground beef", "lean ground beef": "ground beef",
    "hamburger": "ground beef", "ground chuck": "ground beef",
    "parmigiano-reggiano": "parmesan cheese", "parmigiano reggiano": "parmesan cheese",
    "parmigiano-reggiano cheese": "parmesan cheese", "parmesan": "parmesan cheese",
    "grated parmesan": "parmesan cheese", "parmigiano": "parmesan cheese",
    "cheddar": "cheddar cheese", "sharp cheddar cheese": "cheddar cheese",
    "sharp cheddar": "cheddar cheese", "mozzarella": "mozzarella cheese",
    "tomatoe": "tomato", "roma tomato": "plum tomato",
    "water, or as needed": "water", "ice water": "water", "cold water": "water",
    "warm water": "water", "boiling water": "water", "hot water": "water",
    "filtered water": "water", "tap water": "water", "lukewarm water": "water",
    "vegetable shortening": "shortening", "solid vegetable shortening": "shortening",
    "eggs": "egg", "large egg": "egg",
    "lemon, juiced": "lemon juice", "fresh lemon juice": "lemon juice",
    "lime, juiced": "lime juice", "fresh lime juice": "lime juice",
    "soy sauce": "soy sauce", "soya sauce": "soy sauce", "shoyu": "soy sauce",
    "chickpea flour": "chickpea flour", "gram flour": "chickpea flour",
    "garbanzo bean flour": "chickpea flour",
    "flax seed": "flaxseed", "ground flaxseed": "flaxseed", "flax meal": "flaxseed",
    "flaxseed meal": "flaxseed", "ground flax seed": "flaxseed", "ground flax": "flaxseed",
    "apple sauce": "applesauce", "unsweetened applesauce": "applesauce",
    "chickpea liquid": "aquafaba",
}

_SINGULAR_EXCEPTIONS = {
    "molasses", "couscous", "hummus", "asparagus", "citrus", "swiss", "brussels", "bass",
    "grits", "greens", "oats", "lettuce", "watercress", "cress", "hibiscus", "octopus",
    "schnapps", "red pepper flakes", "chile flakes", "chives", "peas", "lentils", "beans",
    "series", "species", "mayonnaise", "cheese", "glass", "anise", "swiss chard", "haas",
    "jus", "bouillabaisse", "tortellini", "ravioli", "penne", "spaghetti", "molasse",
}
_IRREGULAR = {"leaves": "leaf", "loaves": "loaf", "halves": "half", "knives": "knife",
              "potatoes": "potato", "tomatoes": "tomato", "mangoes": "mango",
              "avocadoes": "avocado", "chives": "chive", "peas": "pea", "lentils": "lentil",
              "beans": "bean", "oats": "oat", "olives": "olive", "dates": "date",
              "cookies": "cookie", "anchovies": "anchovy", "chilies": "chili",
              "chiles": "chile", "chillies": "chili", "cherries": "cherry", "berries": "berry"}


def singularize(word: str) -> str:
    if word in _IRREGULAR:
        return _IRREGULAR[word]
    if word in _SINGULAR_EXCEPTIONS or len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("ches", "shes", "sses", "xes", "zes", "oes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


_NUM = r"(?:\d+\s+\d+/\d+|\d+/\d+|\d+(?:\.\d+)?)"
_QTY_RE = re.compile(rf"^\s*({_NUM})(?:\s*(?:-|–|to|or)\s*({_NUM}))?\s*")
_PAREN_SIZE_RE = re.compile(rf"^\(\s*({_NUM})\s*(?:-\s*)?([a-z.]+)\.?(?:\s+[a-z]+)?\s*\)\s*")
_SIZE_RE = re.compile(r"(?<![\d/])\d+(?:/\d+)?(?:\s*(?:x|by)\s*\d+(?:/\d+)?)?\s*-?\s*(?:inch|inches|in\.|cm)\b(?:-thick|-long|-wide)?")
_EQUIV_RE = re.compile(r"^(?:\s*/\s*[\d./ ]+\s*-?\s*[a-z]+\.?)+\s*")
_QTY_WORDS_RE = re.compile(r"^(?:several|a few|few|a couple of|a couple|couple of|couple|a|an|one|two|three|four|six|twelve|a good|about)\s+")
_SALT_PEPPER_RE = re.compile(r"\bsalt\b.*\b(and|&)\b.*\bpepper\b|\bpepper\b.*\b(and|&)\b.*\bsalt\b")


def _to_float(num: str) -> float:
    num = num.strip()
    parts = num.split()
    return float(sum(Fraction(p) for p in parts))


def clean_line(raw: str) -> str:
    s = raw.replace("ADVERTISEMENT", "")
    for k, v in UNICODE_FRACTIONS.items():
        s = s.replace(k, v)
    s = s.replace("�", "").replace("®", "").replace("™", "").replace("\xa0", " ")
    return re.sub(r"\s+", " ", s).strip()


def is_header(line: str) -> bool:
    return line.endswith(":") and len(line) < 60


@dataclass
class ParsedIngredient:
    line: str                      # cleaned display line
    names: list[str] = field(default_factory=list)
    qty: float | None = None       # numeric quantity (lower bound of a range)
    unit: str | None = None        # canonical unit, None = count
    dim: str = "count"             # count | volume | mass | package | piece | unknown
    base_qty: float | None = None  # qty converted to ml / g / each when possible

    @property
    def name(self) -> str:
        return self.names[0] if self.names else ""


def normalize_name(text: str) -> str:
    """Normalize a free-text ingredient phrase (no quantity) to a canonical name."""
    s = text.lower()
    s = s.replace("’", "'").replace("`", "'")
    s = re.sub(r"\([^)]*\)", " ", s)         # parentheticals
    s = re.sub(r"\[[^\]]*\]", " ", s)
    s = re.sub(r"[*\"]", " ", s)
    s = re.sub(r"\bhalf\s*(?:and|&|-n-)\s*half\b", "half-and-half", s)
    # "juice of 1 lemon" -> "lemon juice"; "1 lemon, juiced" -> "lemon juice"
    s = re.sub(r"^(?:the\s+)?(juice|zest|grated zest|finely grated zest|peel)\s+(?:of\s+)?(?:\d+(?:/\d+)?\s*|one\s+|half\s+(?:a\s+)?|a\s+|an\s+)?(?:large\s+|small\s+|medium\s+)?([a-z]+?)s?\b.*$",
               lambda m: f"{m.group(2)} {m.group(1).split()[-1]}", s.strip())
    s = re.sub(r"^([a-z]+?)s?,\s*(juiced|zested)\b.*$", lambda m: f"{m.group(1)} {'juice' if m.group(2) == 'juiced' else 'zest'}", s)
    s = s.strip(" .;-")
    s = re.sub(r"^(?:of|a|an)\s+", "", s)
    # cut trailing usage phrases
    s = _STOP_TAIL.sub("", s).strip(" ,.;")
    # comma: choose first segment that still has content after descriptor removal
    segments = [seg.strip() for seg in s.split(",") if seg.strip()]
    chosen = ""
    for seg in segments:
        # alternatives "butter or margarine" -> butter (first alternative with content)
        for alt in re.split(r"\s+or\s+|/", seg):
            alt = re.sub(r"^\s*plus\b.*", "", alt).strip()
            toks = re.findall(r"[a-z]+(?:[-'][a-z]+)*", alt)
            # drop leading brand possessives (campbell's)
            toks = [t for i, t in enumerate(toks) if not (t.endswith("'s") and i == 0 and len(toks) > 1)]
            toks = [t for t in toks if t not in DESCRIPTORS]
            # measurement units are never part of a name (keep piece words like "stick")
            if len(toks) > 1:
                toks = [t for t in toks if not (t in UNITS and UNITS[t][1] in ("volume", "mass", "package") and len(t) > 1)]
            while toks and toks[0] in {"of", "and", "with", "in", "for", "to", "plus"}:
                toks = toks[1:]
            while toks and toks[-1] in {"of", "and", "with", "in", "for", "to"}:
                toks = toks[:-1]
            if toks:
                chosen = " ".join(toks)
                break
        if chosen:
            break
    if not chosen:
        return ""
    toks = chosen.split()
    # "ground cinnamon" -> cinnamon ; keep "ground beef"
    if len(toks) >= 2 and toks[0] == "ground" and " ".join(toks[1:]) in _SPICES | {"black pepper", "white pepper", "red pepper"}:
        toks = toks[1:]
    if toks and toks[-1] in ("clove", "cloves") and len(toks) > 1 and toks[-2] == "garlic":
        toks = toks[:-1]
    name = " ".join(toks)
    if name in SYNONYMS:
        return SYNONYMS[name]
    toks[-1] = singularize(toks[-1])
    name = " ".join(toks)
    return SYNONYMS.get(name, name)


def parse_line(raw: str) -> ParsedIngredient | None:
    line = clean_line(raw)
    if not line or is_header(line):
        return None
    s = line.lower()
    s = re.sub(r"^(?:optional[:,]?\s*)", "", s)
    s = _SIZE_RE.sub(" ", s)
    pi = ParsedIngredient(line=line)

    m = _QTY_RE.match(s)
    if m:
        try:
            pi.qty = _to_float(m.group(1))
        except (ValueError, ZeroDivisionError):
            pi.qty = None
        s = s[m.end():]
    # "2 (10.75 ounce) cans soup" -> 21.5 oz
    pm = _PAREN_SIZE_RE.match(s)
    inner = None
    if pm and pm.group(2).rstrip(".") in UNITS:
        try:
            inner = (_to_float(pm.group(1)), pm.group(2).rstrip("."))
        except (ValueError, ZeroDivisionError):
            inner = None
        s = s[pm.end():]
    # unit
    s = _QTY_WORDS_RE.sub("", s)
    m2 = re.match(r"^([a-z]+\.?)\s*", s)
    unit_word = None
    if m2 and m2.group(1).rstrip(".") in UNITS:
        candidate = m2.group(1).rstrip(".")
        rest = s[m2.end():]
        # single-letter units only when a quantity precedes them
        if len(candidate) > 1 or pi.qty is not None:
            if rest.strip() or candidate not in {"clove", "cloves", "leaf", "leaves", "stick", "sticks"}:
                unit_word = candidate
                s = rest
    # metric equivalents "/45 ml", "/1.25 oz/35 g" and ", plus 1/2 cup"
    s = _EQUIV_RE.sub("", s)
    s = re.sub(r"^,?\s*plus\s+[\d/ .]+\s*[a-z]+\.?\s*", "", s)
    s = re.sub(r"^of\s+", "", s)
    if inner is None and pi.unit is None and unit_word and UNITS[unit_word][1] == "package":
        pm2 = _PAREN_SIZE_RE.match(s)
        if pm2 and pm2.group(2).rstrip(".") in UNITS and UNITS[pm2.group(2).rstrip(".")][1] in ("mass", "volume"):
            try:
                inner = (_to_float(pm2.group(1)), pm2.group(2).rstrip("."))
                s = s[pm2.end():]
            except (ValueError, ZeroDivisionError):
                pass

    if inner is not None:
        q, u = inner
        canon, dim, factor = UNITS[u]
        mult = pi.qty if pi.qty is not None else 1.0
        pi.qty, pi.unit, pi.dim = q * mult, canon, dim
        pi.base_qty = pi.qty * factor
    elif unit_word:
        canon, dim, factor = UNITS[unit_word]
        pi.unit, pi.dim = canon, dim
        if pi.qty is not None and dim in ("volume", "mass"):
            pi.base_qty = pi.qty * factor
    else:
        pi.dim = "count" if pi.qty is not None else "unknown"
        pi.base_qty = pi.qty

    if _SALT_PEPPER_RE.search(s):
        pi.names = ["salt", "black pepper"]
        return pi
    name = normalize_name(s)
    if not name and unit_word in {"clove", "cloves"}:
        name = "garlic"
    if name:
        pi.names = [name]
    return pi


# ---------------------------------------------------------------- matching helpers

# names implied by having a base ingredient (for pantry matching)
IMPLIED_BY = {
    "egg": {"egg white", "egg yolk", "hard-boiled egg"},
    "lemon": {"lemon juice", "lemon zest", "lemon peel"},
    "lime": {"lime juice", "lime zest", "lime peel"},
    "orange": {"orange juice", "orange zest", "orange peel"},
    "garlic": {"garlic clove"},
}


def name_matches(ing_name: str, query: str) -> bool:
    """Head-noun match: query 'onion' matches 'red onion', 'onion'; not 'onion powder'."""
    if not ing_name or not query:
        return False
    if ing_name == query:
        return True
    if ing_name.endswith(" " + query):
        return True
    return ing_name in IMPLIED_BY.get(query, set())


def word_pattern(term: str) -> re.Pattern:
    """Whole-word, plural-tolerant regex for conservative exclusion checks."""
    words = term.lower().split()
    parts = [re.escape(w) for w in words[:-1]]
    last = re.escape(words[-1])
    last_variants = rf"{last}(?:s|es)?"
    if words[-1].endswith("y"):
        last_variants = rf"(?:{last}|{re.escape(words[-1][:-1])}ies)"
    pat = r"[\s-]+".join(parts + [last_variants])
    return re.compile(rf"(?<![a-z]){pat}(?![a-z])", re.I)
