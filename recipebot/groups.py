"""Dietary / allergen restriction groups.

Each group is a list of whole-word terms checked against the *cleaned ingredient line*
(not just the normalized name), so "cream of mushroom soup" is caught by "cream".
Exception phrases are removed from the line before checking (e.g. "peanut butter" is not
dairy, "coconut milk" is not dairy, "eggplant" is not egg).

This is intentionally conservative (over-excludes rather than under-excludes). It can NOT
detect hidden ingredients inside processed products (e.g. milk solids in chocolate chips),
which the bot states whenever a restriction is applied.
"""
from __future__ import annotations

import re
from functools import lru_cache

GROUPS: dict[str, dict] = {
    "dairy": {
        "label": "dairy",
        "terms": ["milk", "butter", "cream", "cheese", "yogurt", "yoghurt", "buttermilk", "ghee",
                  "whey", "casein", "half-and-half", "half and half", "creme fraiche", "crème fraîche",
                  "kefir", "ice cream", "parmesan", "parmigiano", "mozzarella", "cheddar", "ricotta",
                  "feta", "gouda", "brie", "gruyere", "gruyère", "mascarpone", "provolone",
                  "romano", "pecorino", "monterey jack", "colby", "asiago", "camembert",
                  "gorgonzola", "queso", "paneer", "fontina", "emmental", "manchego", "havarti",
                  "neufchatel", "velveeta", "quark", "custard", "alfredo", "lactose", "curd",
                  "curds", "dulce de leche", "evaporated milk", "condensed milk", "milk chocolate",
                  "bechamel", "béchamel"],
        "exceptions": ["coconut milk", "coconut cream", "almond milk", "soy milk", "soymilk",
                       "rice milk", "oat milk", "cashew milk", "hemp milk", "peanut butter",
                       "almond butter", "cashew butter", "apple butter", "cocoa butter",
                       "nut butter", "sunflower butter", "sunflower seed butter", "seed butter",
                       "cream of tartar", "non-dairy", "nondairy", "dairy-free", "dairy free",
                       "vegan butter", "vegan cheese", "butternut", "butterhead", "butter lettuce",
                       "butter beans", "butter bean", "coconut butter", "shea butter",
                       "milk-free", "lactose-free", "soy creamer", "coconut creamer",
                       "cream soda", "bean curd", "lemon curd"],
    },
    "egg": {
        "label": "eggs",
        "terms": ["egg", "eggs", "egg white", "egg whites", "egg yolk", "egg yolks", "yolk", "yolks",
                  "mayonnaise", "mayo", "meringue", "aioli", "eggnog", "egg noodles", "egg substitute"],
        "exceptions": ["eggplant", "eggplants", "egg replacer", "egg-free", "eggless",
                       "vegan mayonnaise", "egg free"],
    },
    "peanut": {
        "label": "peanuts",
        "terms": ["peanut", "peanuts", "peanut butter", "peanut oil", "groundnut", "groundnuts",
                  "satay"],
        "exceptions": [],
    },
    "tree_nut": {
        "label": "tree nuts",
        "terms": ["almond", "almonds", "walnut", "walnuts", "pecan", "pecans", "cashew", "cashews",
                  "pistachio", "pistachios", "hazelnut", "hazelnuts", "macadamia", "brazil nut",
                  "brazil nuts", "pine nut", "pine nuts", "pignoli", "chestnut", "chestnuts",
                  "praline", "marzipan", "nutella", "amaretto", "frangelico", "nut", "nuts",
                  "mixed nuts"],
        "exceptions": ["nutmeg", "coconut", "water chestnut", "water chestnuts", "doughnut",
                       "donut", "butternut", "peanut", "peanuts", "nutritional yeast",
                       "nut-free", "nutrition", "almond extract-free"],
    },
    "gluten": {
        "label": "gluten",
        "terms": ["flour", "wheat", "bread", "breadcrumbs", "bread crumbs", "panko", "pasta",
                  "spaghetti", "penne", "macaroni", "noodle", "noodles", "couscous", "barley",
                  "rye", "semolina", "farro", "bulgur", "spelt", "seitan", "cracker", "crackers",
                  "biscuit", "biscuits", "tortilla", "tortillas", "pita", "croutons", "soy sauce",
                  "beer", "malt", "orzo", "lasagna", "fettuccine", "linguine", "rigatoni",
                  "tortellini", "ravioli", "gnocchi", "dumpling", "dumplings", "pie crust",
                  "puff pastry", "phyllo", "filo", "graham", "cookie", "cookies", "cake mix",
                  "bun", "buns", "roll", "rolls", "bagel", "baguette", "brioche", "pretzel",
                  "cereal", "teriyaki", "hoisin", "worcestershire", "matzo", "wonton", "udon",
                  "ramen", "bisquick"],
        "exceptions": ["gluten-free", "gluten free", "rice flour", "almond flour", "coconut flour",
                       "corn flour", "cornflour", "chickpea flour", "tapioca flour", "potato flour",
                       "buckwheat flour", "rice noodles", "rice noodle", "corn tortilla",
                       "corn tortillas", "tamari", "rice pasta", "egg roll wrappers-free",
                       "cauliflower rice"],
        "note": "Oats and buckwheat are not flagged; cross-contamination cannot be assessed from recipe text.",
    },
    "shellfish": {
        "label": "shellfish",
        "terms": ["shrimp", "prawn", "prawns", "crab", "lobster", "crawfish", "crayfish", "scallop",
                  "scallops", "clam", "clams", "mussel", "mussels", "oyster", "oysters", "langoustine",
                  "shellfish", "squid", "calamari", "octopus"],
        "exceptions": ["oyster sauce-free", "oyster mushroom", "oyster mushrooms", "crab apple",
                       "crabapple"],
    },
    "fish": {
        "label": "fish",
        "terms": ["fish", "salmon", "tuna", "cod", "halibut", "tilapia", "trout", "anchovy",
                  "anchovies", "sardine", "sardines", "mackerel", "snapper", "sea bass", "bass",
                  "haddock", "swordfish", "catfish", "mahi", "flounder", "sole", "herring",
                  "fish sauce", "worcestershire", "bonito", "caviar", "roe"],
        "exceptions": ["fish-free"],
    },
    "soy": {
        "label": "soy",
        "terms": ["soy", "soya", "soy sauce", "tofu", "tempeh", "edamame", "miso", "tamari",
                  "soybean", "soybeans", "soy milk", "soymilk"],
        "exceptions": [],
    },
    "meat": {
        "label": "meat & poultry",
        "terms": ["beef", "pork", "chicken", "turkey", "lamb", "veal", "bacon", "ham", "sausage",
                  "sausages", "prosciutto", "pancetta", "salami", "pepperoni", "chorizo", "duck",
                  "goose", "venison", "bison", "steak", "meat", "meatballs", "hot dog", "hot dogs",
                  "gelatin", "gelatine", "lard", "suet", "broth", "stock", "bouillon", "rabbit",
                  "goat", "mutton", "liver", "drippings", "brisket", "spareribs", "spare ribs", "back ribs", "pastrami",
                  "mortadella", "guanciale", "andouille", "kielbasa", "bratwurst", "giblets",
                  "quail", "pheasant", "oxtail", "short ribs", "tenderloin", "sirloin", "chuck"],
        "exceptions": ["vegetable broth", "vegetable stock", "mushroom broth", "mushroom stock",
                       "veggie broth", "vegetable bouillon", "meatless", "vegetarian", "vegan",
                       "plant-based", "imitation bacon", "coconut meat", "crab meat-free",
                       "almond meat", "nut meat", "chicken of the woods", "eggplant", "ham-free",
                       "celery rib", "celery ribs", "rib of celery", "ribs of celery"],
    },
}
GROUPS["honey"] = {"label": "honey", "terms": ["honey"], "exceptions": ["honeydew"]}
# derived diets = unions of simple groups
GROUPS["vegetarian"] = {
    "label": "vegetarian (no meat, poultry, fish, shellfish)",
    "union": ["meat", "fish", "shellfish"],
}
GROUPS["vegan"] = {
    "label": "vegan (no animal products)",
    "union": ["meat", "fish", "shellfish", "dairy", "egg", "honey"],
}

ALIASES = {
    "dairy": "dairy", "dairy-free": "dairy", "lactose": "dairy", "milk products": "dairy",
    "egg": "egg", "eggs": "egg", "egg-free": "egg",
    "peanut": "peanut", "peanuts": "peanut",
    "nut": "tree_nut", "nuts": "tree_nut", "tree nut": "tree_nut", "tree nuts": "tree_nut",
    "tree_nut": "tree_nut",
    "gluten": "gluten", "gluten-free": "gluten", "wheat": "gluten",
    "shellfish": "shellfish", "seafood": "shellfish",
    "fish": "fish", "soy": "soy", "meat": "meat",
    "vegetarian": "vegetarian", "vegan": "vegan",
}


def resolve_group(text: str) -> str | None:
    t = text.lower().strip()
    if t in GROUPS:
        return t
    return ALIASES.get(t)


def _pattern(terms: list[str]) -> re.Pattern:
    alts = sorted({re.escape(t.lower()).replace(r"\ ", r"[\s-]+") for t in terms}, key=len, reverse=True)
    return re.compile(r"(?<![a-z])(?:" + "|".join(alts) + r")(?![a-z])", re.I)


@lru_cache(maxsize=None)
def _compiled(group: str):
    g = GROUPS[group]
    exc = _pattern(g["exceptions"]) if g.get("exceptions") else None
    return _pattern(g["terms"]), exc


def members(group: str) -> list[str]:
    """Simple groups that make up `group` (a diet is a union of simple groups)."""
    g = GROUPS[group]
    return list(g["union"]) if "union" in g else [group]


def line_violation(line: str, group: str) -> str | None:
    """Return the matched term if `line` contains an ingredient of `group`, else None."""
    for simple in members(group):
        pat, exc = _compiled(simple)
        text = line.lower()
        if exc is not None:
            text = exc.sub(" ", text)
        m = pat.search(text)
        if m:
            return m.group(0)
    return None


BASE_GROUPS = ["dairy", "egg", "peanut", "tree_nut", "gluten", "shellfish", "fish", "soy", "meat", "honey"]
