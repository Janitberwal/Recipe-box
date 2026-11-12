"""Curated culinary-role table (GENERAL KNOWLEDGE, not dataset evidence).

Used only to (1) name the role an ingredient plays and (2) propose candidate substitutes with
typical ratios and *predicted* effects. Whether a candidate is supported is decided by the
dataset evidence computed in substitute.py; the UI labels this table as general knowledge.
"""

SUBSTITUTES = {
    "egg": {
        "match": ["egg", "eggs", "egg white", "egg yolk"],
        "roles": {"baked": ["binding", "leavening/lift", "moisture", "structure"],
                  "savory": ["binding", "richness", "structure (sets when cooked)"]},
        "candidates": [
            {"name": "applesauce", "ratio": "1/4 cup unsweetened applesauce per egg", "effect": "moister, denser, slightly sweeter crumb; little lift", "roles": ["moisture", "binding"], "contexts": ["baked"]},
            {"name": "banana", "ratio": "1/4 cup mashed ripe banana per egg", "effect": "denser, moist crumb with banana flavor", "roles": ["moisture", "binding"], "contexts": ["baked"]},
            {"name": "flaxseed", "ratio": "1 tbsp ground flaxseed + 3 tbsp water per egg (rest 5 min)", "effect": "good binding, slightly nutty, a little gummier/denser", "roles": ["binding"], "contexts": ["baked", "savory"]},
            {"name": "yogurt", "ratio": "1/4 cup plain yogurt per egg", "effect": "moist, tender crumb; slight tang", "roles": ["moisture", "binding"], "contexts": ["baked"]},
            {"name": "buttermilk", "ratio": "1/4 cup buttermilk per egg", "effect": "tender crumb, tang; reacts with baking soda for some lift", "roles": ["moisture", "leavening/lift"], "contexts": ["baked"]},
            {"name": "sour cream", "ratio": "1/4 cup sour cream per egg", "effect": "rich, moist, slightly denser", "roles": ["moisture", "binding"], "contexts": ["baked"]},
            {"name": "silken tofu", "ratio": "1/4 cup puréed silken tofu per egg", "effect": "dense, moist; neutral flavor", "roles": ["binding", "moisture"], "contexts": ["baked", "savory"]},
            {"name": "aquafaba", "ratio": "3 tbsp chickpea cooking liquid per egg", "effect": "some lift and binding; lighter than fruit purées", "roles": ["binding", "leavening/lift"], "contexts": ["baked"]},
            {"name": "vinegar", "ratio": "1 tbsp vinegar + 1 tsp baking soda per egg", "effect": "adds lift but no binding; can taste slightly tangy", "roles": ["leavening/lift"], "contexts": ["baked"]},
            {"name": "chia seed", "ratio": "1 tbsp chia seeds + 3 tbsp water per egg", "effect": "binding; visible specks, slightly gel-like", "roles": ["binding"], "contexts": ["baked"]},
            {"name": "chickpea flour", "ratio": "3 tbsp chickpea flour + 3 tbsp water per egg", "effect": "sets when cooked like egg; beany flavor, denser", "roles": ["binding", "structure (sets when cooked)"], "contexts": ["savory"]},
        ],
    },
    "cream": {
        "match": ["heavy cream", "cream", "whipping cream", "light cream", "half-and-half", "double cream"],
        "roles": {"any": ["richness/fat", "body/thickening", "moisture", "flavor (dairy)"]},
        "candidates": [
            {"name": "coconut milk", "ratio": "same volume full-fat coconut milk", "effect": "creamy body, noticeable coconut flavor, slightly thinner", "roles": ["richness/fat", "body/thickening", "moisture"]},
            {"name": "coconut cream", "ratio": "same volume coconut cream (thin with water if needed)", "effect": "very rich; coconut flavor", "roles": ["richness/fat", "body/thickening"]},
            {"name": "cashew", "ratio": "1/2 cup soaked cashews blended with 1/2 cup water per cup cream", "effect": "creamy, neutral-nutty; needs blending (tree nut)", "roles": ["richness/fat", "body/thickening"]},
            {"name": "olive oil", "ratio": "2-3 tbsp olive oil + 1/2 cup starchy pasta water per cup cream, emulsified", "effect": "lighter, glossy sauce instead of creamy; olive-oil flavor", "roles": ["richness/fat"]},
            {"name": "soy milk", "ratio": "1 cup unsweetened soy milk + 1 tbsp cornstarch per cup cream", "effect": "thinner and less rich; needs starch to thicken", "roles": ["moisture", "body/thickening"]},
            {"name": "almond milk", "ratio": "1 cup unsweetened almond milk + 1 tbsp cornstarch per cup cream", "effect": "thin, mild; needs starch to thicken (tree nut)", "roles": ["moisture"]},
            {"name": "vegetable broth", "ratio": "same volume broth + 1 tbsp cornstarch", "effect": "much lighter, savory, not creamy", "roles": ["moisture"]},
            {"name": "silken tofu", "ratio": "same volume puréed silken tofu", "effect": "creamy body, neutral; can be slightly grainy", "roles": ["body/thickening", "richness/fat"]},
            {"name": "milk", "ratio": "same volume whole milk + 1 tbsp butter or flour", "effect": "thinner, may curdle with acid", "roles": ["moisture"]},
            {"name": "evaporated milk", "ratio": "same volume", "effect": "slightly caramelized flavor, less fat", "roles": ["moisture", "body/thickening"]},
            {"name": "cream cheese", "ratio": "about 1/2 the volume, whisked in off heat", "effect": "thick, tangy", "roles": ["body/thickening", "richness/fat"]},
        ],
    },
    "butter": {
        "match": ["butter", "unsalted butter", "salted butter"],
        "roles": {"baked": ["fat/tenderness", "flavor", "leavening (creaming)"], "savory": ["fat for cooking", "flavor", "richness"]},
        "candidates": [
            {"name": "olive oil", "ratio": "3/4 the amount of butter", "effect": "fruity flavor; no creaming, so denser bakes", "roles": ["fat/tenderness", "fat for cooking"]},
            {"name": "vegetable oil", "ratio": "3/4 the amount of butter", "effect": "neutral, moist; less buttery flavor", "roles": ["fat/tenderness", "fat for cooking"]},
            {"name": "coconut oil", "ratio": "same amount (solid for creaming)", "effect": "can cream like butter; mild coconut flavor", "roles": ["fat/tenderness", "leavening (creaming)"]},
            {"name": "shortening", "ratio": "same amount", "effect": "tender/flaky, no butter flavor", "roles": ["fat/tenderness", "leavening (creaming)"]},
            {"name": "margarine", "ratio": "same amount (check label: some contain dairy)", "effect": "similar texture, different flavor", "roles": ["fat/tenderness", "flavor"]},
            {"name": "applesauce", "ratio": "half the butter amount (baking only)", "effect": "lower fat, cakier and denser", "roles": ["fat/tenderness"]},
        ],
    },
    "milk": {
        "match": ["milk", "whole milk", "skim milk", "2% milk", "low-fat milk"],
        "roles": {"any": ["moisture", "protein/browning", "flavor"]},
        "candidates": [
            {"name": "soy milk", "ratio": "same volume, unsweetened", "effect": "closest protein content; mild bean flavor", "roles": ["moisture", "protein/browning"]},
            {"name": "almond milk", "ratio": "same volume, unsweetened", "effect": "thinner, nutty (tree nut)", "roles": ["moisture"]},
            {"name": "oat milk", "ratio": "same volume", "effect": "creamy, slightly sweet", "roles": ["moisture"]},
            {"name": "coconut milk", "ratio": "same volume (light/beverage style)", "effect": "richer, coconut flavor", "roles": ["moisture", "flavor"]},
            {"name": "rice milk", "ratio": "same volume", "effect": "thin and sweet", "roles": ["moisture"]},
            {"name": "water", "ratio": "same volume (+1 tbsp oil per cup)", "effect": "less rich, less browning", "roles": ["moisture"]},
        ],
    },
    "sour cream": {
        "match": ["sour cream"],
        "roles": {"any": ["moisture", "tang", "richness/fat"]},
        "candidates": [
            {"name": "yogurt", "ratio": "same volume plain/Greek yogurt", "effect": "tangier, slightly less rich", "roles": ["moisture", "tang"]},
            {"name": "coconut cream", "ratio": "same volume + squeeze of lemon", "effect": "rich, coconut flavor", "roles": ["richness/fat"]},
            {"name": "silken tofu", "ratio": "same volume blended with lemon juice", "effect": "neutral, less rich", "roles": ["moisture"]},
            {"name": "cashew", "ratio": "blended soaked cashews + lemon", "effect": "rich, nutty (tree nut)", "roles": ["richness/fat", "tang"]},
            {"name": "buttermilk", "ratio": "3/4 the volume", "effect": "thinner, tangy", "roles": ["moisture", "tang"]},
        ],
    },
    "cheese": {
        "match": ["cheese", "parmesan cheese", "cheddar cheese", "mozzarella cheese"],
        "roles": {"any": ["savory/umami flavor", "saltiness", "melting/binding"]},
        "candidates": [
            {"name": "nutritional yeast", "ratio": "about 2 tbsp per 1/4 cup grated cheese", "effect": "savory/cheesy flavor, no melting", "roles": ["savory/umami flavor"]},
            {"name": "breadcrumb", "ratio": "toasted, same volume as topping", "effect": "crunch instead of cheesy crust (contains gluten)", "roles": ["melting/binding"]},
            {"name": "miso", "ratio": "1-2 tsp per 1/2 cup cheese, in sauces", "effect": "salty umami, no stretch (soy)", "roles": ["savory/umami flavor", "saltiness"]},
            {"name": "cashew", "ratio": "blended soaked cashews in sauces", "effect": "creamy, mild (tree nut)", "roles": ["melting/binding"]},
        ],
    },
    "flour": {
        "match": ["flour", "all-purpose flour", "wheat flour", "bread flour"],
        "roles": {"baked": ["structure (gluten)", "thickening"], "savory": ["thickening", "coating"]},
        "candidates": [
            {"name": "rice flour", "ratio": "same weight; add binder for baking", "effect": "crumblier, grittier", "roles": ["thickening", "coating"]},
            {"name": "almond flour", "ratio": "1:1 by volume, extra egg/binder", "effect": "moist, dense, nutty (tree nut)", "roles": ["structure (gluten)"]},
            {"name": "cornstarch", "ratio": "half the amount (for thickening only)", "effect": "glossy thickening; not for structure", "roles": ["thickening", "coating"]},
            {"name": "oat flour", "ratio": "1:1, slightly denser (check GF oats)", "effect": "tender, oaty", "roles": ["structure (gluten)"]},
            {"name": "coconut flour", "ratio": "1/4 the amount + extra liquid/eggs", "effect": "very absorbent, dense", "roles": ["structure (gluten)"]},
        ],
    },
    "sugar": {
        "match": ["sugar", "white sugar", "granulated sugar"],
        "roles": {"any": ["sweetness", "moisture/tenderness", "browning"]},
        "candidates": [
            {"name": "honey", "ratio": "3/4 cup per cup sugar, reduce liquid by 1/4 cup", "effect": "moister, browns faster, honey flavor", "roles": ["sweetness", "moisture/tenderness"]},
            {"name": "maple syrup", "ratio": "3/4 cup per cup sugar, reduce liquid", "effect": "maple flavor, moister", "roles": ["sweetness"]},
            {"name": "brown sugar", "ratio": "same amount", "effect": "molasses flavor, moister", "roles": ["sweetness", "browning"]},
            {"name": "agave", "ratio": "2/3 cup per cup sugar", "effect": "sweeter, moister", "roles": ["sweetness"]},
        ],
    },
    "peanut butter": {
        "match": ["peanut butter"],
        "roles": {"any": ["fat/richness", "binding", "nutty flavor"]},
        "candidates": [
            {"name": "almond butter", "ratio": "same amount", "effect": "similar texture, milder (tree nut)", "roles": ["fat/richness", "binding"]},
            {"name": "sunflower seed butter", "ratio": "same amount", "effect": "earthier; can turn green with baking soda", "roles": ["fat/richness", "binding"]},
            {"name": "tahini", "ratio": "same amount (+ a little sweetener)", "effect": "more bitter/sesame flavor, runnier", "roles": ["fat/richness"]},
        ],
    },
    "soy sauce": {
        "match": ["soy sauce"],
        "roles": {"any": ["saltiness", "umami", "color"]},
        "candidates": [
            {"name": "tamari", "ratio": "same amount", "effect": "near-identical (still soy; usually gluten-free)", "roles": ["saltiness", "umami"]},
            {"name": "coconut aminos", "ratio": "same amount + pinch of salt", "effect": "sweeter, less salty", "roles": ["umami"]},
            {"name": "worcestershire sauce", "ratio": "half the amount", "effect": "tangier (contains fish)", "roles": ["umami"]},
        ],
    },
}


def find_entry(target: str):
    t = target.lower().strip()
    for key, e in SUBSTITUTES.items():
        if t == key or t in e["match"]:
            return key, e
    for key, e in SUBSTITUTES.items():
        if any(t.endswith(m) or m.endswith(t) for m in e["match"]):
            return key, e
    return None, None
