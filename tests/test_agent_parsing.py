import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recipebot.agent import repair_action, sanitize_action


def test_sanitize_drops_unknown_and_bad_enums():
    a = sanitize_action({"intent": "stats", "stats_op": "count", "include_all": ["garlic"], "bogus": 1,
                         "metric": "calories", "k": "5", "reply": ""})
    assert a == {"intent": "stats", "stats_op": "count", "include_all": ["garlic"], "k": 5}


def test_sanitize_unknown_intent():
    assert sanitize_action({"intent": "search_recipe"})["intent"] == "other"


def test_repair_recovers_dropped_ingredients():
    a = repair_action({"intent": "stats", "stats_op": "count"},
                      "How many recipes contain both garlic and onion? What percentage of the dataset is that?")
    assert a["include_all"] == ["garlic", "onion"] and "_repaired" in a


def test_repair_or():
    a = repair_action({"intent": "stats", "stats_op": "count"}, "How many recipes contain chicken or turkey?")
    assert a["include_any"] == ["chicken", "turkey"]


def test_repair_leaves_good_actions_alone():
    a = {"intent": "stats", "stats_op": "count", "include_all": ["tomato"]}
    assert repair_action(dict(a), "how many contain tomato and basil") == a
