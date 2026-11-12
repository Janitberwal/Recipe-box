import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from recipebot.groups import line_violation
from recipebot.normalize import parse_line
from recipebot.store import inclusion_match


@pytest.mark.parametrize("line,name", [
    ("1 onion, finely diced ADVERTISEMENT", "onion"),
    ("2 cloves garlic, finely chopped", "garlic"),
    ("1 teaspoon garlic powder", "garlic powder"),
    ("1/4 cup thinly sliced green onions", "scallion"),
    ("1 1/2 cups 1/4-inch diced red onion", "red onion"),
    ("2 tablespoons tomato paste", "tomato paste"),
    ("1 (28 ounce) can diced tomatoes", "tomato"),
    ("4 large eggs, beaten", "egg"),
    ("1 cup heavy whipping cream", "heavy cream"),
    ("2 cups all-purpose flour", "flour"),
    ("3 tablespoons/45 ml tomato paste", "tomato paste"),
    ("Juice of 1 lemon", "lemon juice"),
    ("3/4 cup half and half", "half-and-half"),
    ("4 skinless, boneless chicken breast halves", "chicken breast"),
])
def test_names(line, name):
    assert parse_line(line).names[0] == name


def test_salt_and_pepper_split():
    assert parse_line("Kosher salt and freshly ground pepper").names == ["salt", "black pepper"]


def test_package_size_quantity():
    p = parse_line("2 (10.75 ounce) cans condensed cream of chicken soup")
    assert p.unit == "oz" and abs(p.qty - 21.5) < 1e-9


def test_headers_and_ads_dropped():
    assert parse_line("For the sauce:") is None
    assert parse_line("ADVERTISEMENT") is None


@pytest.mark.parametrize("ing,query,expected", [
    ("red onion", "onion", True), ("onion powder", "onion", False), ("scallion", "onion", False),
    ("chicken breast", "chicken", True), ("brown rice", "rice", True), ("rice vinegar", "rice", False),
    ("egg yolk", "egg", True), ("eggplant", "egg", False),
])
def test_inclusion(ing, query, expected):
    assert inclusion_match(ing, query) is expected


@pytest.mark.parametrize("line,group,hit", [
    ("1 cup coconut milk", "dairy", False), ("2 tablespoons butter", "dairy", True),
    ("1/2 cup creamy peanut butter", "dairy", False), ("1/2 cup creamy peanut butter", "peanut", True),
    ("1 eggplant", "egg", False), ("1/4 cup mayonnaise", "egg", True), ("1 tsp nutmeg", "tree_nut", False),
    ("2 ribs celery", "meat", False), ("1 cup chicken broth", "vegetarian", True),
    ("1 tablespoon cream of tartar", "dairy", False), ("1 tbsp honey", "vegan", True),
])
def test_groups(line, group, hit):
    assert (line_violation(line, group) is not None) is hit
