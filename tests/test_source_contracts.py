import ast
from pathlib import Path

SOURCE = Path("src/backend/main.py")


def source_and_tree():
    source = SOURCE.read_text(encoding="utf-8")
    return source, ast.parse(source)


def test_http_method_and_path_pairs_are_unique():
    _, tree = source_and_tree()
    routes = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            func = decorator.func
            if not isinstance(func, ast.Attribute) or func.attr not in {
                "get", "post", "put", "patch", "delete"
            }:
                continue
            if not isinstance(func.value, ast.Name) or func.value.id != "app":
                continue
            if decorator.args and isinstance(decorator.args[0], ast.Constant):
                routes.append((func.attr.upper(), decorator.args[0].value))
    assert len(routes) == len(set(routes)), "Duplicate HTTP method/path route found"


def test_tmdb_key_is_not_hardcoded_and_uses_environment():
    source, tree = source_and_tree()
    assert "adebe489963f054d04023e4f8ec84924" not in source
    assignments = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "TMDB_API_KEY"
                for target in node.targets)
    ]
    assert assignments
    assert any(
        isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == "os"
        and node.value.func.attr == "get"
        for node in assignments
    )


def test_movie_search_route_is_registered_once():
    _, tree = source_and_tree()
    count = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            if (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and isinstance(decorator.func.value, ast.Name)
                and decorator.func.value.id == "app"
                and decorator.func.attr == "get"
                and decorator.args
                and isinstance(decorator.args[0], ast.Constant)
                and decorator.args[0].value == "/movies/search"
            ):
                count += 1
    assert count == 1
