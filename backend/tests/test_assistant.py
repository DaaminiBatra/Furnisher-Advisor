from app import assistant
from app.main import app
from app.routers.assistant import get_generate
from app.routers.recommend import get_catalog

CATALOG = (
    {"asin": "s1", "type": "sofa", "title": "Beige sofa", "brand": "B", "colour_words": "Beige",
     "rgb": [222, 205, 178], "rgb_source": "listing"},
)


def scripted(*steps):
    """Fake Gemini: each step is a list of parts to return; records what it was sent."""
    seen = []
    it = iter(steps)

    def gen(contents):
        seen.append([dict(c) for c in contents])
        return {"candidates": [{"content": {"role": "model", "parts": next(it)}}]}

    return gen, seen


def test_tool_loop_grounds_answer_in_tool_results():
    gen, seen = scripted(
        [{"functionCall": {"name": "find_products", "args": {"item_type": "sofa", "wall_rgb": [60, 100, 170]}}}],
        [{"text": "Try the Beige sofa."}],
    )
    out = assistant.chat([{"role": "user", "text": "sofa for a blue wall?"}], gen, CATALOG)
    assert out["reply"] == "Try the Beige sofa."
    assert out["tool_calls"][0]["name"] == "find_products"
    sent = seen[1][-1]["parts"][0]["functionResponse"]["response"]
    assert sent["products"][0]["title"] == "Beige sofa"


def test_bad_tool_args_become_errors_not_crashes():
    assert "error" in assistant.run_tool("plan_room", {"room_type": "garage"}, CATALOG, None)
    assert "error" in assistant.run_tool("find_products", {"item_type": "sofa", "wall_rgb": [1, 2, 999]}, CATALOG, None)
    assert "error" in assistant.run_tool("find_products", {"item_type": "boat", "wall_rgb": [1, 2, 3]}, CATALOG, None)
    assert "error" in assistant.run_tool("board_decision", {}, CATALOG, None)
    assert "error" in assistant.run_tool("rm_rf", {}, CATALOG, None)
    ok = assistant.run_tool("plan_room", {"room_type": "bedroom", "wall_rgb": [200, 200, 200],
                                          "budget_inr": 30000}, CATALOG, None)
    assert ok["estimated_total_inr"] <= 30000 and "not shop prices" in ok["note"]


def test_loop_is_capped():
    call = [{"functionCall": {"name": "find_products", "args": {"item_type": "sofa", "wall_rgb": [1, 2, 3]}}}]
    gen, seen = scripted(*[call] * 10)
    out = assistant.chat([{"role": "user", "text": "x"}], gen, CATALOG, max_rounds=3)
    assert len(seen) == 3 and "too many steps" in out["reply"]


def test_endpoint_requires_auth_config_and_membership(client, auth_headers):
    body = {"messages": [{"role": "user", "text": "hi"}]}
    assert client.post("/api/assistant/chat", json=body).status_code == 401
    app.dependency_overrides[get_generate] = lambda: None
    assert client.post("/api/assistant/chat", headers=auth_headers, json=body).status_code == 503
    gen, _ = scripted([{"text": "Hello!"}])
    app.dependency_overrides[get_generate] = lambda: gen
    app.dependency_overrides[get_catalog] = lambda: CATALOG
    try:
        assert client.post("/api/assistant/chat", headers=auth_headers, json=body).json()["reply"] == "Hello!"
        bad = {**body, "board_id": 999}
        assert client.post("/api/assistant/chat", headers=auth_headers, json=bad).status_code == 404
        wrong_last = {"messages": [{"role": "assistant", "text": "hi"}]}
        assert client.post("/api/assistant/chat", headers=auth_headers, json=wrong_last).status_code == 422
    finally:
        app.dependency_overrides.pop(get_generate, None)
        app.dependency_overrides.pop(get_catalog, None)
