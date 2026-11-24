import pytest

from app.recommend.group import decide, score_pick


def _register(client, name, email):
    r = client.post("/api/auth/register", json={"name": name, "email": email, "password": "password123"})
    assert r.status_code == 201
    return {"Authorization": f"Bearer {r.json()['token']}"}


def _board(client, h):
    r = client.post("/api/boards", headers=h, json={
        "name": "Flat 4B living room", "room_type": "Living Room", "wall_rgb": [220, 210, 190],
        "budget_inr": 30000})
    assert r.status_code == 201
    return r.json()


def _pick(client, h, bid, asin, item="sofa", price=None, title=None):
    return client.post(f"/api/boards/{bid}/picks", headers=h, json={
        "item_type": item, "asin": asin, "title": title or f"Product {asin}", "price_inr": price})


def test_group_scoring_prefers_agreement_over_polarised_love():
    split = score_pick(1, [1, 1, -1, -1], 4)  # half love, half hate
    liked = score_pick(2, [1, 0, 0, 0], 4)  # mild but nobody objects
    assert liked.consensus > split.consensus
    assert split.vetoes == 2
    # votes not yet cast count as 'fine' but are visible
    assert score_pick(3, [1], 4).votes_in == 1 and score_pick(3, [1], 4).mean == 0.25
    out = decide({"sofa": [split, liked]})
    assert out["sofa"]["winner"] == 2 and out["sofa"]["ranking"] == [2, 1]


def test_contested_flag():
    a, b = score_pick(1, [1, 1], 2), score_pick(2, [1, 1], 2)
    assert decide({"x": [a, b]})["x"]["contested"] is True  # near-tie
    veto = score_pick(3, [1, -1], 2)
    assert decide({"x": [veto]})["x"]["contested"] is True
    clear = decide({"x": [score_pick(4, [1, 1], 2), score_pick(5, [-1, 0], 2)]})
    assert clear["x"]["contested"] is False


def test_board_lifecycle_and_decision(client):
    owner = _register(client, "Asha", "asha@example.com")
    friend = _register(client, "Ravi", "ravi@example.com")
    outsider = _register(client, "Zed", "zed@example.com")
    b = _board(client, owner)
    bid = b["id"]

    assert client.get(f"/api/boards/{bid}", headers=outsider).status_code == 404
    assert client.post("/api/boards/join", headers=friend, json={"invite_code": "nope-nope"}).status_code == 404
    assert client.post("/api/boards/join", headers=friend, json={"invite_code": b["invite_code"]}).status_code == 200
    # joining twice is harmless
    assert client.post("/api/boards/join", headers=friend, json={"invite_code": b["invite_code"]}).json()["member_count"] == 2

    p1 = _pick(client, owner, bid, "A1", price=18000).json()
    p2 = _pick(client, friend, bid, "A2", price=14000).json()
    lamp = _pick(client, friend, bid, "L1", item="lamp").json()
    assert _pick(client, owner, bid, "A1").status_code == 409
    assert _pick(client, owner, bid, "X", item="spaceship").status_code == 422

    # votes: owner loves A1, friend vetoes A1 and likes A2
    assert client.put(f"/api/boards/{bid}/picks/{p1['id']}/vote", headers=owner, json={"score": 1}).status_code == 200
    client.put(f"/api/boards/{bid}/picks/{p1['id']}/vote", headers=friend, json={"score": -1})
    client.put(f"/api/boards/{bid}/picks/{p2['id']}/vote", headers=friend, json={"score": 1})
    client.put(f"/api/boards/{bid}/picks/{p2['id']}/vote", headers=owner, json={"score": 0})
    assert client.put(f"/api/boards/{bid}/picks/{p2['id']}/vote", headers=owner, json={"score": 5}).status_code == 422
    # changing a vote replaces it
    r = client.put(f"/api/boards/{bid}/picks/{p2['id']}/vote", headers=owner, json={"score": 1})
    assert r.json()["my_vote"] == 1 and r.json()["votes_in"] == 2

    d = client.get(f"/api/boards/{bid}/decision", headers=owner).json()
    sofa = next(i for i in d["items"] if i["item_type"] == "sofa")
    assert sofa["pick"]["asin"] == "A2"  # agreement beats the split vote
    lamp_row = next(i for i in d["items"] if i["item_type"] == "lamp")
    assert lamp_row["price_is_estimate"] is True
    assert d["estimated_total_inr"] == 14000 + lamp_row["estimated_price_inr"]
    assert d["within_budget"] is True

    detail = client.get(f"/api/boards/{bid}", headers=friend).json()
    assert detail["invite_code"] is None  # only the owner sees the code
    assert {m["name"] for m in detail["members"]} == {"Asha", "Ravi"}
    assert len(client.get("/api/boards", headers=friend).json()) == 1

    # removal: friend can remove own pick, not the owner's; owner can remove anyone's
    assert client.delete(f"/api/boards/{bid}/picks/{p1['id']}", headers=friend).status_code == 403
    assert client.delete(f"/api/boards/{bid}/picks/{lamp['id']}", headers=friend).status_code == 204
    assert client.delete(f"/api/boards/{bid}/picks/{p2['id']}", headers=owner).status_code == 204


def test_board_validation(client):
    h = _register(client, "Asha", "asha@example.com")
    bad_room = client.post("/api/boards", headers=h, json={
        "name": "x", "room_type": "garage", "wall_rgb": [1, 2, 3], "budget_inr": 10})
    assert bad_room.status_code == 422
    bad_rgb = client.post("/api/boards", headers=h, json={
        "name": "x", "room_type": "bedroom", "wall_rgb": [1, 2, 300], "budget_inr": 10})
    assert bad_rgb.status_code == 422
    assert client.post("/api/boards", json={}).status_code == 401
