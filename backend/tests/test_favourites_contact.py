NOW = "2026-11-24T10:00:00+00:00"


def _furniture_payload(catalog, **over):
    body = {
        "room_type": "Living room",
        "timestamp": NOW,
        "budget_set": 30000,
        "recommendations": [
            {
                "item_id": catalog["item"].id,
                "subcategory_id": catalog["sub"].id,
                "link": "https://example.com/sofa",
                "color": "Beige",
                "estimated_price": 12000,
            }
        ],
    }
    body.update(over)
    return body


def test_favourites_require_auth(client):
    assert client.get("/api/favourites").status_code == 401
    assert client.post("/api/favourites/furniture", json={}).status_code == 401
    assert client.delete("/api/favourites/1").status_code == 401


def test_save_and_list_furniture_and_colors(client, auth_headers, catalog):
    r = client.post("/api/favourites/furniture", json=_furniture_payload(catalog), headers=auth_headers)
    assert r.status_code == 201
    r = client.post(
        "/api/favourites/colors",
        json={
            "room_type": "Living room",
            "timestamp": NOW,
            "colors": [{"color_name": "Sage", "rgb_value": "156,175,136"}],
        },
        headers=auth_headers,
    )
    assert r.status_code == 201

    data = client.get("/api/favourites", headers=auth_headers).json()
    assert len(data) == 1  # same session timestamp + room => one group
    session = data[0]
    assert session["room_type"] == "Living room"
    assert session["saved_items"][0]["item_name"] == "sofa"
    assert session["saved_items"][0]["recommendations"][0]["category"] == "Fabric sofa"
    assert session["saved_items"][0]["recommendations"][0]["estimated_price"] == 12000
    assert session["saved_colors"][0]["rgb_value"] == "156,175,136"


def test_save_rejects_unknown_or_mismatched_items_atomically(client, auth_headers, catalog):
    bad_pair = _furniture_payload(catalog)
    bad_pair["recommendations"].append(
        {"item_id": catalog["item"].id, "subcategory_id": catalog["other_sub"].id, "link": "x"}
    )
    assert client.post("/api/favourites/furniture", json=bad_pair, headers=auth_headers).status_code == 422
    unknown = _furniture_payload(catalog)
    unknown["recommendations"][0]["item_id"] = 9999
    assert client.post("/api/favourites/furniture", json=unknown, headers=auth_headers).status_code == 422
    # Nothing from the rejected batches may have been saved.
    assert client.get("/api/favourites", headers=auth_headers).json() == []


def test_save_validation(client, auth_headers, catalog):
    empty = _furniture_payload(catalog, recommendations=[])
    assert client.post("/api/favourites/furniture", json=empty, headers=auth_headers).status_code == 422
    neg = _furniture_payload(catalog, budget_set=-5)
    assert client.post("/api/favourites/furniture", json=neg, headers=auth_headers).status_code == 422
    bad_ts = _furniture_payload(catalog, timestamp="yesterday-ish")
    assert client.post("/api/favourites/furniture", json=bad_ts, headers=auth_headers).status_code == 422


def test_color_validation(client, auth_headers):
    base = {"room_type": "Bed room", "timestamp": NOW}
    for bad in ("300,0,0", "1,2", "a,b,c", "1,2,3,4"):
        r = client.post(
            "/api/favourites/colors",
            json={**base, "colors": [{"color_name": "x", "rgb_value": bad}]},
            headers=auth_headers,
        )
        assert r.status_code == 422, bad


def test_delete_only_own_favourites(client, auth_headers, catalog):
    client.post("/api/favourites/furniture", json=_furniture_payload(catalog), headers=auth_headers)
    rec_id = client.get("/api/favourites", headers=auth_headers).json()[0]["saved_items"][0][
        "recommendations"
    ][0]["id"]

    other = client.post(
        "/api/auth/register",
        json={"name": "Other", "email": "other@example.com", "password": "password123"},
    ).json()["token"]
    other_headers = {"Authorization": f"Bearer {other}"}

    assert client.delete(f"/api/favourites/{rec_id}", headers=other_headers).status_code == 404
    assert client.get("/api/favourites", headers=other_headers).json() == []
    assert client.delete(f"/api/favourites/{rec_id}", headers=auth_headers).status_code == 200
    assert client.get("/api/favourites", headers=auth_headers).json() == []
    assert client.delete(f"/api/favourites/{rec_id}", headers=auth_headers).status_code == 404


def test_contact_form(client):
    ok = {
        "first_name": "Riya",
        "email": "riya@example.com",
        "phone": "+91 98765-43210",
        "message": "Hello!",
    }
    r = client.post("/api/contact", json=ok)
    assert r.status_code == 201

    for field, bad in (("phone", "abc"), ("phone", "123"), ("email", "nope"), ("message", ""), ("first_name", "")):
        assert client.post("/api/contact", json={**ok, field: bad}).status_code == 422, (field, bad)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
