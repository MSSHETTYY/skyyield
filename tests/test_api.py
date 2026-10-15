from fastapi.testclient import TestClient

from cargokit.api.app import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_forecast_lane():
    body = client.get("/forecast/DXB-JFK", params={"horizon": 3}).json()
    assert len(body["forecast"]) == 3
    assert all(r["lower"] <= r["forecast"] <= r["upper"] for r in body["forecast"])


def test_forecast_unknown_lane():
    assert client.get("/forecast/DXB-XXX").status_code == 404


def test_scenario_fuel_up_costs_money():
    body = client.post("/scenario", json={"fuel_shift": 0.3}).json()
    assert body["total_delta_usd"] < 0


def test_allocate_and_pricing():
    body = client.post("/allocate", json={"freighters": 6}).json()
    assert len(body["plan"]) == 8
    price = client.get("/pricing/pharma").json()
    assert 0.7 <= price["price_multiplier"] <= 1.6
