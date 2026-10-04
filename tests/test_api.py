import pytest
from fastapi.testclient import TestClient

from api.main import app, runner


@pytest.fixture
def client():
    runner.reset()
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/api/health").json() == {"ok": True}


def test_websocket_sends_layout_then_state(client):
    with client.websocket_connect("/ws") as ws:
        first = ws.receive_json()
        assert first["type"] == "layout"
        assert len(first["nodes"]) > 0
        second = ws.receive_json()
        assert second["type"] == "state"
        assert "cars" in second and "stats" in second


def test_pause_and_resume(client):
    client.post("/api/pause")
    assert runner.paused
    client.post("/api/resume")
    assert not runner.paused


def test_speed_is_validated(client):
    assert client.post("/api/speed", json={"speed": 4}).status_code == 200
    assert runner.speed == 4
    assert client.post("/api/speed", json={"speed": 0}).status_code == 422
    assert client.post("/api/speed", json={"speed": 100}).status_code == 422


def test_spawn_adds_cars_to_queues(client):
    r = client.post("/api/spawn", json={"count": 10})
    assert r.json()["added"] == 10
    assert sum(len(q) for q in runner.sim.spawn_queues.values()) >= 10


def test_rush_hour_toggle(client):
    client.post("/api/rush-hour", json={"on": True})
    assert runner.sim.rush_hour
    client.post("/api/rush-hour", json={"on": False})
    assert not runner.sim.rush_hour


def test_ai_mode_needs_a_model(client):
    runner.ai_policy = None
    assert client.post("/api/lights", json={"mode": "ai"}).status_code == 409
    assert client.post("/api/lights", json={"mode": "banana"}).status_code == 400
    assert client.post("/api/lights", json={"mode": "fixed"}).status_code == 200


def test_paused_sim_does_not_advance():
    runner.reset()
    runner.paused = True
    before = runner.sim.time
    runner.advance(1.0)
    assert runner.sim.time == before


def test_speed_multiplies_sim_time():
    runner.reset()
    runner.speed = 4
    runner.advance(0.5)
    assert runner.sim.time == pytest.approx(2.0)


def test_ai_mode_works_with_model_loaded(client):
    from api.main import MODEL_PATH
    import os
    if not os.path.exists(MODEL_PATH):
        pytest.skip("no trained model")
    from engine.control import AIController
    from engine.policy import NumpyPolicy
    runner.ai_policy = AIController(NumpyPolicy(MODEL_PATH))
    assert client.post("/api/lights", json={"mode": "ai"}).status_code == 200
    runner.advance(1.0)
    assert runner.sim.light_mode == "ai"
    client.post("/api/lights", json={"mode": "fixed"})


def test_history_samples_every_five_sim_seconds():
    from api.main import HISTORY_EVERY
    runner.reset()
    runner.speed = 10
    for _ in range(100):
        runner.advance(0.1)   # 100 frames of 100 ms at 10x = 100 sim seconds
    assert len(runner.history) == pytest.approx(runner.sim.time / HISTORY_EVERY, abs=1)
    t, wait, mode = runner.history[-1]
    assert wait >= 0 and mode == "fixed"


def test_history_is_capped_and_samples_since_works():
    from api.main import HISTORY_LEN
    runner.reset()
    for _ in range(HISTORY_LEN + 10):
        runner._take_sample()
    assert len(runner.history) == HISTORY_LEN
    assert runner.samples_since(runner.samples_taken) == []
    assert len(runner.samples_since(runner.samples_taken - 3)) == 3


def test_history_records_light_mode(client):
    import os
    from api.main import MODEL_PATH
    if not os.path.exists(MODEL_PATH):
        pytest.skip("no trained model")
    from engine.control import AIController
    from engine.policy import NumpyPolicy
    runner.reset()
    runner.ai_policy = AIController(NumpyPolicy(MODEL_PATH))
    client.post("/api/lights", json={"mode": "ai"})
    runner.speed = 10
    for _ in range(20):
        runner.advance(0.1)   # 20 frames of 100 ms = 20 sim seconds at 10x
    assert runner.history[-1][2] == "ai"
    client.post("/api/lights", json={"mode": "fixed"})


def test_layout_message_includes_history(client):
    with client.websocket_connect("/ws") as ws:
        first = ws.receive_json()
        assert "history" in first and "ai_available" in first


def test_reset_clears_history(client):
    runner._take_sample()
    client.post("/api/reset")
    assert runner.history == []
