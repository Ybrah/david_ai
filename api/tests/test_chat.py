import json
import logging
from types import SimpleNamespace as NS

import pytest
from fastapi.testclient import TestClient

from app import config, gateway, harness, main, tools

TVA = {"messages": [{"role": "user", "content": "TVA 18% sur 1 250 000 ?"}]}


class FakeToolCall:
    def __init__(self, id, name, args):
        self.id = id
        self.type = "function"
        self.function = NS(name=name, arguments=args)

    def model_dump(self):
        return {"id": self.id, "type": self.type,
                "function": {"name": self.function.name, "arguments": self.function.arguments}}


def tool_msg(name, args, id="c1"):
    return NS(content=None, tool_calls=[FakeToolCall(id, name, args)])


def text_msg(content):
    return NS(content=content, tool_calls=None)


GOOD_CALL = tool_msg("calculate", '{"expression": "1250000*0.18"}')
ANSWER = text_msg("TVA = 225 000 FCFA")


class FakeLLM:
    """Plays back a script of assistant messages, one per model call."""

    def __init__(self, *script):
        self.script = list(script)
        self.calls = []
        self.chat = NS(completions=NS(create=self.create))

    async def create(self, **kw):
        self.calls.append({**kw, "messages": list(kw["messages"])})  # snapshot: the harness appends
        usage = NS(prompt_tokens=100, completion_tokens=20, cost=0.0001)
        return NS(choices=[NS(message=self.script.pop(0))], usage=usage)


@pytest.fixture
def use(monkeypatch):
    monkeypatch.setattr(config, "API_KEYS", set())
    monkeypatch.setattr(config, "MAX_REPAIRS", 1)
    gateway._recent.clear()

    def install(*script):
        llm = FakeLLM(*script)
        monkeypatch.setattr(harness, "get_client", lambda: llm)
        return llm
    return install


def post(path="/chat", payload=TVA, **kw):
    return TestClient(main.app).post(path, json=payload, **kw)


def test_tool_loop(use):
    fake = use(GOOD_CALL, ANSWER)
    r = post()
    assert r.status_code == 200
    body = r.json()
    assert body["reply"] == "TVA = 225 000 FCFA"
    assert body["tool_calls"][0]["name"] == "calculate"
    assert body["tool_calls"][0]["ok"] is True
    assert body["repairs"] == 0
    assert body["usage"] == {"prompt_tokens": 200, "completion_tokens": 40, "cost_usd": pytest.approx(0.0002)}
    assert fake.calls[1]["messages"][-1] == {
        "role": "tool", "tool_call_id": "c1", "content": '{"result": 225000.0}'}
    assert fake.calls[0]["messages"][0]["role"] == "system"


def test_bad_tool_call_is_repaired(use):
    fake = use(tool_msg("calc", '{"expression": "1250000*0.18"}'), GOOD_CALL, ANSWER)
    body = post().json()
    assert body["reply"] == "TVA = 225 000 FCFA"
    assert body["repairs"] == 1
    assert [t["ok"] for t in body["tool_calls"]] == [False, True]
    feedback = json.loads(fake.calls[1]["messages"][-1]["content"])
    assert "unknown tool 'calc'" in feedback["error"]


def test_repairs_exhausted_is_typed_error(use):
    use(tool_msg("calculate", "{not json"), tool_msg("calculate", '{"expr": "1+1"}'))
    r = post()
    assert r.status_code == 502
    detail = r.json()["detail"]
    assert detail["error"] == "tool_call_failed"
    assert "missing ['expression']" in detail["message"]
    assert detail["usage"]["prompt_tokens"] == 200  # failed runs still report what they cost


@pytest.mark.parametrize("bad", ["", "  ", '<tool_call>{"name": "calculate"}</tool_call>'])
def test_invalid_reply_is_repaired(use, bad):
    fake = use(text_msg(bad), ANSWER)
    body = post().json()
    assert body["reply"] == "TVA = 225 000 FCFA"
    assert body["repairs"] == 1
    assert fake.calls[1]["messages"][-1]["content"].startswith("[David harness]")


def test_max_rounds(use, monkeypatch):
    monkeypatch.setattr(config, "MAX_TOOL_ROUNDS", 2)
    use(GOOD_CALL, GOOD_CALL)
    r = post()
    assert r.status_code == 502
    assert r.json()["detail"]["error"] == "max_rounds"


def test_upstream_error(use, monkeypatch):
    fake = use()

    async def boom(**kw):
        raise ConnectionError("down")
    monkeypatch.setattr(fake.chat.completions, "create", boom)
    r = post()
    assert r.status_code == 502
    assert r.json()["detail"]["error"] == "upstream_error"


def test_openai_compatible_endpoint(use):
    fake = use(GOOD_CALL, ANSWER)
    payload = {"model": "anything", "messages": [
        {"role": "system", "content": "Réponds en français."},
        {"role": "user", "content": "TVA 18% sur 1 250 000 ?"}]}
    r = post("/v1/chat/completions", payload)
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == config.LLM_MODEL
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "TVA = 225 000 FCFA"}
    assert body["usage"]["total_tokens"] == 240
    assert body["david"]["tool_calls"][0]["name"] == "calculate"
    sent = fake.calls[0]["messages"]
    assert [m["role"] for m in sent] == ["system", "user"]  # client system prompt merged into ours
    assert sent[0]["content"].endswith("Réponds en français.")


def test_openai_endpoint_rejects_stream(use):
    use()
    assert post("/v1/chat/completions", {**TVA, "stream": True}).status_code == 400
    assert post("/v1/chat/completions", {"messages": [{"role": "system", "content": "x"}]}).status_code == 422


def test_auth(use, monkeypatch):
    use(ANSWER)
    monkeypatch.setattr(config, "API_KEYS", {"k1"})
    assert post().status_code == 401
    assert post(headers={"Authorization": "Bearer k1"}).status_code == 200


def test_rate_limit(use, monkeypatch):
    use(ANSWER, ANSWER, ANSWER)
    monkeypatch.setattr(config, "RATE_LIMIT_RPM", 2)
    assert post().status_code == 200
    assert post().status_code == 200
    r = post()
    assert r.status_code == 429
    assert int(r.headers["Retry-After"]) > 0


def test_audit_log_has_metadata_not_content(use, monkeypatch, caplog):
    use(GOOD_CALL, ANSWER)
    monkeypatch.setattr(config, "API_KEYS", {"secret-key"})
    with caplog.at_level(logging.INFO, logger="david.audit"):
        post(headers={"Authorization": "Bearer secret-key"})
    line = json.loads(caplog.records[-1].getMessage())
    assert line["status"] == "ok"
    assert line["tools"] == ["calculate"]
    assert line["cost_usd"] == pytest.approx(0.0002)
    assert "secret-key" not in caplog.text and "1 250 000" not in caplog.text


def test_empty_messages(use):
    assert post(payload={"messages": []}).status_code == 422


def test_tool_validation():
    run = tools.run_tool
    assert run("calculate", '{"expression": "2**10"}') == ('{"result": 1024}', True)
    assert run("calculate", json.dumps({"expression": "__import__('os')"}))[1] is False
    assert run("nope", "{}")[1] is False
    assert run("calculate", "[1]")[1] is False
    out, ok = run("calculate", '{"expression": "1/0"}')
    assert not ok and "ZeroDivisionError" in out
