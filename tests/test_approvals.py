from pathlib import Path

from compositor.approvals.buffer import SideEffectBuffer
from compositor.approvals.classify import is_side_effect_tool, split_tool_calls
from compositor.approvals.store import ApprovalStatus, ApprovalStore


def test_classifies_write_and_web_search_as_side_effect():
    assert is_side_effect_tool("write")
    assert is_side_effect_tool("bash")
    assert is_side_effect_tool("web_search")
    assert is_side_effect_tool("WebFetch")
    assert not is_side_effect_tool("read")
    assert not is_side_effect_tool("grep")


def test_split_tool_calls():
    calls = [
        {"id": "1", "type": "function", "function": {"name": "read", "arguments": "{}"}},
        {
            "id": "2",
            "type": "function",
            "function": {"name": "write", "arguments": '{"path":"a"}'},
        },
        {
            "id": "3",
            "type": "function",
            "function": {"name": "web_search", "arguments": '{"q":"x"}'},
        },
    ]
    safe, risky = split_tool_calls(calls)
    assert [c["function"]["name"] for c in safe] == ["read"]
    assert [c["function"]["name"] for c in risky] == ["write", "web_search"]


def test_buffer_removes_side_effects_and_enqueues(tmp_path: Path):
    store = ApprovalStore(tmp_path)
    buf = SideEffectBuffer(store, enabled=True)
    response = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "arguments": '{"command":"rm -rf /"}',
                            },
                        }
                    ],
                },
            }
        ]
    }
    out = buf.process(response, trace_id="tr-1")
    assert len(out.buffered) == 1
    assert out.buffered[0].tool_name == "bash"
    assert out.buffered[0].status == ApprovalStatus.PENDING.value
    msg = out.response["choices"][0]["message"]
    assert msg.get("tool_calls") in (None, [])
    assert out.response["choices"][0]["finish_reason"] == "stop"
    assert "appr-" in (msg.get("content") or "")
    assert store.get(out.buffered[0].id) is not None


def test_buffer_keeps_safe_tools(tmp_path: Path):
    store = ApprovalStore(tmp_path)
    buf = SideEffectBuffer(store)
    response = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "function": {"name": "read", "arguments": '{"path":"a"}'},
                        },
                        {
                            "id": "c2",
                            "function": {"name": "write", "arguments": '{"path":"b"}'},
                        },
                    ],
                },
            }
        ]
    }
    out = buf.process(response)
    assert len(out.buffered) == 1
    assert len(out.passed_through) == 1
    assert out.response["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "read"


def test_approve_and_reject(tmp_path: Path):
    store = ApprovalStore(tmp_path)
    action = store.enqueue(
        tool_name="write",
        arguments="{}",
        raw_tool_call={"function": {"name": "write"}},
    )
    store.decide(action.id, approve=True, note="ok")
    assert store.get(action.id).status == ApprovalStatus.APPROVED.value
    action2 = store.enqueue(
        tool_name="bash",
        arguments="{}",
        raw_tool_call={"function": {"name": "bash"}},
    )
    store.decide(action2.id, approve=False, note="no")
    assert store.get(action2.id).status == ApprovalStatus.REJECTED.value
