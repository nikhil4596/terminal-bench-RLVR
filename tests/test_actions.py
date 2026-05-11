import pytest

from tb_rlvr.actions import ActionParseError, parse_action


def test_action_parser_accepts_bash() -> None:
    action = parse_action("<bash>pytest -q</bash>")
    assert action.kind == "bash"
    assert action.payload == "pytest -q"


def test_action_parser_accepts_patch() -> None:
    action = parse_action("<patch path='/app/a.py'>print(1)</patch>")
    assert action.kind == "patch"
    assert action.path == "/app/a.py"
    assert action.payload == "print(1)"


def test_action_parser_rejects_two_actions() -> None:
    with pytest.raises(ActionParseError):
        parse_action("<bash>ls</bash><finish>done</finish>")

