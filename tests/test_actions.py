import pytest

from tb_rlvr.contracts.actions import ActionParseError, parse_action


def test_action_parser_accepts_command_json() -> None:
    action = parse_action(
        '{"rationale": "inspect", "command": "ls -la", "task_complete": false}'
    )
    assert action.kind == "command"
    assert action.command == "ls -la"
    assert action.rationale == "inspect"


def test_action_parser_accepts_single_commands_array() -> None:
    action = parse_action('{"commands": ["python solve.py"], "task_complete": false}')
    assert action.kind == "command"
    assert action.command == "python solve.py"


def test_action_parser_accepts_finish() -> None:
    action = parse_action('{"rationale": "done", "command": "", "task_complete": true}')
    assert action.kind == "finish"
    assert action.task_complete


def test_action_parser_rejects_multiple_commands() -> None:
    with pytest.raises(ActionParseError):
        parse_action('{"commands": ["ls", "pwd"], "task_complete": false}')


def test_action_parser_rejects_xml() -> None:
    with pytest.raises(ActionParseError):
        parse_action("<bash>ls</bash>")
