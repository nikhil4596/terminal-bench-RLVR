from tb_rlvr.contracts.actions import AgentAction
from tb_rlvr.contracts.safety import check_action_safety


def test_command_rejects_tests_directory_access() -> None:
    result = check_action_safety(
        AgentAction(kind="command", command="cat /tests/test.sh")
    )
    assert not result.ok


def test_command_rejects_solution_access() -> None:
    result = check_action_safety(
        AgentAction(kind="command", command="ls /solution && cat /solution/solve.sh")
    )
    assert not result.ok


def test_command_rejects_verifier_tampering() -> None:
    result = check_action_safety(
        AgentAction(kind="command", command="echo 1 > /logs/verifier/reward.json")
    )
    assert not result.ok


def test_command_allows_workspace_file() -> None:
    result = check_action_safety(
        AgentAction(kind="command", command="python /app/solve.py")
    )
    assert result.ok


def test_forbidden_command_rejected() -> None:
    result = check_action_safety(AgentAction(kind="command", command="rm -rf /"))
    assert not result.ok
