from tb_rlvr.actions import AgentAction
from tb_rlvr.safety import check_action_safety


def test_patch_rejects_tests_directory() -> None:
    result = check_action_safety(
        AgentAction(kind="patch", path="/app/tests/test_task.py", payload="pass")
    )
    assert not result.ok


def test_patch_rejects_traversal_into_tests_directory() -> None:
    result = check_action_safety(
        AgentAction(kind="patch", path="/app/src/../tests/test_task.py", payload="pass")
    )
    assert not result.ok


def test_patch_allows_workspace_file() -> None:
    result = check_action_safety(
        AgentAction(kind="patch", path="/app/src/main.py", payload="print(1)")
    )
    assert result.ok


def test_forbidden_command_rejected() -> None:
    result = check_action_safety(AgentAction(kind="bash", payload="rm -rf /"))
    assert not result.ok
