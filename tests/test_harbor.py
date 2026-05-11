from tb_rlvr.harbor import (
    HarborRunConfig,
    build_harbor_run_command,
    command_to_shell,
    run_harbor_smoke,
)


def test_build_harbor_uvx_oracle_command() -> None:
    command = build_harbor_run_command(
        HarborRunConfig(task_id="openssl-selfsigned-cert")
    )
    assert command == (
        "uvx",
        "harbor",
        "run",
        "-d",
        "terminal-bench@2.0",
        "-t",
        "openssl-selfsigned-cert",
        "-a",
        "oracle",
    )


def test_harbor_smoke_dry_run_skips_execution() -> None:
    result = run_harbor_smoke(
        HarborRunConfig(task_id="sqlite-db-truncate"), dry_run=True
    )
    assert result.skipped
    assert result.returncode is None
    assert "sqlite-db-truncate" in command_to_shell(result.command)
