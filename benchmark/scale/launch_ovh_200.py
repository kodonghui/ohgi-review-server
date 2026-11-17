"""Run the pgvector experiment over SSH without writing credentials on OVH."""
import argparse
import json
import re
import shlex
import subprocess

from benchmark.langsmith_providers import required_setting, setting


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scratch", help="Absolute scratch directory reserved for this run")
    parser.add_argument("stage", choices=["lesson", "api"])
    parser.add_argument("--host", required=True, help="SSH host for this experiment")
    parser.add_argument("--commit", required=True)
    parser.add_argument("--provider", choices=["gpt", "claude", "deepseek"])
    args = parser.parse_args()
    if not re.fullmatch(r"/tmp/ohgi-eval-200p\.[a-zA-Z0-9]{8}", args.scratch):
        parser.error("이 실험의 임시 경로를 지정하세요.")
    if not re.fullmatch(r"[0-9a-f]{40}", args.commit):
        parser.error("실행 코드의 전체 Git 해시를 지정하세요.")
    if not re.fullmatch(r"[\w.-]+", args.host) or args.host.startswith("-"):
        parser.error("SSH 별칭을 지정하세요.")

    repo = args.scratch + "/repo"
    key_environment = {
        "OPENAI_API_KEY": required_setting("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": required_setting("ANTHROPIC_API_KEY"),
        "DEEPSEEK_RELAY_API_KEY": required_setting(
            "DEEPSEEK_RELAY_API_KEY", "DEEPSEEK_RELAY_API_KEY_1", "DEEPSEEK_RELAY_API_KEY_2",
        ),
        "LANGSMITH_API_KEY": required_setting("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY"),
        "OHGI_PGHOST": args.scratch + "/socket",
        "OHGI_CODE_COMMIT": args.commit,
        "XDG_CACHE_HOME": args.scratch + "/cache",
        "TIKTOKEN_CACHE_DIR": args.scratch + "/cache/tiktoken",
        "PYTHONUTF8": "1",
    }
    for name in ("LANGSMITH_WORKSPACE_ID", "LANGSMITH_ENDPOINT", "OHGI_GPT_MODEL",
                 "OHGI_CLAUDE_MODEL", "OHGI_DEEPSEEK_MODEL"):
        if value := setting(name):
            key_environment[name] = value

    bootstrap = (
        "import json,os,sys\n"
        "os.environ.update(json.load(sys.stdin))\n"
        "os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n"
    )
    command = [
        repo + "/.venv/bin/python", "-c", bootstrap, "-m", "benchmark.scale.eval_200", "pgvector", args.stage,
    ]
    if args.provider:
        command.extend(("--provider", args.provider))
    remote = "cd " + shlex.quote(repo) + " && exec " + shlex.join(command)
    result = subprocess.run(
        ["ssh", "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ServerAliveInterval=30",
         args.host, remote],
        input=json.dumps(key_environment).encode("utf-8"), check=False,
    )
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
