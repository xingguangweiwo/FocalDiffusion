"""Utility script to help recover from Git pull failures on flaky connections."""
from __future__ import annotations

import argparse
import subprocess
import sys
from textwrap import dedent


def run_git_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a git command and return the completed process."""
    return subprocess.run(["git", *args], capture_output=True, text=True)


def ensure_buffer_config() -> None:
    """Enlarge the global HTTP post buffer to better tolerate unstable networks."""
    subprocess.run(
        ["git", "config", "--global", "http.postBuffer", "524288000"],
        check=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Attempt a 'git pull --rebase' and print step-by-step recovery hints when "
            "the connection is reset."
        )
    )
    parser.add_argument("remote", nargs="?", default="origin", help="Remote name")
    parser.add_argument("branch", nargs="?", default="main", help="Branch name")
    parser.add_argument(
        "--skip-buffer",
        action="store_true",
        help="Do not automatically enlarge the HTTP post buffer before pulling.",
    )
    args = parser.parse_args()

    if not args.skip_buffer:
        ensure_buffer_config()

    pull_cmd = ["pull", "--rebase", args.remote, args.branch]
    result = run_git_command(pull_cmd)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)

    if result.returncode == 0:
        return

    if "Connection was reset" in result.stderr or result.returncode != 0:
        hint = dedent(
            f"""
            Git 仍然报告连接被重置。可以依次尝试以下方案：

            1. 切换到 SSH：
               a) 生成密钥：`ssh-keygen -t ed25519 -C \"you@example.com\"`
               b) 启动 ssh-agent 并添加密钥：
                  - Git Bash: `eval \"$(ssh-agent -s)\"`，然后 `ssh-add ~/.ssh/id_ed25519`
               c) 将 `~/.ssh/id_ed25519.pub` 内容复制到 GitHub -> Settings -> SSH and GPG keys。
               d) 更新远端：`git remote set-url {args.remote} git@github.com:xingguangweiwo/FocalDiffusion.git`
               e) 验证：`ssh -T git@github.com`，成功后重新运行本脚本。

            2. 使用个人访问令牌 (PAT)：
               a) 在 https://github.com/settings/tokens 创建带 repo 权限的令牌。
               b) 更新远端：
                  `git remote set-url {args.remote} https://<TOKEN>@github.com/xingguangweiwo/FocalDiffusion.git`
               c) 再次运行 `git pull --rebase {args.remote} {args.branch}`。

            3. 若仍失败，可设置 HTTP 版本：
               `git config --global http.version HTTP/1.1`
               然后重试拉取。

            4. 确认网络代理/防火墙未拦截 Git：
               - 如果使用代理，请设置 `git config --global http.proxy http://user:pass@proxy:port`
                 （或 `https.proxy`）。
               - 关闭 VPN/杀毒软件后再试。

            完成上述任一方案后，重新运行 `git pull --rebase {args.remote} {args.branch}` 即可同步最新代码。
            """
        ).strip()
        print(hint)


if __name__ == "__main__":
    main()
