"""Command-line helper for synchronising local changes with a GitHub repository."""
from __future__ import annotations

import argparse
from textwrap import dedent


def build_instructions(remote: str, branch: str) -> str:
    remote_hint = remote or "origin"
    branch_hint = branch or "main"
    return dedent(
        f"""
        1. 切换到仓库目录：`cd /path/to/FocalDiffusion`。
        2. 查看分支：`git status -sb`；若需要新分支，可执行
           `git checkout -b {branch_hint}`。
        3. 拉取远端更新：`git pull --rebase {remote_hint} {branch_hint}`。
           • `git pull` 会先下载远端的新提交，再尝试把它们合并进当前分支；
             `--rebase` 让 Git 把你的本地提交“挪到”最新远端提交之后，
             避免产生额外的 merge commit，便于保持线性历史。
           - 若提示 `Recv failure: Connection was reset`：
             * 先确认网络可达：`ping github.com`（检测到 GitHub 的往返延迟）。
             * 可增大 HTTP buffer：`git config --global http.postBuffer 524288000`，
               这是把 Git 传输时允许的最大缓冲区调大，以应对网络不稳定。 
             * 仍失败时改用 SSH：
               `git remote set-url {remote_hint} git@github.com:用户/仓库.git`，
               然后 `ssh-keygen -t ed25519 -C "you@example.com"` 生成密钥，
               将 `~/.ssh/id_ed25519.pub` 内容添加到 GitHub，并通过
               `ssh -T git@github.com` 验证连接。
             * 或使用个人访问令牌：
               `git remote set-url {remote_hint} https://<token>@github.com/用户/仓库.git`。
        4. 应用或复制新的代码文件后，执行 `git add .` 标记变更。
        5. 提交：`git commit -m "sync latest focal diffusion updates"`。
        6. 推送：`git push -u {remote_hint} {branch_hint}`。
           - 如果 HTTPS 推送仍被重置，可继续使用 SSH/令牌方案。
        7. 在其它机器（如 PyCharm）上同步时运行
           `git pull {remote_hint} {branch_hint}`。
        """
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Print Git/GitHub sync instructions")
    parser.add_argument("remote", nargs="?", default="origin", help="Remote name or URL")
    parser.add_argument("branch", nargs="?", default="main", help="Branch to sync")
    args = parser.parse_args()
    print(build_instructions(args.remote, args.branch))


if __name__ == "__main__":
    main()
