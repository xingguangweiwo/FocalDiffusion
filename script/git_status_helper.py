"""Summarise git status and suggest the next action in plain language."""
from __future__ import annotations

import re
import subprocess
from textwrap import dedent


def run_git_status() -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "status", "-sb"],
        capture_output=True,
        text=True,
        check=False,
    )


def parse_branch_line(line: str) -> tuple[str, str | None, int, int]:
    branch = line[3:] if line.startswith("## ") else line
    remote = None
    ahead = behind = 0

    if "..." in branch:
        branch_name, remote_part = branch.split("...", 1)
        branch = branch_name
        remote_match = re.match(r"([^\[]+)(.*)", remote_part)
        if remote_match:
            remote = remote_match.group(1)
            suffix = remote_match.group(2)
            ahead_match = re.search(r"ahead (\d+)", suffix)
            behind_match = re.search(r"behind (\d+)", suffix)
            if ahead_match:
                ahead = int(ahead_match.group(1))
            if behind_match:
                behind = int(behind_match.group(1))
        else:
            remote = remote_part
    else:
        branch = branch.strip()

    return branch.strip(), remote.strip() if remote else None, ahead, behind


def build_message(result: subprocess.CompletedProcess[str]) -> str:
    if result.returncode != 0:
        return dedent(
            f"""
            运行 `git status -sb` 失败（退出码 {result.returncode}）。
            stderr:
            {result.stderr.strip()}
            """
        ).strip()

    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return "当前仓库没有可用的 Git 状态输出。"

    branch, remote, ahead, behind = parse_branch_line(lines[0])
    has_changes = len(lines) > 1

    if has_changes:
        return dedent(
            f"""
            分支 `{branch}` 上存在尚未提交的修改：
            {chr(10).join(lines[1:])}

            下一步：
            1. 查看差异：`git diff`
            2. 需要保留的修改执行 `git add <文件>`
            3. 提交：`git commit -m "你的提交说明"`
            4. 推送：`git push -u {remote or 'origin'} {branch}`
            如果不想保留这些改动，可用 `git restore <文件>` 或 `git checkout -- <文件>` 恢复。
            """
        ).strip()

    if ahead > 0 and behind > 0:
        return dedent(
            f"""
            分支 `{branch}` 相对于 `{remote}` 同时存在未推送的提交 ({ahead} 个) 和远端新提交 ({behind} 个)。
            建议顺序：
            1. `git pull --rebase {remote or 'origin'} {branch}` 合并远端更新；
            2. 若出现冲突，按提示解决后 `git rebase --continue`；
            3. 完成后执行 `git push -u {remote or 'origin'} {branch}` 推送。
            """
        ).strip()

    if ahead > 0:
        return dedent(
            f"""
            分支 `{branch}` 有 {ahead} 个本地提交尚未推送到 `{remote}`。
            执行 `git push -u {remote or 'origin'} {branch}` 即可让 GitHub 显示这些更新。
            """
        ).strip()

    if behind > 0:
        return dedent(
            f"""
            分支 `{branch}` 落后 `{remote}` {behind} 个提交。
            运行 `git pull --rebase {remote or 'origin'} {branch}` 以获取远端最新代码。
            若担心网络重置，可先执行：
              git config --global http.version HTTP/1.1
              git config --global http.postBuffer 524288000
            然后重试拉取。
            """
        ).strip()

    remote_hint = remote or "origin"
    return dedent(
        f"""
        分支 `{branch}` 工作区干净，且已与 `{remote_hint}` 同步。
        这意味着当前没有本地改动需要推送，GitHub 上的同名分支已经包含相同的提交。

        如果你期望看到新的文件，说明这些文件尚未添加到仓库。可以按以下步骤操作：
        1. 将需要的代码复制到仓库目录；
        2. 使用 `git status` 核对新文件；
        3. 执行 `git add`、`git commit`；
        4. 最后 `git push -u {remote_hint} {branch}`，GitHub 就会同步显示。
        """
    ).strip()


def main() -> None:
    result = run_git_status()
    print(build_message(result))


if __name__ == "__main__":
    main()
