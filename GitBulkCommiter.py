from __future__ import annotations

import subprocess
from pathlib import Path
from time import sleep

# ----- limites ----------------------------------------------------------
SOLO_LIMIT_MB = 80  # ficheiros > 80 MB: commit isolado
GITHUB_MAX_MB = 100  # >100 MB não é aceite pelo GitHub
BATCH_MB = 70  # tamanho máximo por batch
BATCH_FILES = 300  # nº máximo de ficheiros por batch
# ------------------------------------------------------------------------

REPO_ROOT = Path.cwd()
GIT_BRANCH = "master"
PUSH_TMO = 240


def run(cmd: str, *, input: str | None = None, timeout: int | None = None):
    res = subprocess.run(cmd, shell=True, input=input,
                         capture_output=True, text=True, timeout=timeout)
    if res.stdout:
        print(res.stdout.strip())
    if res.stderr:
        print(res.stderr.strip())
    return res


def mb(path: Path) -> float:
    return path.stat().st_size / 1_048_576


def changed_files() -> list[Path]:
    out = run("git status --porcelain -z").stdout
    if not out:
        return []
    entries = out.split("\0")[:-1]
    files: list[Path] = []
    for e in entries:
        status, p = e[:2].strip(), e[3:]
        if status == "D":
            continue
        path = REPO_ROOT / p
        if path.is_dir():
            files.extend([f for f in path.rglob("*") if f.is_file() and ".git" not in f.parts])
        elif path.is_file():
            files.append(path)
    return files


def switch_branch(name: str):
    run(f"git checkout -B {name}")
    run(f"git push -u origin {name}")


def add_commit(paths: list[Path], msg: str) -> bool:
    rel = [str(p.relative_to(REPO_ROOT)) for p in paths]
    run("git add " + " ".join(f'"{p}"' for p in rel))
    return run(f'git commit -m "{msg}" --no-verify').returncode == 0


def push():
    if run(f"git push origin {GIT_BRANCH}", timeout=PUSH_TMO).returncode != 0:
        run(f"git push --force origin {GIT_BRANCH}", timeout=PUSH_TMO)


def bulk_commit():
    if "origin" not in run("git remote").stdout:
        print("remote 'origin' not set")
        return

    files = changed_files()
    if not files:
        print("nothing to commit")
        return

    # 1. commits isolados p/ ficheiros > SOLO_LIMIT_MB
    idx = 1
    for f in files[:]:
        size = mb(f)
        if size <= SOLO_LIMIT_MB:
            continue
        if size > GITHUB_MAX_MB:
            print(f"skip {f}: {size:.2f} MB exceeds GitHub limit ({GITHUB_MAX_MB} MB)")
            files.remove(f)
            continue

        print(f"solo commit {f} ({size:.2f} MB)")
        if add_commit([f], f"large file commit {idx}"):
            push()
            sleep(1)
        idx += 1
        files.remove(f)

    # 2. batches normais
    batch, total = [], 0.0
    for f in files:
        size = mb(f)
        if size > BATCH_MB:
            print(f"skip {f}: {size:.2f} MB > batch limit {BATCH_MB} MB")
            continue

        if total + size > BATCH_MB or len(batch) >= BATCH_FILES:
            if add_commit(batch, f"bulk commit {idx}"):
                push()
                sleep(1)
            idx += 1
            batch, total = [], 0.0

        batch.append(f)
        total += size

    if batch and add_commit(batch, f"bulk commit {idx}"):
        push()


if __name__ == "__main__":
    switch_branch(GIT_BRANCH)
    bulk_commit()
