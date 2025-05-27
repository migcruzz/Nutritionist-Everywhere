import os
import subprocess
from pathlib import Path
from time import sleep

MAX_MB = 100
MAX_FILES = 1000
REPO_ROOT = Path.cwd()
GIT_BRANCH = "master"
PUSH_TIMEOUT = 240

def run(cmd, input=None, timeout=None):
    print(f"\n▶️ Executing: {cmd}")
    result = subprocess.run(cmd, shell=True, input=input, capture_output=True, text=True, timeout=timeout)
    if result.stdout:
        print(f"📤 stdout:\n{result.stdout.strip()}")
    if result.stderr:
        print(f"📥 stderr:\n{result.stderr.strip()}")
    return result

def get_file_size_mb(path):
    size = path.stat().st_size / (1024 * 1024)
    print(f"📏 File: {path} → {size:.2f} MB")
    return size

def get_ignored_files(paths):
    if not paths:
        return set()
    print("🔍 Checking .gitignore exclusions...")
    input_str = "\n".join(str(p) for p in paths)
    result = run("git check-ignore --stdin", input=input_str)
    return set(result.stdout.strip().splitlines()) if result.stdout else set()

def switch_to_branch(branch_name):
    print(f"\n🔀 Switching to branch: {branch_name}")
    subprocess.run(f"git checkout -B {branch_name}", shell=True, check=True)
    print("🔄 Ensuring upstream is set...")
    run(f"git push --set-upstream origin {branch_name}")

def already_tracked():
    print("📋 Listing already tracked files...")
    result = run("git ls-files")
    return set(result.stdout.strip().splitlines())

def add_and_commit(batch, commit_index):
    batch = [p for p in batch if p.exists()]
    if not batch:
        print(f"⚠️ Batch {commit_index} is empty, skipping...")
        return False

    print(f"➕ Adding {len(batch)} files to staging...")
    joined = " ".join(f'"{p}"' for p in batch)
    result = run(f"git add {joined}")
    if result.returncode != 0:
        print(f"❌ git add failed: {result.stderr.strip()}")
        return False

    print(f"✅ Committing batch {commit_index}...")
    result = run(f'git commit --no-verify -m "bulk commit {commit_index}"')
    if result.returncode == 0:
        print(f"🎯 Commit {commit_index} succeeded.")
        return True
    else:
        print(f"❌ Commit {commit_index} failed: {result.stderr.strip()}")
        return False

def push_to_remote(commit_index):
    print(f"🚚 Pushing commit {commit_index} to remote...")
    try:
        result = run(f"git push origin {GIT_BRANCH}", timeout=PUSH_TIMEOUT)
        if result.returncode == 0:
            print(f"✅ Commit {commit_index} pushed successfully.")
            run("git status")
            return
        elif "no upstream branch" in result.stderr or "set-upstream" in result.stderr:
            print("🛠️ Setting upstream and pushing...")
            result = run(f"git push --set-upstream origin {GIT_BRANCH}", timeout=PUSH_TIMEOUT)
            if result.returncode == 0:
                print(f"✅ Upstream set and pushed successfully.")
                run("git status")
                return
        else:
            print(f"❌ Push failed after commit {commit_index}: {result.stderr.strip()}")
            print(f"⚠️ Trying force push for commit {commit_index}...")
            force_result = run(f"git push --force origin {GIT_BRANCH}", timeout=PUSH_TIMEOUT)
            if force_result.returncode == 0:
                print(f"💥 Force push successful for commit {commit_index}.")
                run("git status")
                return
            else:
                print(f"❌ Force push also failed: {force_result.stderr.strip()}")
                raise RuntimeError(f"Push and force push failed for commit {commit_index}")
    except subprocess.TimeoutExpired:
        print(f"⏱️ Push timeout after commit {commit_index}")
        raise


def collect_files(directory):
    print(f"📁 Scanning files under {directory}...")
    files = [p for p in directory.rglob("*") if p.is_file() and ".git" not in p.parts]
    print(f"🔎 Total files found: {len(files)}")
    return files

def check_origin():
    print("🔗 Checking if remote 'origin' exists...")
    result = run("git remote")
    return "origin" in result.stdout

def bulk_commit_by_size():
    if not check_origin():
        print("❌ No remote 'origin' found. Set up your Git remote before proceeding.")
        return

    all_files = collect_files(REPO_ROOT)
    ignored_files = get_ignored_files(all_files)
    tracked_files = already_tracked()

    batch = []
    total_size = 0.0
    commit_index = 1

    print(f"\n📦 {len(all_files)} total files found.")
    print(f"⛔ {len(ignored_files)} files ignored via .gitignore.")
    print(f"✅ {len(tracked_files)} files already tracked.\n")

    for file_path in all_files:
        if str(file_path) in ignored_files or str(file_path) in tracked_files:
            continue

        size = get_file_size_mb(file_path)
        if (total_size + size > MAX_MB) or (len(batch) >= MAX_FILES):
            if batch:
                print(f"\n🧾 Committing batch {commit_index} ({len(batch)} files, {total_size:.2f} MB)...")
                committed = add_and_commit(batch, commit_index)
                if committed:
                    push_to_remote(commit_index)
                    sleep(1)
                commit_index += 1
                batch = []
                total_size = 0.0

        batch.append(file_path)
        total_size += size

    if batch:
        print(f"\n🧾 Final batch {commit_index} ({len(batch)} files, {total_size:.2f} MB)...")
        committed = add_and_commit(batch, commit_index)
        if committed:
            push_to_remote(commit_index)

    print("\n✅ All batch commits and pushes are complete.")

if __name__ == "__main__":
    print("🚀 Starting Git bulk commit process...")
    switch_to_branch(GIT_BRANCH)
    bulk_commit_by_size()
