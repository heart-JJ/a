import os
import time
import subprocess
import traceback

WORK_DIR = "/mnt/workspace/a"
ENTRY_FILE = "main.py"
LOG_FILE = "run_log.txt"
PULL_INTERVAL = 20  # 每20秒拉取一次

def run_cmd(cmd):
    try:
        result = subprocess.run(
            cmd, shell=True, cwd=WORK_DIR,
            capture_output=True, text=True, timeout=300
        )
        return result.stdout + result.stderr
    except Exception as e:
        return traceback.format_exc()

def git_pull():
    return run_cmd("git pull origin main")

def git_push():
    run_cmd("git add -A")
    run_cmd('git commit -m "auto: update run result"')
    return run_cmd("git push origin main")

def main():
    print("Worker started, watching for updates...")
    while True:
        pull_log = git_pull()
        if "Already up to date" not in pull_log:
            print("New code detected, running...")
            # 自动运行入口文件
            run_output = run_cmd(f"python {ENTRY_FILE}")
            # 写入日志文件
            log_path = os.path.join(WORK_DIR, LOG_FILE)
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("=== Pull log ===\n")
                f.write(pull_log)
                f.write("\n=== Run output ===\n")
                f.write(run_output)
            # 回传结果到 GitHub
            git_push()
            print("Run finished, result pushed.")
        time.sleep(PULL_INTERVAL)

if __name__ == "__main__":
    main()
