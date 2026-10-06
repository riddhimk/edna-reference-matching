"""
Backs up the run history (data/app/) to cloud storage with rclone.

Only run data is copied -- never the rest of the project:

    <remote>/runs/<run id>/results.csv     per-read results of each run
    <remote>/uploads/<content hash>.*.gz   one copy of each distinct uploaded file
    <remote>/edna_app.db                   run history + chat history (a snapshot)

The remote is an rclone remote, by default "edna-drive:", set up once to point
at the project's Google Drive folder:

    ~/.local/bin/rclone config create edna-drive drive scope=drive \\
        root_folder_id=1eT3ivjyrwg0JSL4eaokOWSGurkmOdLYt

Design choices:

  * `rclone copy`, never `sync`: the cloud copy only ever gains files. Deleting
    a run here -- or losing data/app/ entirely -- can never delete the backup.
  * The database is snapshotted with SQLite's online-backup API before upload,
    so a run finishing mid-upload cannot produce a torn copy.
  * Backups run in the background after each run, one at a time; a request
    that arrives during a backup is folded into one more pass afterwards.
  * A backup failure is recorded and reported, never raised into a run: the
    analysis has already succeeded locally.

Deleting a run (DELETE /runs/{id}) is the one operation that removes files from
the cloud copy: delete_remote() purges that run's folder and, when no other run
shares it, its stored upload. If the cloud cannot be reached, the deletion is
queued in pending_cloud_deletes.json and retried at the start of every backup,
so a run deleted here never lingers in the cloud.

Usage:
    python -m src.cloud_backup status
    python -m src.cloud_backup push
    python -m src.cloud_backup pull      # restore into data/app (stop the API first)

Configuration (environment):
    EDNA_CLOUD_BACKUP   "off" disables automatic backups after runs (the tests set this)
    EDNA_RCLONE_REMOTE  rclone remote and optional path, default "edna-drive:"
    EDNA_RCLONE_BIN     path to rclone, default: on PATH, else ~/.local/bin/rclone
    EDNA_APP_DIR        the run-history folder, default data/app
"""
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from typing import List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_NAME = "edna_app.db"
STATUS_FILE = "backup_status.json"
PENDING_FILE = "pending_cloud_deletes.json"
# Backups and deletions never touch the remote at the same time: a backup in
# flight could otherwise re-upload a run that is being deleted.
_remote_lock = threading.Lock()
SETUP_HINT = ("~/.local/bin/rclone config create edna-drive drive scope=drive "
              "root_folder_id=1eT3ivjyrwg0JSL4eaokOWSGurkmOdLYt")


def remote() -> str:
    r = os.environ.get("EDNA_RCLONE_REMOTE", "edna-drive:")
    return r if ":" in r else r + ":"


def _join(base: str, *parts: str) -> str:
    return base.rstrip("/") + ("" if base.endswith(":") else "/") + "/".join(parts) if parts else base


def rclone_bin() -> Optional[str]:
    explicit = os.environ.get("EDNA_RCLONE_BIN")
    if explicit:
        return explicit if os.path.exists(explicit) else None
    found = shutil.which("rclone")
    if found:
        return found
    local = os.path.expanduser("~/.local/bin/rclone")
    return local if os.path.exists(local) else None


def _run(args: List[str], timeout: int = 3600) -> subprocess.CompletedProcess:
    return subprocess.run([rclone_bin()] + args, capture_output=True, text=True, timeout=timeout)


def check() -> dict:
    """Is a backup possible right now? {configured, remote, reason}."""
    rem = remote()
    if not rclone_bin():
        return {"configured": False, "remote": rem, "reason": "rclone is not installed"}
    try:
        names = _run(["listremotes"], timeout=30).stdout.split()
    except (OSError, subprocess.SubprocessError) as e:
        return {"configured": False, "remote": rem, "reason": f"could not run rclone: {e}"}
    name = rem.split(":")[0] + ":"
    if name not in names:
        return {"configured": False, "remote": rem,
                "reason": f"rclone remote '{name}' is not set up -- run: {SETUP_HINT}"}
    return {"configured": True, "remote": rem, "reason": None}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_status(app_dir: str, status: dict) -> None:
    os.makedirs(app_dir, exist_ok=True)
    tmp = os.path.join(app_dir, STATUS_FILE + ".tmp")
    with open(tmp, "w") as f:
        json.dump(status, f, indent=2)
    os.replace(tmp, os.path.join(app_dir, STATUS_FILE))


def read_status(app_dir: str) -> Optional[dict]:
    path = os.path.join(app_dir, STATUS_FILE)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def snapshot_db(app_dir: str) -> Optional[str]:
    """A consistent copy of the live database, safe to upload while the API runs."""
    src = os.path.join(app_dir, DB_NAME)
    if not os.path.exists(src):
        return None
    os.makedirs(os.path.join(app_dir, "tmp"), exist_ok=True)
    dst = os.path.join(app_dir, "tmp", DB_NAME + ".snapshot")
    s, d = sqlite3.connect(src), sqlite3.connect(dst)
    try:
        s.backup(d)
    finally:
        d.close()
        s.close()
    return dst


# ------------------------------------------------------------- deletion ---

def _load_pending(app_dir: str) -> List[str]:
    path = os.path.join(app_dir, PENDING_FILE)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def _save_pending(app_dir: str, paths: List[str]) -> None:
    path = os.path.join(app_dir, PENDING_FILE)
    if not paths:
        if os.path.exists(path):
            os.remove(path)
        return
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(sorted(set(paths)), f, indent=2)
    os.replace(tmp, path)


def _gone(stderr: str) -> bool:
    """rclone's way of saying the thing to delete is not there (never backed up,
    or already deleted) -- which, for a deletion, is success."""
    e = stderr.lower()
    return "not found" in e or "doesn't exist" in e or "does not exist" in e


def _delete_paths(paths: List[str]) -> (List[str], List[str], Optional[str]):
    """Deletes remote-relative paths ("runs/<id>" folders, "uploads/<file>" files).
    Returns (deleted, failed, last_error)."""
    deleted, failed, err = [], [], None
    for rel in paths:
        is_dir = rel.startswith("runs/") and rel.count("/") == 1
        r = _run(["purge" if is_dir else "deletefile", _join(remote(), rel)], timeout=300)
        if r.returncode == 0 or _gone(r.stderr):
            deleted.append(rel)
        else:
            failed.append(rel)
            err = r.stderr.strip()[-400:] or f"rclone exited with {r.returncode}"
    return deleted, failed, err


def delete_remote(app_dir: str, paths: List[str]) -> dict:
    """Removes a deleted run's files from the cloud copy now; anything that
    cannot be removed right now is queued and retried by the next backup."""
    if not paths:
        return {"ok": True, "deleted": [], "queued": []}
    if not enabled():
        return {"ok": True, "deleted": [], "queued": [], "skipped": "cloud backup is switched off (EDNA_CLOUD_BACKUP=off)"}
    chk = check()
    if not chk["configured"]:
        _save_pending(app_dir, _load_pending(app_dir) + paths)
        return {"ok": False, "deleted": [], "queued": paths, "error": chk["reason"]}
    with _remote_lock:
        try:
            deleted, failed, err = _delete_paths(paths)
        except (OSError, subprocess.SubprocessError) as e:
            deleted, failed, err = [], list(paths), f"{type(e).__name__}: {e}"
    if failed:
        _save_pending(app_dir, _load_pending(app_dir) + failed)
    return {"ok": not failed, "deleted": deleted, "queued": failed, "error": err, "remote": chk["remote"]}


def push(app_dir: str) -> dict:
    """Copies runs, uploads and a database snapshot to the remote. Never raises."""
    with _remote_lock:
        return _push(app_dir)


def _push(app_dir: str) -> dict:
    started, t0 = _now(), time.time()
    status = {"action": "push", "started_at": started, "remote": remote()}
    ok = check()
    if not ok["configured"]:
        status.update(ok=False, finished_at=_now(), seconds=0.0, error=ok["reason"])
        _write_status(app_dir, status)
        return status
    try:
        # Finish any deletions that could not reach the cloud when they were made.
        pending = _load_pending(app_dir)
        if pending:
            _, still, _ = _delete_paths(pending)
            _save_pending(app_dir, still)
            status["pending_deletes"] = len(still)
        # Everything except the live database (snapshotted below) and scratch files.
        r = _run(["copy", app_dir, remote(), "--exclude", "tmp/**", "--exclude", f"{DB_NAME}*",
                  "--exclude", f"{STATUS_FILE}*", "--exclude", f"{PENDING_FILE}*",
                  "--transfers", "4", "--stats-one-line", "--stats", "0"])
        if r.returncode != 0:
            raise RuntimeError(r.stderr.strip()[-600:] or f"rclone exited with {r.returncode}")
        snap = snapshot_db(app_dir)
        if snap:
            r = _run(["copyto", snap, _join(remote(), DB_NAME)])
            os.remove(snap)
            if r.returncode != 0:
                raise RuntimeError(r.stderr.strip()[-600:] or f"rclone exited with {r.returncode}")
        n_runs = len(os.listdir(os.path.join(app_dir, "runs"))) if os.path.isdir(os.path.join(app_dir, "runs")) else 0
        status.update(ok=True, error=None, runs_in_backup_folder=n_runs)
    except (OSError, subprocess.SubprocessError, RuntimeError, sqlite3.Error) as e:
        status.update(ok=False, error=f"{type(e).__name__}: {e}")
    status.update(finished_at=_now(), seconds=round(time.time() - t0, 1))
    if status["ok"]:
        status["last_success_at"] = status["finished_at"]
    else:
        prev = read_status(app_dir) or {}
        status["last_success_at"] = prev.get("last_success_at")
    _write_status(app_dir, status)
    return status


def pull(app_dir: str, overwrite_db: bool = False) -> dict:
    """Restores runs and uploads from the remote; the database only if absent
    locally (or overwrite_db). Stop the API first: it holds the database open."""
    ok = check()
    if not ok["configured"]:
        return {"ok": False, "error": ok["reason"]}
    os.makedirs(app_dir, exist_ok=True)
    r = _run(["copy", remote(), app_dir, "--exclude", DB_NAME, "--exclude", f"{STATUS_FILE}*",
              "--exclude", f"{PENDING_FILE}*"])
    if r.returncode != 0:
        return {"ok": False, "error": r.stderr.strip()[-600:]}
    db = os.path.join(app_dir, DB_NAME)
    restored_db = False
    if overwrite_db or not os.path.exists(db):
        r = _run(["copyto", _join(remote(), DB_NAME), db])
        if r.returncode != 0:
            return {"ok": False, "error": r.stderr.strip()[-600:]}
        restored_db = True
    return {"ok": True, "restored_db": restored_db, "app_dir": app_dir}


def enabled() -> bool:
    return os.environ.get("EDNA_CLOUD_BACKUP", "on").lower() not in ("off", "0", "false", "no")


class BackupScheduler:
    """Runs push() in the background, one at a time. A request made while a
    backup is in progress triggers exactly one more pass when it ends, so the
    latest run is always included without queueing a pass per run."""

    def __init__(self):
        self._lock = threading.Lock()
        self._running = False
        self._again = False
        self.app_dir: Optional[str] = None

    def request(self, app_dir: str) -> None:
        if not enabled():
            return
        with self._lock:
            self.app_dir = app_dir
            if self._running:
                self._again = True
                return
            self._running = True
        threading.Thread(target=self._loop, name="edna-cloud-backup", daemon=True).start()

    def busy(self) -> bool:
        return self._running

    def _loop(self) -> None:
        while True:
            try:
                push(self.app_dir)
            except Exception as e:      # last resort: a backup must never take the server down
                _write_status(self.app_dir, {"action": "push", "ok": False, "error": f"{type(e).__name__}: {e}",
                                             "finished_at": _now()})
            with self._lock:
                if not self._again:
                    self._running = False
                    return
                self._again = False


scheduler = BackupScheduler()


def _default_app_dir() -> str:
    return os.environ.get("EDNA_APP_DIR", os.path.join(ROOT, "data", "app"))


def main(argv: List[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "status"
    app_dir = _default_app_dir()
    if cmd == "status":
        print(json.dumps({**check(), "app_dir": app_dir, "last": read_status(app_dir)}, indent=2))
        return 0
    if cmd == "push":
        res = push(app_dir)
        print(json.dumps(res, indent=2))
        return 0 if res["ok"] else 1
    if cmd == "pull":
        res = pull(app_dir, overwrite_db="--overwrite-db" in argv)
        print(json.dumps(res, indent=2))
        return 0 if res["ok"] else 1
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
