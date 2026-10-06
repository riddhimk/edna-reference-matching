"""
Tests for the cloud backup of the run history (src/cloud_backup.py).

A fake `rclone` stands in for the real one: it understands `listremotes`,
`copy` and `copyto`, and maps the remote "fake:" onto a local folder, so the
real logic -- what is uploaded, the database snapshot, restore -- runs end to
end with no network and no Google account.
"""
import os
import sqlite3
import stat
import sys
import textwrap

import pytest

from src import cloud_backup as cb

FAKE_RCLONE = textwrap.dedent('''\
    #!{python}
    import os, shutil, sys
    CLOUD = os.environ["FAKE_CLOUD"]
    args = sys.argv[1:]
    if os.environ.get("FAKE_RCLONE_FAIL") and args[0] != "listremotes":
        sys.stderr.write("simulated failure: 403 insufficient permissions\\n"); sys.exit(7)
    def loc(p):  # "fake:sub/x" -> CLOUD/sub/x ; local paths pass through
        return os.path.join(CLOUD, p.split(":", 1)[1]) if p.startswith("fake:") else p
    if args[0] == "listremotes":
        print("fake:")
    elif args[0] == "copy":
        src, dst = loc(args[1]), loc(args[2])
        excludes = [args[i + 1] for i, a in enumerate(args) if a == "--exclude"]
        import fnmatch
        for root, _dirs, files in os.walk(src):
            for f in files:
                rel = os.path.relpath(os.path.join(root, f), src)
                if any(fnmatch.fnmatch(rel, pat) for pat in excludes):
                    continue
                out = os.path.join(dst, rel)
                os.makedirs(os.path.dirname(out), exist_ok=True)
                shutil.copy2(os.path.join(root, f), out)
    elif args[0] == "purge":
        target = loc(args[1])
        if not os.path.isdir(target):
            sys.stderr.write("directory not found\\n"); sys.exit(3)
        shutil.rmtree(target)
    elif args[0] == "deletefile":
        target = loc(args[1])
        if not os.path.exists(target):
            sys.stderr.write("object not found\\n"); sys.exit(4)
        os.remove(target)
    elif args[0] == "copyto":
        src, dst = loc(args[1]), loc(args[2])
        if not os.path.exists(src):
            sys.stderr.write("not found\\n"); sys.exit(3)
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        shutil.copy2(src, dst)
    ''').format(python=sys.executable)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    rclone = tmp_path / "rclone"
    rclone.write_text(FAKE_RCLONE)
    rclone.chmod(rclone.stat().st_mode | stat.S_IEXEC)
    cloud = tmp_path / "cloud"
    cloud.mkdir()
    monkeypatch.setenv("EDNA_RCLONE_BIN", str(rclone))
    monkeypatch.setenv("EDNA_RCLONE_REMOTE", "fake:")
    monkeypatch.setenv("FAKE_CLOUD", str(cloud))

    app = tmp_path / "app"
    (app / "runs" / "r1").mkdir(parents=True)
    (app / "runs" / "r1" / "results.csv").write_text("Read_ID\nx\n")
    (app / "uploads").mkdir()
    (app / "uploads" / "abc.fastq.gz").write_bytes(b"\x1f\x8bdata")
    (app / "tmp").mkdir()
    (app / "tmp" / "half-uploaded.upload").write_text("scratch")
    db = sqlite3.connect(app / "edna_app.db")
    db.execute("CREATE TABLE runs (id TEXT)")
    db.execute("INSERT INTO runs VALUES ('r1')")
    db.commit()
    db.close()
    return app, cloud


def _files(root):
    return sorted(os.path.relpath(os.path.join(d, f), root) for d, _, fs in os.walk(root) for f in fs)


def test_push_copies_only_run_data(env):
    app, cloud = env
    res = cb.push(str(app))
    assert res["ok"], res
    assert _files(cloud) == ["edna_app.db", "runs/r1/results.csv", "uploads/abc.fastq.gz"]
    # the uploaded database is a real, consistent copy
    assert sqlite3.connect(cloud / "edna_app.db").execute("SELECT id FROM runs").fetchall() == [("r1",)]
    assert not (app / "tmp" / "edna_app.db.snapshot").exists()        # snapshot cleaned up
    assert cb.read_status(str(app))["last_success_at"] == res["finished_at"]


def test_deleting_a_run_locally_never_deletes_it_from_the_backup(env):
    app, cloud = env
    cb.push(str(app))
    os.remove(app / "runs" / "r1" / "results.csv")
    os.rmdir(app / "runs" / "r1")
    assert cb.push(str(app))["ok"]
    assert (cloud / "runs" / "r1" / "results.csv").exists()


def test_pull_restores_runs_and_missing_database(env, tmp_path):
    app, cloud = env
    cb.push(str(app))
    fresh = tmp_path / "restored"
    res = cb.pull(str(fresh))
    assert res["ok"] and res["restored_db"], res
    assert _files(fresh) == ["edna_app.db", "runs/r1/results.csv", "uploads/abc.fastq.gz"]


def test_pull_keeps_an_existing_database_unless_asked(env, tmp_path):
    app, _ = env
    cb.push(str(app))
    fresh = tmp_path / "restored"
    fresh.mkdir()
    (fresh / "edna_app.db").write_text("local")
    assert cb.pull(str(fresh))["restored_db"] is False
    assert (fresh / "edna_app.db").read_text() == "local"


def test_failures_are_reported_not_raised(env, monkeypatch):
    app, _ = env
    cb.push(str(app))
    first_success = cb.read_status(str(app))["last_success_at"]
    monkeypatch.setenv("FAKE_RCLONE_FAIL", "1")
    res = cb.push(str(app))
    assert res["ok"] is False and "insufficient permissions" in res["error"]
    assert cb.read_status(str(app))["last_success_at"] == first_success     # last good backup still known


def test_unconfigured_remote_is_explained(env, monkeypatch):
    app, _ = env
    monkeypatch.setenv("EDNA_RCLONE_REMOTE", "missing:")
    chk = cb.check()
    assert chk["configured"] is False and "rclone config create" in chk["reason"]
    assert cb.push(str(app))["ok"] is False


def test_switched_off_backups_do_nothing(env, monkeypatch):
    app, cloud = env
    monkeypatch.setenv("EDNA_CLOUD_BACKUP", "off")
    cb.BackupScheduler().request(str(app))
    assert _files(cloud) == []


def test_status_endpoint(env, monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from src import app_routes
    from src.api import app as api_app
    app_routes.state.configure(str(tmp_path / "api_app"))
    try:
        body = TestClient(api_app).get("/backup/status").json()
        assert body["configured"] is True and body["remote"] == "fake:" and body["enabled"] is False
    finally:
        app_routes.state.store = None


# ------------------------------------------------------------- deletion ---

@pytest.fixture()
def on(monkeypatch):
    monkeypatch.setenv("EDNA_CLOUD_BACKUP", "on")


def test_delete_remote_removes_run_folder_and_upload(env, on):
    app, cloud = env
    cb.push(str(app))
    res = cb.delete_remote(str(app), ["runs/r1", "uploads/abc.fastq.gz"])
    assert res["ok"] and res["queued"] == [], res
    assert not (cloud / "runs" / "r1").exists() and not (cloud / "uploads" / "abc.fastq.gz").exists()
    assert (cloud / "edna_app.db").exists()                      # nothing else touched


def test_deleting_what_was_never_backed_up_is_fine(env, on):
    app, _ = env
    res = cb.delete_remote(str(app), ["runs/never-uploaded", "uploads/nope.gz"])
    assert res["ok"] and sorted(res["deleted"]) == ["runs/never-uploaded", "uploads/nope.gz"]


def test_failed_cloud_delete_is_queued_and_finished_by_next_backup(env, on, monkeypatch):
    app, cloud = env
    cb.push(str(app))
    monkeypatch.setenv("FAKE_RCLONE_FAIL", "1")
    res = cb.delete_remote(str(app), ["runs/r1"])
    assert res["ok"] is False and res["queued"] == ["runs/r1"]
    assert (cloud / "runs" / "r1").exists()
    monkeypatch.delenv("FAKE_RCLONE_FAIL")
    os.remove(app / "runs" / "r1" / "results.csv"); os.rmdir(app / "runs" / "r1")   # deleted locally meanwhile
    assert cb.push(str(app))["ok"]
    assert not (cloud / "runs" / "r1").exists()
    assert cb._load_pending(str(app)) == []


def test_switched_off_backup_never_touches_the_cloud_on_delete(env):
    app, cloud = env
    os.environ.pop("EDNA_CLOUD_BACKUP", None)
    cb.push(str(app))
    os.environ["EDNA_CLOUD_BACKUP"] = "off"
    res = cb.delete_remote(str(app), ["runs/r1"])
    assert "skipped" in res and (cloud / "runs" / "r1").exists()
