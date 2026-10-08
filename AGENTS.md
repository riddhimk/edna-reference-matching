# AGENTS.md

## Long-running jobs (training, evaluations)
- Stream progress live to the terminal and give a status update every 5 minutes confirming the job is progressing.
- Set PYTHONUNBUFFERED=1 for Python jobs whose logs will be tailed. Block buffering hides progress and loses it if the process is killed.
- Launch long jobs with `setsid nohup ... &` so they survive session teardown.
- Make the heartbeat also fire on failure (grep for Traceback, Killed, OOM) so silence is never mistaken for progress.
- Verify liveness via CPU-time progression, not just process existence.

Why: a fine-tuning run was once killed on session teardown and its buffered output was lost, so progress was unverifiable.
