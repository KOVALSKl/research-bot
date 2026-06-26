from celery import Celery

from research_shared.config.settings import get_settings

settings = get_settings()

app = Celery(
    "worker",
    broker=settings.effective_celery_broker_url,
    backend=settings.effective_celery_result_backend,
    include=["worker.tasks", "worker.beat"],
)

app.conf.task_track_started = True

# Periodic scan of researches/ for new/changed files. The task itself is a
# no-op unless RESEARCHES_SCAN_ENABLED=true, so registering the schedule is
# safe even when the scanner is disabled (the default).
app.conf.beat_schedule = {
    "scan-researches": {
        "task": "worker.beat.scan_researches",
        "schedule": settings.researches_scan_interval_seconds,
    },
    "cleanup-staging": {
        "task": "worker.beat.cleanup_staging",
        "schedule": 3600.0,
    },
}
