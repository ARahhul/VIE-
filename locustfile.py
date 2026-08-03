"""Load test for concurrent uploads (PRD Phase 10 deliverable).

    locust -f locustfile.py --host http://localhost:8000

Needs a small sample clip at ./locust_sample.mp4 (not checked in — generate
one with OpenCV, or point SAMPLE_CLIP_PATH at any short local .mp4).
"""

import os
import time

from locust import HttpUser, between, task

SAMPLE_CLIP_PATH = os.environ.get("SAMPLE_CLIP_PATH", "locust_sample.mp4")


class InvestigatorUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def ingest_and_poll(self):
        with open(SAMPLE_CLIP_PATH, "rb") as f:
            resp = self.client.post("/ingest", files={"video": ("clip.mp4", f, "video/mp4")})
        if resp.status_code != 200:
            return

        job_id = resp.json()["job"]["id"]
        for _ in range(60):
            job = self.client.get(f"/jobs/{job_id}").json()
            if job["status"] in ("completed", "failed"):
                break
            time.sleep(1)
