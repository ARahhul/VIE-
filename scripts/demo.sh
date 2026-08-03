#!/usr/bin/env bash
# Short end-to-end demo: generates a sample clip, ingests it, waits for the
# report, and downloads it. Assumes the backend is already running at
# $VIE_HOST (default http://localhost:8000) — e.g. `docker compose up` or
# `uvicorn app.main:app` locally.
set -euo pipefail

VIE_HOST="${VIE_HOST:-http://localhost:8000}"
CLIP="${1:-demo_clip.mp4}"

if [ ! -f "$CLIP" ]; then
  echo "Generating a synthetic sample clip at $CLIP..."
  python -c "
import cv2, numpy as np
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('$CLIP', fourcc, 10.0, (640, 480))
for i in range(60):
    frame = np.full((480, 640, 3), 40, dtype=np.uint8)
    cv2.rectangle(frame, (40 + i * 6, 260), (160 + i * 6, 340), (0, 0, 200), -1)
    out.write(frame)
out.release()
"
fi

echo "Uploading $CLIP to $VIE_HOST/ingest..."
RESPONSE=$(curl -s -X POST "$VIE_HOST/ingest" -F "video=@$CLIP;type=video/mp4")
echo "$RESPONSE"

INCIDENT_ID=$(echo "$RESPONSE" | python -c "import sys,json; print(json.load(sys.stdin)['incident_id'])")
JOB_ID=$(echo "$RESPONSE" | python -c "import sys,json; print(json.load(sys.stdin)['job']['id'])")

echo "Incident: $INCIDENT_ID  Job: $JOB_ID"
echo "Waiting for the investigation pipeline to finish..."

for _ in $(seq 1 120); do
  STATUS=$(curl -s "$VIE_HOST/jobs/$JOB_ID" | python -c "import sys,json; print(json.load(sys.stdin)['status'])")
  echo "  status: $STATUS"
  if [ "$STATUS" = "completed" ] || [ "$STATUS" = "failed" ]; then
    break
  fi
  sleep 2
done

if [ "$STATUS" != "completed" ]; then
  echo "Job did not complete successfully (status: $STATUS)."
  exit 1
fi

OUT="report_${INCIDENT_ID}.pdf"
curl -s -o "$OUT" "$VIE_HOST/incidents/$INCIDENT_ID/report"
echo "Report saved to $OUT"
