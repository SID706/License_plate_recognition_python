# Project notes for agents

Licence plate recognition demo: YOLO (`best.pt`) + EasyOCR behind a Flask page on port 3000.

## Running

```bash
docker compose -f docker-compose.base44.yml up -d --build
```

- `docker-compose.base44.yml` builds `Dockerfile.base44` (Python 3.11 + CPU torch + ultralytics +
  easyocr) and bind-mounts the repo at `/app`; flask `--debug` reloads on source edits.
- The trained weights are NOT in the repo. `Dockerfile.base44` downloads them from the Google
  Drive link in `README.md` to `/opt/lpr/models/best.pt` at build time; compose passes
  `MODEL_PATH`. If the Drive link dies, the image build fails at that step with a clear message.
- EasyOCR english models are pre-downloaded in the image, so the first request is not a cold
  start. `YOLO_CONFIG_DIR=/tmp/ultralytics` keeps Ultralytics from writing into the bind mount.

## Quirks

- `best.pt` is an older Ultralytics checkpoint and needs `omegaconf` (pinned in
  `requirements-web.txt`); without it Ultralytics tries to `pip install` at first predict.
- The original `predict.py` imports `ultralytics.yolo.*` internals from an old release and will
  not import against the installed ultralytics; `webapp.py` calls the supported `YOLO` API
  instead while keeping predict.py's OCR selection rules (`read_plate_text`).
- `number_plate.py` needs a real webcam (`cv2.VideoCapture(0)`), so it cannot run in the sandbox.

## Verifying

```bash
curl -s http://localhost:3000/health
curl -s -F "image=@static/sample_car.jpg" http://localhost:3000/api/detect
```

The sample photo should come back with one plate and the text `MHOIA5755`. CPU inference takes
roughly 0.3–3 s per image.
