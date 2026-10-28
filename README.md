# License Plate Recognition

Web demo: detect a number plate in a photo with the trained YOLO model, then read the
characters with EasyOCR.

```bash
docker compose -f docker-compose.base44.yml up -d --build
```

Open http://localhost:3000, upload a vehicle photo (or press **Use sample photo**).

How it fits together:

- `webapp.py` — Flask app that runs the same pipeline as `predict.py` (YOLO `best.pt`
  detection → easyocr on each detected box) behind `POST /api/detect`, plus `/health`.
- `templates/index.html` — the upload page (`static/sample_car.jpg` is the sample image,
  public domain, via Wikimedia Commons).
- `Dockerfile.base44` — development image with the Python dependencies, the EasyOCR
  models and `best.pt` (downloaded from the Google Drive link below at build time).
- The source is bind-mounted, and Flask runs with `--debug`, so edits reload.

The scripts below still work on their own (they need a webcam / local video files and
a GPU for comfortable speed).

---

for webcam:
python number_plate.py

Save image using 's'
images will be saved in "D:\project python\LPR_trained\LPR\runs\detect\cam"

run ocr notebook to extract text from saved images

for any image:
python predict.py model='best.pt' source="path of image"

example:-
python predict.py model='best.pt' source="D:\project python\LPR_trained\LPR\sources\image1.jpg"


for any video:
python predict.py model='best.pt' source="D:\project python\LPR_trained\LPR\sources\demo.mp4"



best.pt model - https://drive.google.com/file/d/1IW4Hiw8C7L__o01-WTS6RMqHWpXvQeSg/view?usp=sharing
