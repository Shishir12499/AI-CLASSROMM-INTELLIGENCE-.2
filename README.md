# AI Classroom Intelligence

This Streamlit project shows classroom monitoring scenarios such as person present or absent, empty chairs, laptop detection, and mobile phone detection.

## Files

- `app.py` - Main Streamlit dashboard.
- `requirements.txt` - Python packages needed to run the app.
- `students.csv` - Seed student dataset used by the registration and student dashboards.
- `yolov8n.pt` - Downloaded automatically into the app cache on first use. Do not commit model files.
- `sample_classroom.jpg` - Sample classroom image. If this file is not a valid image, the app creates a demo scene automatically.
- `README.md` - Project notes.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The deployment includes YOLOv8 nano and OpenCV dependencies for image, uploaded
video, and live camera detection.

If you install manually, use:

```bash
pip install ultralytics opencv-python-headless
```

## Streamlit Cloud Deploy

This project includes `packages.txt` for OpenCV system libraries required on Streamlit Cloud.
If deployment still shows a redacted `ImportError`, open **Manage app > Reboot app** after pushing the latest files.

If the app says `yolov8n.pt` was not found, click the download button in the app or let the app download it automatically on first detection.

If the app says the model is corrupted, click the download button inside the Streamlit app. This replaces the bad file with a fresh model from Ultralytics.

## Detected Classes

The app uses the pretrained `yolov8n.pt` model to detect these COCO classes:

```text
person
chair
laptop
cell phone
```

Empty chairs are estimated by checking which detected chair boxes do not overlap with a detected person:

```text
empty chair = chair without nearby/overlapping person
```

## Output Scenarios

```text
Person Present
Person Absent
Chair Empty
Chair Occupied
Laptop Detected
Mobile Phone Detected
Classroom Empty
Classroom Occupied
Device Detected
No Device Detected
```

## Dashboards

```text
Teacher Dashboard - classroom report, image detection, student records
Student Dashboard - selected student profile and detection meaning
New Student Registration - add students into students.csv
Image Detection - upload classroom images
Live Video Detection - webcam detection with YOLOv8 nano fast mode
Uploaded Video Detection - scan uploaded classroom videos
```
