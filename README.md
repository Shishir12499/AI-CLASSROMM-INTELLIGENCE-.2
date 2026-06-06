# AI Classroom Intelligence

This Streamlit project shows classroom monitoring scenarios such as person present or absent, empty chairs, laptop detection, and mobile phone detection.

## Files

- `app.py` - Main Streamlit dashboard.
- `requirements.txt` - Python packages needed to run the app.
- `yolov8s.pt` - Pretrained YOLOv8 small model. Put this file in the same folder as `app.py`.
- `sample_classroom.jpg` - Sample classroom image. If this file is not a valid image, the app creates a demo scene automatically.
- `README.md` - Project notes.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

If the app says `yolov8s.pt was not found`, download the model once and place it beside `app.py`.

If the app says the model is corrupted, click **Download fresh yolov8s.pt** inside the Streamlit app. This replaces the bad file with a fresh model from Ultralytics.

## Detected Classes

The app uses the pretrained `yolov8s.pt` model to detect these COCO classes:

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
Live Video Detection - webcam detection with YOLOv8s fast mode
Uploaded Video Detection - scan uploaded classroom videos
```
