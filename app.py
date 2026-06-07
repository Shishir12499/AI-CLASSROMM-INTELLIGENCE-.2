import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont


try:
    import cv2
except Exception:
    cv2 = None

BASE_DIR = Path(__file__).resolve().parent
MODEL_NAME = "yolov8n.pt"
MODEL_DIR = Path.home() / ".cache" / "ai-classroom-intelligence"
MODEL_PATH = MODEL_DIR / MODEL_NAME
LEGACY_MODEL_PATHS = [BASE_DIR / MODEL_NAME, BASE_DIR / "yolov8s.pt"]
MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"
MIN_MODEL_SIZE_MB = 5
SAMPLE_IMAGE_PATH = BASE_DIR / "sample_classroom.jpg"
STUDENTS_PATH = BASE_DIR / "students.csv"
STUDENT_COLUMNS = ["student_id", "name", "class_name", "roll_no", "email", "phone", "registered_at"]

TARGET_CLASSES = {
    "person": "Person Present",
    "chair": "Chair Detected",
    "laptop": "Laptop Detected",
    "cell phone": "Mobile Phone Detected",
    "mobile phone": "Mobile Phone Detected",
}

CLASS_CONFIDENCE = {
    "person": 0.12,
    "chair": 0.12,
    "laptop": 0.10,
    "cell phone": 0.08,
    "mobile phone": 0.08,
}


def create_demo_classroom_image() -> Image.Image:
    image = Image.new("RGB", (1100, 700), "#f4f1e8")
    draw = ImageDraw.Draw(image)

    draw.rectangle((0, 0, 1100, 95), fill="#d9e2dc")
    draw.rectangle((80, 115, 1020, 260), fill="#eef7f4", outline="#607d72", width=4)
    draw.text((430, 170), "AI Classroom Intelligence", fill="#27423b")

    desk_y = [350, 500]
    for row, y in enumerate(desk_y):
        for col in range(4):
            x = 90 + col * 245
            draw.rounded_rectangle((x, y, x + 185, y + 55), radius=8, fill="#c9b28e", outline="#7e684d")
            draw.rectangle((x + 20, y + 60, x + 165, y + 105), fill="#8f775a")
            draw.rounded_rectangle((x + 45, y + 95, x + 140, y + 155), radius=10, fill="#5f7180")

            if not (row == 1 and col in [1, 3]):
                cx = x + 92
                draw.ellipse((cx - 24, y - 82, cx + 24, y - 34), fill="#8d5a44")
                draw.rectangle((cx - 35, y - 35, cx + 35, y + 25), fill="#315d89")

            if col in [0, 2]:
                draw.rectangle((x + 45, y + 12, x + 125, y + 42), fill="#2f3740", outline="#14181d")
                draw.rectangle((x + 51, y + 17, x + 119, y + 35), fill="#87bde3")

            if col == 3:
                draw.rounded_rectangle((x + 128, y + 14, x + 150, y + 46), radius=5, fill="#20262c")

    draw.rectangle((875, 280, 995, 335), fill="#50616e")
    draw.text((905, 298), "Laptop", fill="white")
    draw.text((120, 625), "Demo scene: people, empty chairs, laptops, and mobile phones", fill="#27423b")
    return image


def load_sample_image() -> Image.Image:
    try:
        return Image.open(SAMPLE_IMAGE_PATH).convert("RGB")
    except Exception:
        return create_demo_classroom_image()


@st.cache_resource
def load_model():
    try:
        from ultralytics import YOLO
    except Exception as exc:
        return None, f"Ultralytics/OpenCV could not be imported: {exc}"

    model_path = MODEL_PATH if MODEL_PATH.exists() else next(
        (path for path in LEGACY_MODEL_PATHS if path.exists()),
        MODEL_PATH,
    )
    if not model_path.exists():
        try:
            download_model()
            model_path = MODEL_PATH
        except Exception as exc:
            return None, f"Could not download {MODEL_NAME} automatically: {exc}"
    model_size_mb = model_path.stat().st_size / (1024 * 1024)
    if model_size_mb < MIN_MODEL_SIZE_MB:
        return None, (
            f"{MODEL_NAME} looks incomplete or corrupted. Current size: {model_size_mb:.2f} MB. "
            "Download a fresh copy using the button below."
        )
    try:
        return YOLO(str(model_path)), None
    except Exception as exc:
        return None, f"Could not load {MODEL_NAME}: {exc}"


def download_model():
    MODEL_DIR.mkdir(exist_ok=True)
    temp_path = MODEL_PATH.with_suffix(".pt.download")
    urllib.request.urlretrieve(MODEL_URL, temp_path)
    temp_size_mb = temp_path.stat().st_size / (1024 * 1024)
    if temp_size_mb < MIN_MODEL_SIZE_MB:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError("Downloaded model file is too small. Please check your internet connection.")
    temp_path.replace(MODEL_PATH)


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0


def overlap_ratio(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    return inter / area_a if area_a else 0


def dedupe_detections(detections, iou_threshold=0.35):
    filtered = []
    for det in sorted(detections, key=lambda item: item["confidence"], reverse=True):
        duplicate = any(
            det["class_name"] == kept["class_name"]
            and box_iou(det["box"], kept["box"]) > iou_threshold
            for kept in filtered
        )
        if not duplicate:
            filtered.append(det)
    return filtered


def collect_detections(result, offset_x=0, offset_y=0):
    names = result.names
    detections = []
    for box in result.boxes:
        class_name = names[int(box.cls[0])]
        confidence = float(box.conf[0])
        if class_name not in TARGET_CLASSES:
            continue
        if confidence < CLASS_CONFIDENCE.get(class_name, 0.15):
            continue

        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
        detections.append(
            {
                "class_name": class_name,
                "confidence": confidence,
                "box": (x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y),
            }
        )
    return detections


def tile_boxes(width, height, cols=3, rows=2, overlap=0.22):
    tile_w = width / cols
    tile_h = height / rows
    boxes = []
    for row in range(rows):
        for col in range(cols):
            x1 = max(0, int(col * tile_w - tile_w * overlap))
            y1 = max(0, int(row * tile_h - tile_h * overlap))
            x2 = min(width, int((col + 1) * tile_w + tile_w * overlap))
            y2 = min(height, int((row + 1) * tile_h + tile_h * overlap))
            boxes.append((x1, y1, x2, y2))
    return boxes


def is_chair_empty(chair_box, person_boxes):
    cx1, cy1, cx2, cy2 = chair_box
    chair_w = cx2 - cx1
    chair_h = cy2 - cy1
    expanded_chair = (
        cx1 - chair_w * 0.25,
        cy1 - chair_h * 0.75,
        cx2 + chair_w * 0.25,
        cy2 + chair_h * 0.20,
    )

    for person_box in person_boxes:
        px1, py1, px2, py2 = person_box
        person_bottom_center = ((px1 + px2) / 2, py2)
        bottom_inside_chair_area = (
            expanded_chair[0] <= person_bottom_center[0] <= expanded_chair[2]
            and expanded_chair[1] <= person_bottom_center[1] <= expanded_chair[3]
        )
        if bottom_inside_chair_area or overlap_ratio(chair_box, person_box) > 0.08:
            return False
    return True


def draw_detections(image, detections, empty_chair_boxes):
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    colors = {
        "person": "#175cff",
        "chair": "#b0008b",
        "laptop": "#00c9a7",
        "cell phone": "#ff9f1c",
        "mobile phone": "#ff9f1c",
    }

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        class_name = det["class_name"]
        label = f"{class_name} {det['confidence']:.2f}"
        color = colors.get(class_name, "#ffffff")
        draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
        text_box = draw.textbbox((x1, y1), label)
        label_width = text_box[2] - text_box[0] + 8
        draw.rectangle((x1, max(0, y1 - 24), x1 + label_width, y1), fill=color)
        draw.text((x1 + 4, max(0, y1 - 22)), label, fill="white")

    for box in empty_chair_boxes:
        x1, y1, x2, y2 = [int(v) for v in box]
        draw.rectangle((x1, y1, x2, y2), outline="#ff3b30", width=5)

    return annotated


def run_detection(image: Image.Image):
    model, error = load_model()
    if model is None:
        return image, {}, error

    source = np.array(image)
    width, height = image.size
    detections = []

    full_result = model.predict(
        source,
        conf=0.08,
        iou=0.45,
        imgsz=1280,
        max_det=500,
        augment=True,
        verbose=False,
    )[0]
    detections.extend(collect_detections(full_result))

    for x1, y1, x2, y2 in tile_boxes(width, height):
        tile = source[y1:y2, x1:x2]
        tile_result = model.predict(
            tile,
            conf=0.08,
            iou=0.45,
            imgsz=960,
            max_det=300,
            augment=True,
            verbose=False,
        )[0]
        detections.extend(collect_detections(tile_result, x1, y1))

    detections = dedupe_detections(detections, iou_threshold=0.35)
    person_boxes = [det["box"] for det in detections if det["class_name"] == "person"]
    chair_boxes = [det["box"] for det in detections if det["class_name"] == "chair"]
    empty_chair_boxes = [box for box in chair_boxes if is_chair_empty(box, person_boxes)]

    counts = {
        "person": len(person_boxes),
        "chair": len(chair_boxes),
        "empty_chair": len(empty_chair_boxes),
        "occupied_chair": len(chair_boxes) - len(empty_chair_boxes),
        "laptop": sum(1 for det in detections if det["class_name"] == "laptop"),
        "cell phone": sum(1 for det in detections if det["class_name"] in ["cell phone", "mobile phone"]),
    }

    annotated = draw_detections(image, detections, empty_chair_boxes)

    return annotated, counts, None


def run_fast_detection(frame_rgb):
    model, error = load_model()
    if model is None:
        return frame_rgb, {}, error

    result = model.predict(
        frame_rgb,
        conf=0.20,
        iou=0.45,
        imgsz=640,
        max_det=200,
        verbose=False,
    )[0]
    detections = dedupe_detections(collect_detections(result), iou_threshold=0.35)
    person_boxes = [det["box"] for det in detections if det["class_name"] == "person"]
    chair_boxes = [det["box"] for det in detections if det["class_name"] == "chair"]
    empty_chair_boxes = [box for box in chair_boxes if is_chair_empty(box, person_boxes)]
    counts = {
        "person": len(person_boxes),
        "chair": len(chair_boxes),
        "empty_chair": len(empty_chair_boxes),
        "occupied_chair": len(chair_boxes) - len(empty_chair_boxes),
        "laptop": sum(1 for det in detections if det["class_name"] == "laptop"),
        "cell phone": sum(1 for det in detections if det["class_name"] in ["cell phone", "mobile phone"]),
    }
    annotated = draw_detections(Image.fromarray(frame_rgb), detections, empty_chair_boxes)
    return np.array(annotated), counts, None


def build_status(counts):
    persons = counts.get("person", 0)
    chairs = counts.get("chair", 0)
    laptops = counts.get("laptop", 0)
    mobiles = counts.get("cell phone", 0) + counts.get("mobile phone", 0)
    empty_chairs = counts.get("empty_chair", 0)
    occupied_chairs = counts.get("occupied_chair", 0)

    return {
        "Classroom Status": "Occupied" if persons > 0 else "Empty",
        "Persons Detected": persons,
        "Person Presence": "Person Present" if persons > 0 else "Person Absent",
        "Chairs Detected": chairs,
        "Occupied Chairs": occupied_chairs,
        "Estimated Empty Chairs": empty_chairs,
        "Laptops Detected": laptops,
        "Mobile Phones Detected": mobiles,
        "Device Usage": "Device Detected" if laptops + mobiles > 0 else "No Device Detected",
    }


def load_students():
    if not STUDENTS_PATH.exists():
        return pd.DataFrame(columns=STUDENT_COLUMNS)
    try:
        students = pd.read_csv(STUDENTS_PATH, dtype=str).fillna("")
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError) as exc:
        st.warning(f"Could not load students.csv: {exc}")
        return pd.DataFrame(columns=STUDENT_COLUMNS)

    for column in STUDENT_COLUMNS:
        if column not in students.columns:
            students[column] = ""
    return students[STUDENT_COLUMNS]


def save_student(student):
    students = load_students()
    students = pd.concat([students, pd.DataFrame([student])], ignore_index=True)
    try:
        students.to_csv(STUDENTS_PATH, index=False)
    except OSError as exc:
        st.error(f"Could not save students.csv: {exc}")


def render_report(counts):
    status = build_status(counts)
    for label, value in status.items():
        st.metric(label, value)


def render_model_error(error):
    if not error:
        return
    st.warning(error)
    if st.button(f"Download fresh {MODEL_NAME}"):
        with st.spinner(f"Downloading {MODEL_NAME}..."):
            try:
                download_model()
                st.cache_resource.clear()
                st.success(f"Downloaded {MODEL_NAME} successfully. Reloading the app...")
                st.rerun()
            except Exception as exc:
                st.error(f"Download failed: {exc}")
                st.info(f"You can also manually download {MODEL_NAME} and place it in {MODEL_DIR}.")
    st.info(f"The app is open. Add a valid {MODEL_NAME} to enable real detection.")


def image_detection_dashboard():
    st.subheader("Image Detection")
    uploaded_file = st.file_uploader("Upload classroom image", type=["jpg", "jpeg", "png"])

    input_image = Image.open(uploaded_file).convert("RGB") if uploaded_file else load_sample_image()

    left, right = st.columns([2, 1])
    with left:
        image_placeholder = st.empty()
        image_placeholder.image(input_image, use_container_width=True)
    with right:
        st.subheader("Classroom Report")
        if st.button("Run Detection"):
            with st.spinner("Running classroom detection..."):
                annotated_image, counts, error = run_detection(input_image)
            image_placeholder.image(annotated_image, use_container_width=True)
            render_model_error(error)
            render_report(counts)
        else:
            st.info("Upload an image, then run detection.")
            render_report({})


def teacher_dashboard():
    st.subheader("Teacher Dashboard")
    students = load_students()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Registered Students", len(students))
    col2.metric("Active Classes", students["class_name"].nunique() if not students.empty else 0)
    col3.metric("Attendance Source", "AI Vision")
    col4.metric("Model", MODEL_NAME)

    tab1, tab2 = st.tabs(["Classroom Detection", "Student Records"])
    with tab1:
        image_detection_dashboard()
    with tab2:
        st.dataframe(students, use_container_width=True, hide_index=True)
        if not students.empty:
            csv = students.to_csv(index=False).encode("utf-8")
            st.download_button("Download Student CSV", csv, "students.csv", "text/csv")


def student_dashboard():
    st.subheader("Student Dashboard")
    students = load_students()
    if students.empty:
        st.info("No students registered yet.")
        return

    student_names = students["name"].fillna("").tolist()
    selected_name = st.selectbox("Select student", student_names)
    selected = students[students["name"] == selected_name].iloc[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("Student ID", selected["student_id"])
    col2.metric("Class", selected["class_name"])
    col3.metric("Roll No", selected["roll_no"])

    st.write("Profile")
    st.dataframe(pd.DataFrame([selected]), use_container_width=True, hide_index=True)

    st.write("Detection Meaning")
    st.write(
        {
            "Person Present": "Student is visible in the classroom frame.",
            "Person Absent": "Student/person is not visible in the frame.",
            "Laptop Detected": "Laptop is visible near the seating area.",
            "Mobile Phone Detected": "Mobile phone is visible in the frame.",
        }
    )


def registration_dashboard():
    st.subheader("New Student Registration")
    with st.form("student_registration_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        student_id = col1.text_input("Student ID")
        name = col2.text_input("Full Name")
        class_name = col1.text_input("Class / Section")
        roll_no = col2.text_input("Roll No")
        email = col1.text_input("Email")
        phone = col2.text_input("Phone")
        submitted = st.form_submit_button("Register Student")

    if submitted:
        if not student_id or not name or not class_name:
            st.error("Student ID, full name, and class are required.")
        else:
            students = load_students()
            if student_id in students["student_id"].astype(str).tolist():
                st.error("This Student ID already exists.")
            else:
                save_student(
                    {
                        "student_id": student_id,
                        "name": name,
                        "class_name": class_name,
                        "roll_no": roll_no,
                        "email": email,
                        "phone": phone,
                        "registered_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )
                st.success("Student registered successfully.")

    st.write("Registered Students")
    st.dataframe(load_students(), use_container_width=True, hide_index=True)


def live_video_detection_dashboard():
    st.subheader("Live Video Detection")
    st.caption("Uses your webcam. Fast mode is used for smoother live detection.")

    if cv2 is None:
        st.error("OpenCV is not available. On Streamlit Cloud, push packages.txt and reboot the app.")
        return

    camera_index = st.number_input("Camera index", min_value=0, max_value=5, value=0, step=1)
    max_frames = st.slider("Frames to process after pressing Start", 10, 300, 80, 10)
    start = st.button("Start Live Detection")

    frame_placeholder = st.empty()
    report_placeholder = st.empty()

    if not start:
        st.info("Press Start Live Detection to open the webcam.")
        return

    cap = cv2.VideoCapture(int(camera_index))
    if not cap.isOpened():
        st.error("Could not open webcam. Try camera index 1 or close other camera apps.")
        return

    last_counts = {}
    try:
        for _ in range(max_frames):
            ok, frame_bgr = cap.read()
            if not ok:
                st.error("Could not read frame from webcam.")
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            annotated, last_counts, error = run_fast_detection(frame_rgb)
            frame_placeholder.image(annotated, channels="RGB", use_container_width=True)
            with report_placeholder.container():
                render_model_error(error)
                render_report(last_counts)
    finally:
        cap.release()


def video_file_detection_dashboard():
    st.subheader("Uploaded Video Detection")

    if cv2 is None:
        st.error("OpenCV is not available. On Streamlit Cloud, push packages.txt and reboot the app.")
        return

    video_file = st.file_uploader("Upload classroom video", type=["mp4", "avi", "mov", "mkv"])
    if not video_file:
        st.info("Upload a video file to scan frames.")
        return

    temp_video = BASE_DIR / "_uploaded_video.tmp"
    temp_video.write_bytes(video_file.read())
    cap = cv2.VideoCapture(str(temp_video))
    frame_step = st.slider("Analyze every Nth frame", 5, 60, 15, 5)
    max_frames = st.slider("Maximum frames to analyze", 10, 200, 50, 10)
    run = st.button("Analyze Video")

    if not run:
        cap.release()
        return

    frame_placeholder = st.empty()
    report_placeholder = st.empty()
    best_counts = {}
    processed = 0
    frame_id = 0

    while processed < max_frames:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_id += 1
        if frame_id % frame_step != 0:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        annotated, counts, error = run_fast_detection(frame_rgb)
        best_counts = {
            key: max(best_counts.get(key, 0), counts.get(key, 0))
            for key in set(best_counts) | set(counts)
        }
        frame_placeholder.image(annotated, channels="RGB", use_container_width=True)
        with report_placeholder.container():
            render_model_error(error)
            render_report(best_counts)
        processed += 1
    cap.release()


st.set_page_config(page_title="AI Classroom Intelligence", layout="wide")

st.title("AI Classroom Intelligence")
st.caption("Person presence, empty chair, laptop, and mobile phone monitoring")

with st.expander("Problem Statement", expanded=True):
    st.write(
        "Build an AI-based classroom monitoring system that analyzes classroom images "
        "or video streams to detect person presence or absence, empty chairs, laptops, "
        "mobile phones, and classroom occupancy."
    )

page = st.sidebar.radio(
    "Dashboard",
    [
        "Teacher Dashboard",
        "Student Dashboard",
        "New Student Registration",
        "Image Detection",
        "Live Video Detection",
        "Uploaded Video Detection",
    ],
)

st.sidebar.divider()
st.sidebar.write("Scenario Labels")
st.sidebar.write(
    [
        "Person Present",
        "Person Absent",
        "Chair Empty",
        "Chair Occupied",
        "Laptop Detected",
        "Mobile Phone Detected",
        "Classroom Empty",
        "Classroom Occupied",
    ]
)

if page == "Teacher Dashboard":
    teacher_dashboard()
elif page == "Student Dashboard":
    student_dashboard()
elif page == "New Student Registration":
    registration_dashboard()
elif page == "Image Detection":
    image_detection_dashboard()
elif page == "Live Video Detection":
    live_video_detection_dashboard()
else:
    video_file_detection_dashboard()
