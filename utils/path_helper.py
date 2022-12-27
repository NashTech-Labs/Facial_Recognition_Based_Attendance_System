from pathlib import Path
import os

project_dir = Path(__file__).resolve().parents[1]


utils_dir = project_dir / "utils"
haarCascade_file = os.path.join(utils_dir, "haarcascade_frontalface_default.xml")
database = project_dir / "database"
images = database / "images"
labels = database / "labels"
attendance_database = database / "attendance_database"
model = project_dir / "model"
trained_model = model / "trained_model"
