import os

from dotenv import load_dotenv
from roboflow import Roboflow
from ultralytics import YOLO

from config import (
  CURRENT_PROJECT_FOLDER,
  ROBOFLOW_PROJECT_ID,
  ROBOFLOW_PROJECT_VERSION,
  ROBOFLOW_PROJECT_WORKSPACE,
  YOLO_VERSION,
)

if __name__ == "__main__":
  load_dotenv()

  HOME = os.path.join(os.getcwd(), CURRENT_PROJECT_FOLDER)

  os.makedirs(os.path.join(HOME, "datasets"), exist_ok=True)
  os.chdir(os.path.join(HOME, "datasets"))

  if os.getenv("ROBOFLOW_API_KEY") is None:
    raise ValueError("ROBOFLOW_API_KEY is not set. Please set it in the .env file.")

  rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
  project = rf.workspace(ROBOFLOW_PROJECT_WORKSPACE).project(ROBOFLOW_PROJECT_ID)
  version = project.version(ROBOFLOW_PROJECT_VERSION)
  dataset = version.download(YOLO_VERSION)

  os.chdir(HOME)

  model = YOLO(os.path.join("runs", "detect", f"v{ROBOFLOW_PROJECT_VERSION}", "weights", "best.pt"))
  results = model.val(
    data=f"{dataset.location}/data.yaml",
    imgsz=640,
    device=0,
    plots=True,
  )
