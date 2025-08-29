import os

from roboflow import Roboflow
from ultralytics import YOLO

if __name__ == "__main__":
  # Use absolute path to avoid multiprocessing issues
  print(os.getcwd())

  HOME = os.path.join(os.getcwd(), "training", "resources_model")

  os.makedirs(os.path.join(HOME, "datasets"), exist_ok=True)
  os.chdir(os.path.join(HOME, "datasets"))

  print(os.getcwd())

  rf = Roboflow(api_key="your-api-key")
  project = rf.workspace("joshmiquel").project("dofus-resources-5vl3i")
  version = project.version(6)
  dataset = version.download("yolov11")

  os.chdir(HOME)

  print(os.getcwd())

  model = YOLO(os.path.join("runs", "detect", "v6", "weights", "best.pt"))
  results = model.val(
    data=f"{dataset.location}/data.yaml",
    imgsz=640,
    device=0,
    plots=True,
  )
