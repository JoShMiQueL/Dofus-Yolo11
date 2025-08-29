import os

from dotenv import load_dotenv
from roboflow import Roboflow

from config import (
  ROBOFLOW_PROJECT_ID,
  ROBOFLOW_PROJECT_VERSION,
  ROBOFLOW_PROJECT_WORKSPACE,
  YOLO_VERSION,
)

if __name__ == "__main__":
  load_dotenv()

  if os.getenv("ROBOFLOW_API_KEY") is None:
    raise ValueError("ROBOFLOW_API_KEY is not set. Please set it in the .env file.")

  rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
  project = rf.workspace(ROBOFLOW_PROJECT_WORKSPACE).project(ROBOFLOW_PROJECT_ID)
  version = project.version(ROBOFLOW_PROJECT_VERSION)

  version.deploy(YOLO_VERSION, f"training/resources_model/runs/detect/v{ROBOFLOW_PROJECT_VERSION}")
