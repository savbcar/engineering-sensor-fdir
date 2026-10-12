from sensor_fdir.config import load_config
from sensor_fdir.pipeline import run_pipeline

result = run_pipeline(load_config("config/default.json"), "artifacts/demo")
print(result["comparison"])
