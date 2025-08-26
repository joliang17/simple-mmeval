import os

env_dir = os.getenv('ENV_DIR') or ""

series_mapping = {
    "pllava": ["pllava-7b", "pllava-13b", "pllava-34b"]
}

series_infer_env_mapping = {
    "pllava": {
        "env": os.path.join(env_dir, "pllava"),
        "infer_file": "pllava.py",
    }
}