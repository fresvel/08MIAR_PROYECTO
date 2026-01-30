docker compose exec jupyter bash -lc "jupyter nbconvert --to pdf --execute --ExecutePreprocessor.timeout=0 --output main --output-dir /workspace /workspace/main.ipynb"
