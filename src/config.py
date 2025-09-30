from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "data"
OUTPUTS      = PROJECT_ROOT / "outputs"
MODELS_DIR   = OUTPUTS / "models"
REPORTS_DIR  = OUTPUTS / "reports"

TARGET       = "stroke"
RANDOM_STATE = 42
TEST_SIZE    = 0.2
CV_FOLDS     = 5

for p in (OUTPUTS, MODELS_DIR, REPORTS_DIR):
    p.mkdir(parents=True, exist_ok=True)
