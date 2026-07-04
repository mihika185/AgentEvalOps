from typing import Optional
from sqlalchemy.orm import Session

from backend.app.database.models import Experiment


class ExperimentServiceError(Exception):
    pass

def normalize_experiment_id(experiment_id: Optional[str]) -> Optional[str]:
    if experiment_id is None:
        return None

    cleaned = experiment_id.strip()

    if not cleaned:
        return None

    return cleaned

def ensure_experiment_exists(
    db: Session,
    experiment_id: Optional[str],
) -> Optional[str]:
    cleaned_experiment_id = normalize_experiment_id(experiment_id)

    if cleaned_experiment_id is None:
        return None

    experiment = db.get(Experiment, cleaned_experiment_id)

    if experiment is None:
        raise ExperimentServiceError(
            f"Experiment was not found: {cleaned_experiment_id}"
        )

    return cleaned_experiment_id