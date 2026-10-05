# MachineSense Digital Twin

MachineSense is a demonstration dashboard and FastAPI service for predictive maintenance. It scores sensor snapshots with a bundled Random Forest model, displays four simulated factory assets in an interactive dashboard, and generates maintenance recommendations through an optional local Ollama model.

This repository is a prototype: the dashboard uses in-memory sample telemetry, not a live industrial sensor feed or persistent machine database.

## Problem Statement

Unexpected equipment failures can interrupt production and make maintenance reactive. MachineSense demonstrates how recent operating measurements can be turned into an estimated 24-hour failure risk and a concise maintenance view. It is an educational decision-support prototype, not a validated safety or control system.

## Features

- View a CNC machining center, pump, compressor, and robotic arm with sensor readings and model-generated failure risk.
- Switch each sample asset between idle, normal, and peak operating profiles.
- Review fleet and assembly-line summaries, simulated alerts, and a 3D machine visualization.
- Request structured maintenance recommendations. If Ollama or its model is unavailable, the service returns a built-in fallback recommendation.
- Explore the original dataset, modeling workflow, processed splits, and evaluation outputs included with the project.

## Architecture and Workflow

1. FastAPI serves the dashboard and JSON API from one application; the browser uses same-origin requests.
2. The dashboard reads one of four seeded machine records. Mode simulation updates that record in process memory.
3. The prediction path applies saved preprocessing, runs the Random Forest, and maps its probability to a risk state.
4. Fleet and line endpoints assemble evaluated machine records, summary values, and demo alerts for the UI.
5. The recommendation path sends the selected machine's risk and sensor values to local Ollama when available, otherwise it returns the built-in fallback.

## Run Locally

Python 3.13 is the version used by the Docker image. From the repository root, create an environment and install the dependencies:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Start the API and dashboard:

```powershell
uvicorn backend.main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). FastAPI's interactive API documentation is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs); the health endpoint is [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health).

### Optional: Ollama Recommendations

The API can use the locally installed `llama3.2` Ollama model for recommendations. Install and start Ollama separately, then download the model:

```powershell
ollama pull llama3.2
```

The dashboard and Random Forest predictions do not require Ollama. If the model service is unavailable, the API uses its built-in fallback. The separate LLM exploration notebook also references `nomic-embed-text`; that embedding model is not used by the dashboard service.

## Run With Docker

Build and start the container from the repository root:

```powershell
docker build -t machinesense-digital-twin .
docker run --rm -p 8000:8000 machinesense-digital-twin
```

The image uses Python 3.13, installs `requirements.txt`, and includes `backend/` and `models/`. Training data, notebooks, and project output archives are excluded from the Docker build context because they are not needed by the running service.

## API Overview

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/` | Serve the dashboard. |
| `GET` | `/dashboard` | Serve the dashboard. |
| `GET` | `/health` | Service status and API version. |
| `POST` | `/predict` | Score a supplied machine sensor record and return its failure probability, risk level, and recommendation. |
| `GET` | `/api/fleet` | Return evaluated records for the four sample assets. |
| `GET` | `/api/machines/{machine_id}` | Return one asset by ID or configured alias. |
| `POST` | `/api/machines/{machine_id}/simulate` | Change an asset's simulated operating mode and sensor profile. |
| `GET` | `/api/line/{line_id}` | Return the line summary, fleet, and generated alerts. |
| `POST` | `/api/copilot/chat` | Return a recommendation based on the selected asset's telemetry. |

The dashboard is served by the same FastAPI application, so its relative API requests work from the root page. The copilot request currently accepts a `query` field but generates its recommendation from the selected machine's status and sensor values; it does not use the query as LLM prompt content.

### Risk Thresholds

The classifier estimates the probability of failure within 24 hours. The API maps the probability to the included risk states:

| Risk state | Probability |
| --- | --- |
| Normal | Less than `0.19` |
| Warning | `0.19` to less than `0.37` |
| High Risk | `0.37` or greater |

These thresholds are project configuration values, not universal maintenance limits.

## Dataset Overview

The source CSV contains 24,042 timestamped records across CNC, pump, compressor, and robotic-arm equipment. Sensor fields include vibration, motor temperature, phase current, pressure, RPM, hours since maintenance, and ambient temperature; labels include `failure_within_24h`, with additional failure type, remaining-useful-life, and repair-cost columns. Training preprocessing excludes `failure_type`, `rul_hours`, and `estimated_repair_cost` from model features.

The archived split summary reports 14,142 training rows, 3,542 validation rows, and 4,566 test rows. Splitting is grouped by machine, with temporal ordering used for training and validation within each machine. The root `MachineSense_Final_Result.csv` is a separate 4,438-row results artifact; it should not be assumed to be identical to the archived test split.

## Machine Learning Approach

The training notebook compares logistic regression, decision tree, random forest, and gradient boosting classifiers for the binary `failure_within_24h` target. The selected dashboard artifact is a Random Forest. Inference fills missing numeric values with saved training medians, one-hot encodes `machine_type` and `operating_mode`, applies saved training min/range scaling, restores the saved feature order, and calls `predict_proba`.

The saved preprocessing parameters and feature order are part of the model contract. A replacement model must be paired with matching `preprocessing_stats.json`, `feature_names.csv`, and `risk_state_thresholds.csv` files.

## Model Evaluation

The included evaluation archive reports these Random Forest results on held-out test data, using a binary decision threshold of `0.19`:

| Metric | Reported result |
| --- | ---: |
| Macro F1 | 0.876 |
| Failure recall | 93.44% |
| Failure precision | 68.81% |
| False alarm rate | 6.90% |
| PR-AUC | 0.894 |
| ROC-AUC | 0.979 |
| Accuracy | 93.14% |
| 12-24 hour early-warning recall | 87.82% |
| Median prediction latency | 60.28 ms |

These figures are transcribed from the committed evaluation outputs and have not been independently reproduced in this environment. They are dataset-specific, not a guarantee of real-world performance.

## Project Files

```text
.
├── 01_Source_Code/
│   ├── machinesense_EDA_preprocessing.py
│   ├── MachineSense_Model_Training_(9).ipynb
│   └── MachineSense_LLM_Integration (2).ipynb
├── 02_Data/
│   ├── predictive_maintenance_v3.csv
│   └── MachineSense_Processed_Data_with_Metadata.zip
├── 03_Project_Outputs/
│   └── MachineSense_Final_Project_Files (8).zip
├── backend/
│   ├── main.py
│   ├── llm_service.py
│   ├── model.py
│   ├── predictor.py
│   ├── preprocessing.py
│   └── static/
│       ├── index.html
│       └── models/robot_arm.glb
├── models/
│   ├── FINAL_random_forest_model.joblib
│   ├── feature_names.csv
│   ├── preprocessing_stats.json
│   └── risk_state_thresholds.csv
├── MachineSense_Final_Result.csv
├── Dockerfile
├── requirements.txt
├── .dockerignore
└── .gitignore
```

- `backend/main.py` defines the API, four in-memory machine records, simulation behavior, and static dashboard route.
- `backend/preprocessing.py`, `backend/predictor.py`, and `backend/model.py` prepare model inputs, load the saved classifier, and classify output probabilities.
- `backend/llm_service.py` calls Ollama when available and supplies a deterministic fallback otherwise.
- `backend/static/index.html` contains the dashboard UI and browser-side 3D scene. It loads Tailwind CSS and Three.js from public CDNs, so those assets require network access in the browser.
- `backend/static/models/robot_arm.glb` is a valid glTF binary asset; the current dashboard builds its robot visualization from Three.js geometry rather than loading this file.
- `models/` contains the classifier, the ordered feature list, training-time preprocessing statistics, and risk thresholds. The service expects these files to stay together.
- `02_Data/predictive_maintenance_v3.csv` contains 24,042 sensor records and labels. `MachineSense_Final_Result.csv` contains 4,438 rows of result data.
- `02_Data/MachineSense_Processed_Data_with_Metadata.zip` contains preprocessed train, validation, and test splits with metadata.
- `03_Project_Outputs/MachineSense_Final_Project_Files (8).zip` contains model-selection results, evaluation tables, predictions, figures, and additional model artifacts. Its included `README.txt` summarizes the archive.
- The training notebook compares logistic regression, decision tree, random forest, and gradient boosting models. The EDA/preprocessing script reads the source CSV; the LLM notebook is an Ollama/LangChain experiment separate from the dashboard service.
- `.dockerignore` omits notebooks, source data, and output archives from the container build. `.gitignore` omits local environments, caches, secrets, and editor files.

## Model Input

The prediction API requires these fields: `machine_id`, `vibration_rms`, `temperature_motor`, `current_phase_avg`, `pressure_level`, `rpm`, `hours_since_maintenance`, `ambient_temp`, `machine_type`, and `operating_mode`. `timestamp` is accepted but is not used for model inference.

The numeric features are normalized using the training statistics in `models/preprocessing_stats.json`; categorical values are one-hot encoded to the feature order in that file. Keep the model, preprocessing statistics, feature list, and threshold CSV aligned when replacing model artifacts. Only load model files from a trusted source because joblib model files use Python serialization.

## Prototype Limitations

- Machine records and simulated sensor changes exist only in process memory. They reset when the service restarts and are not synchronized across multiple workers.
- The service has no authentication or persistent storage, and its CORS configuration is permissive. Do not expose it to an untrusted network without adding suitable access controls and deployment configuration.
- The dashboard uses sample values and generated alert/activity content; it is not connected to industrial equipment.
- `machinesense_EDA_preprocessing.py` references `processed_folder` in its final export block without defining it. That block must be corrected before running the script end-to-end.
- The dashboard's 3D libraries are loaded from CDNs, and the included robot-arm GLB is not currently used by the page.

## Data and Model Use

The bundled dataset, serialized model, and archived evaluation outputs are provided as project artifacts. Review their provenance and applicable permissions before using them outside this project. Model probabilities and recommendations are decision-support outputs and should not replace qualified engineering review or established plant safety procedures.

## Technologies

- Python 3.13, FastAPI, Pydantic, and Uvicorn for the API and static dashboard delivery.
- pandas, scikit-learn, and joblib for preprocessing and Random Forest inference.
- Ollama for optional local LLM recommendations.
- HTML, Tailwind CSS, JavaScript, and Three.js for the browser dashboard and 3D scene.
- Docker for containerized deployment.

## Future Improvements

- Connect validated sensor ingestion and persistent, timestamped machine state.
- Add authentication, explicit CORS allowlists, and deployment-specific secret management.
- Add automated API/model tests and a repeatable evaluation pipeline.
- Version and validate model/preprocessing artifacts together; monitor calibration and drift.
- Load the included robot-arm GLB in the dashboard and vendor or pin frontend CDN assets.

## Author / Team

MachineSense Digital Twin is maintained by [Shatha Alharbi](https://github.com/ShathaKhAlharbi).
