import os
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.llm_service import generate_maintenance_recommendation
from backend.predictor import predict_record

app = FastAPI(
    title="MachineSense Digital Twin Backend",
    description="Real-time telemetry, Random Forest predictive maintenance, and LLM diagnostics for factory machines.",
    version="2.0.0",
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"


# -------------------------------------------------------------------------
# In-Memory Fleet State (Pre-populated from MachineSense Dataset)
# -------------------------------------------------------------------------
MACHINES_STORE: Dict[str, dict] = {
    "4": {
        "machine_id": "4",
        "alias": "CNC-01",
        "legacy_alias": "CNC-04",
        "name": "CNC Machining Center #01",
        "machine_type": "CNC",
        "operating_mode": "normal",
        "vibration_rms": 0.82,
        "temperature_motor": 48.6,
        "current_phase_avg": 8.45,
        "pressure_level": 52.1,
        "rpm": 2021.4,
        "cycle_rate": 65.0,
        "target_cycle_rate": 70.0,
        "hours_since_maintenance": 81.9,
        "ambient_temp": 14.8,
        "serial_no": "#AL213-CNC01",
        "station": "Station 01 — Milling & Shaping",
        "line_id": "213",
        "uptime": 99.1,
        "active_cycle_min": 65,
        "energy_consumption_kwh": 480.0,
        "units_processed_today": 890,
    },
    "6": {
        "machine_id": "6",
        "alias": "Pump-02",
        "legacy_alias": "PMP-06",
        "name": "Centrifugal Process Pump #02",
        "machine_type": "Pump",
        "operating_mode": "normal",
        "vibration_rms": 0.78,
        "temperature_motor": 43.5,
        "current_phase_avg": 5.40,
        "pressure_level": 69.5,
        "rpm": 980.0,
        "cycle_rate": 45.0,
        "target_cycle_rate": 50.0,
        "hours_since_maintenance": 141.0,
        "ambient_temp": 14.2,
        "serial_no": "#AL213-PMP02",
        "station": "Station 02 — Coolant Circulation",
        "line_id": "213",
        "uptime": 99.5,
        "active_cycle_min": 45,
        "energy_consumption_kwh": 180.0,
        "units_processed_today": 2100,
    },
    "12": {
        "machine_id": "12",
        "alias": "Compressor-03",
        "legacy_alias": "CMP-12",
        "name": "Industrial Air Compressor #03",
        "machine_type": "Compressor",
        "operating_mode": "normal",
        "vibration_rms": 1.45,
        "temperature_motor": 54.2,
        "current_phase_avg": 9.80,
        "pressure_level": 80.4,
        "rpm": 1288.0,
        "cycle_rate": 110.0,
        "target_cycle_rate": 120.0,
        "hours_since_maintenance": 156.1,
        "ambient_temp": 14.5,
        "serial_no": "#AL213-CMP03",
        "station": "Station 03 — Pneumatic Utility",
        "line_id": "213",
        "uptime": 97.8,
        "active_cycle_min": 110,
        "energy_consumption_kwh": 620.0,
        "units_processed_today": 1750,
    },
    "20": {
        "machine_id": "20",
        "alias": "Robot-04",
        "legacy_alias": "MCH-AX01",
        "name": "6-Axis Industrial Robotic Arm #04",
        "machine_type": "Robotic Arm",
        "operating_mode": "peak",
        "vibration_rms": 1.84,
        "temperature_motor": 62.4,
        "current_phase_avg": 6.85,
        "pressure_level": 32.5,
        "rpm": 866.5,
        "cycle_rate": 84.0,
        "target_cycle_rate": 95.0,
        "hours_since_maintenance": 140.7,
        "ambient_temp": 15.2,
        "serial_no": "#AL213-ROB04",
        "station": "Station 04 — Welding & Assembly",
        "line_id": "213",
        "uptime": 98.4,
        "active_cycle_min": 84,
        "energy_consumption_kwh": 350.0,
        "units_processed_today": 1248,
    },
}



# -------------------------------------------------------------------------
# Request Models
# -------------------------------------------------------------------------
class PredictRequest(BaseModel):
    machine_id: str
    timestamp: Optional[str] = None
    vibration_rms: float
    temperature_motor: float
    current_phase_avg: float
    pressure_level: float
    rpm: float
    hours_since_maintenance: float
    ambient_temp: float
    machine_type: str
    operating_mode: str


class SimulateModeRequest(BaseModel):
    operating_mode: str  # "idle" | "normal" | "peak"
    custom_temp: Optional[float] = None
    custom_vibration: Optional[float] = None
    custom_rpm: Optional[float] = None


class CopilotChatRequest(BaseModel):
    query: str
    machine_id: Optional[str] = "20"


# -------------------------------------------------------------------------
# Health & Original Predict Endpoints
# -------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": "2.0.0"}


@app.post("/predict")
def predict(payload: PredictRequest) -> dict:
    ml_result = predict_record(
        {
            "vibration_rms": payload.vibration_rms,
            "temperature_motor": payload.temperature_motor,
            "current_phase_avg": payload.current_phase_avg,
            "pressure_level": payload.pressure_level,
            "rpm": payload.rpm,
            "hours_since_maintenance": payload.hours_since_maintenance,
            "ambient_temp": payload.ambient_temp,
            "machine_type": payload.machine_type,
            "operating_mode": payload.operating_mode,
        }
    )

    llm_result = generate_maintenance_recommendation(
        machine_id=payload.machine_id,
        status=ml_result["risk_level"],
        temperature=payload.temperature_motor,
        speed=payload.rpm,
        torque=payload.current_phase_avg,
    )

    return {
        "machine_id": payload.machine_id,
        "failure_probability": ml_result["failure_probability"],
        "risk_level": ml_result["risk_level"],
        "llm": llm_result,
    }


# -------------------------------------------------------------------------
# Digital Twin Fleet & Line API Endpoints
# -------------------------------------------------------------------------
def _get_evaluated_machine(machine_data: dict) -> dict:
    """Passes current machine telemetry through the ML model to get real-time risk."""
    ml_res = predict_record(
        {
            "vibration_rms": machine_data["vibration_rms"],
            "temperature_motor": machine_data["temperature_motor"],
            "current_phase_avg": machine_data["current_phase_avg"],
            "pressure_level": machine_data["pressure_level"],
            "rpm": machine_data["rpm"],
            "hours_since_maintenance": machine_data["hours_since_maintenance"],
            "ambient_temp": machine_data["ambient_temp"],
            "machine_type": machine_data["machine_type"],
            "operating_mode": machine_data["operating_mode"],
        }
    )
    result = dict(machine_data)
    result["failure_probability"] = ml_res["failure_probability"]
    result["risk_level"] = ml_res["risk_level"]
    # Overall performance score (OEE %) inversely related to risk & load
    performance_score = max(40.0, min(99.0, round((1.0 - ml_res["failure_probability"] * 0.6) * 100, 1)))
    result["performance_score"] = performance_score
    return result


@app.get("/api/fleet")
def get_fleet() -> List[dict]:
    """Returns all machines in the factory line with live evaluated ML health."""
    return [_get_evaluated_machine(m) for m in MACHINES_STORE.values()]


@app.get("/api/machines/{machine_id}")
def get_machine(machine_id: str) -> dict:
    """Returns single machine digital twin telemetry and state."""
    # Allow lookup by id, alias, or legacy_alias
    machine = MACHINES_STORE.get(machine_id)
    if not machine:
        for m in MACHINES_STORE.values():
            if m["alias"].lower() == machine_id.lower() or m.get("legacy_alias", "").lower() == machine_id.lower():
                machine = m
                break
    if not machine:
        raise HTTPException(status_code=404, detail="Machine not found")

    return _get_evaluated_machine(machine)


@app.post("/api/machines/{machine_id}/simulate")
def simulate_machine(machine_id: str, payload: SimulateModeRequest) -> dict:
    """Simulates real-time mode transitions and telemetry variations."""
    machine = MACHINES_STORE.get(machine_id)
    if not machine:
        for m in MACHINES_STORE.values():
            if m["alias"].lower() == machine_id.lower() or m.get("legacy_alias", "").lower() == machine_id.lower():
                machine = m
                break
    if not machine:
        raise HTTPException(status_code=404, detail="Machine not found")

    mode = payload.operating_mode.lower()
    machine["operating_mode"] = mode
    m_type = machine.get("machine_type", "")

    if mode == "idle":
        if m_type == "CNC":
            machine["vibration_rms"] = payload.custom_vibration or 0.48
            machine["temperature_motor"] = payload.custom_temp or 39.5
            machine["current_phase_avg"] = 3.80
            machine["rpm"] = payload.custom_rpm or 600.0
            machine["cycle_rate"] = 18.0
            machine["pressure_level"] = 38.0
        elif m_type == "Pump":
            machine["vibration_rms"] = payload.custom_vibration or 0.40
            machine["temperature_motor"] = payload.custom_temp or 36.2
            machine["current_phase_avg"] = 2.40
            machine["rpm"] = payload.custom_rpm or 420.0
            machine["cycle_rate"] = 15.0
            machine["pressure_level"] = 45.0
        elif m_type == "Compressor":
            machine["vibration_rms"] = payload.custom_vibration or 0.55
            machine["temperature_motor"] = payload.custom_temp or 42.0
            machine["current_phase_avg"] = 3.50
            machine["rpm"] = payload.custom_rpm or 550.0
            machine["cycle_rate"] = 30.0
            machine["pressure_level"] = 62.0
        else: # Robotic Arm
            machine["vibration_rms"] = payload.custom_vibration or 0.35
            machine["temperature_motor"] = payload.custom_temp or 36.0
            machine["current_phase_avg"] = 2.10
            machine["rpm"] = payload.custom_rpm or 120.0
            machine["cycle_rate"] = 14.0
            machine["pressure_level"] = 22.0
    elif mode == "normal":
        if m_type == "CNC":
            machine["vibration_rms"] = payload.custom_vibration or 0.82
            machine["temperature_motor"] = payload.custom_temp or 48.6
            machine["current_phase_avg"] = 8.45
            machine["rpm"] = payload.custom_rpm or 2021.4
            machine["cycle_rate"] = 65.0
            machine["pressure_level"] = 52.1
        elif m_type == "Pump":
            machine["vibration_rms"] = payload.custom_vibration or 0.78
            machine["temperature_motor"] = payload.custom_temp or 43.5
            machine["current_phase_avg"] = 5.40
            machine["rpm"] = payload.custom_rpm or 980.0
            machine["cycle_rate"] = 45.0
            machine["pressure_level"] = 69.5
        elif m_type == "Compressor":
            machine["vibration_rms"] = payload.custom_vibration or 1.45
            machine["temperature_motor"] = payload.custom_temp or 54.2
            machine["current_phase_avg"] = 9.80
            machine["rpm"] = payload.custom_rpm or 1288.0
            machine["cycle_rate"] = 110.0
            machine["pressure_level"] = 80.4
        else: # Robotic Arm
            machine["vibration_rms"] = payload.custom_vibration or 0.88
            machine["temperature_motor"] = payload.custom_temp or 49.2
            machine["current_phase_avg"] = 5.20
            machine["rpm"] = payload.custom_rpm or 450.0
            machine["cycle_rate"] = 58.0
            machine["pressure_level"] = 32.5
    elif mode == "peak":
        if m_type == "CNC":
            machine["vibration_rms"] = payload.custom_vibration or 2.15
            machine["temperature_motor"] = payload.custom_temp or 68.5
            machine["current_phase_avg"] = 14.20
            machine["rpm"] = payload.custom_rpm or 3100.0
            machine["cycle_rate"] = 88.0
            machine["pressure_level"] = 65.0
        elif m_type == "Pump":
            machine["vibration_rms"] = payload.custom_vibration or 1.90
            machine["temperature_motor"] = payload.custom_temp or 62.0
            machine["current_phase_avg"] = 9.80
            machine["rpm"] = payload.custom_rpm or 1480.0
            machine["cycle_rate"] = 62.0
            machine["pressure_level"] = 88.0
        elif m_type == "Compressor":
            machine["vibration_rms"] = payload.custom_vibration or 2.35
            machine["temperature_motor"] = payload.custom_temp or 74.0
            machine["current_phase_avg"] = 16.10
            machine["rpm"] = payload.custom_rpm or 1850.0
            machine["cycle_rate"] = 135.0
            machine["pressure_level"] = 98.0
        else: # Robotic Arm
            machine["vibration_rms"] = payload.custom_vibration or 1.84
            machine["temperature_motor"] = payload.custom_temp or 62.4
            machine["current_phase_avg"] = 6.85
            machine["rpm"] = payload.custom_rpm or 866.5
            machine["cycle_rate"] = 84.0
            machine["pressure_level"] = 42.0

    return _get_evaluated_machine(machine)



@app.get("/api/line/{line_id}")
def get_line_status(line_id: str = "213") -> dict:
    """Returns assembly line #213 status, aggregate efficiency, and active alerts."""
    fleet = [_get_evaluated_machine(m) for m in MACHINES_STORE.values() if m.get("line_id") == line_id]
    
    avg_efficiency = round(sum(m["performance_score"] for m in fleet) / max(len(fleet), 1), 1)
    high_risks = [m for m in fleet if m["risk_level"] == "High Risk"]
    warnings = [m for m in fleet if m["risk_level"] == "Warning"]

    alerts = []
    for m in high_risks:
        alerts.append({
            "severity": "critical",
            "title": f"High vibration & risk on {m['name']}",
            "description": f"RMS {m['vibration_rms']} mm/s, Motor Temp {m['temperature_motor']}°C. Failure probability: {int(m['failure_probability']*100)}%",
            "timestamp": "Just now",
            "machine_id": m["machine_id"]
        })
    for m in warnings:
        alerts.append({
            "severity": "warning",
            "title": f"Elevated profile on {m['name']}",
            "description": f"Hours since maintenance: {m['hours_since_maintenance']}h (limit 200h).",
            "timestamp": "12m ago",
            "machine_id": m["machine_id"]
        })
    alerts.append({
        "severity": "success",
        "title": "Cooling cycle recovered",
        "description": "Hydraulic Coolant Pump #06 operating within nominal thermal envelope.",
        "timestamp": "42m ago",
        "machine_id": "6"
    })

    return {
        "line_id": line_id,
        "name": f"Assembly Line #{line_id}",
        "station_name": "Weld & Assembly",
        "overall_efficiency": avg_efficiency,
        "risks_solved": 24,
        "failed_to_solve": 12,
        "active_units_count": len(fleet),
        "high_risk_count": len(high_risks),
        "warning_count": len(warnings),
        "alerts": alerts,
        "fleet": fleet,
    }


@app.post("/api/copilot/chat")
def copilot_chat(payload: CopilotChatRequest) -> dict:
    """Interactive industrial LLM maintenance co-pilot."""
    machine = MACHINES_STORE.get(payload.machine_id or "20", MACHINES_STORE["20"])
    evaluated = _get_evaluated_machine(machine)

    llm_res = generate_maintenance_recommendation(
        machine_id=evaluated["machine_id"],
        status=evaluated["risk_level"],
        temperature=evaluated["temperature_motor"],
        speed=evaluated["rpm"],
        torque=evaluated["current_phase_avg"],
    )

    return {
        "query": payload.query,
        "machine_id": evaluated["machine_id"],
        "machine_name": evaluated["name"],
        "risk_level": evaluated["risk_level"],
        "failure_probability": evaluated["failure_probability"],
        "recommendation": llm_res,
    }


# -------------------------------------------------------------------------
# Static Web App Mount
# -------------------------------------------------------------------------
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    @app.get("/dashboard")
    def serve_index():
        return FileResponse(STATIC_DIR / "index.html")
