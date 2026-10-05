import re

try:
    import ollama
except ImportError:
    ollama = None

SYSTEM_PROMPT = """
You are an expert Industrial Predictive Maintenance AI Specialist for 'MachineSense'.

Your job is to analyze the provided machine sensor data and machine health status,
then generate clear, actionable, and concise maintenance recommendations for plant engineers.

GUARDRAILS & RULES:

1. ONLY provide maintenance advice based on:
   - Speed
   - Torque
   - Temperature
   - Failure Status

2. NEVER guess, invent, or hallucinate sensor readings, failure causes,
   machine conditions, or maintenance history that were not provided.

3. NEVER change, override, or contradict the provided Predicted Status.

4. NEVER claim that a machine failure is guaranteed to occur.

5. NEVER recommend bypassing safety systems, disabling alarms,
   ignoring safety procedures, or continuing unsafe operation.

6. If the exact cause of the abnormal condition cannot be determined
   from the provided data, clearly state that the exact cause requires
   further inspection.

7. Keep the response strictly structured into these 3 short sections:
   - Status Summary
   - Immediate Recommended Action
   - Safety Alert

8. Keep the tone highly professional, precise, concise, and practical
   for an industrial maintenance context.

9. Do not mention any specific machine component or failure cause
   unless it is explicitly provided in the input data.

10. Base all recommendations only on the provided temperature, speed,
    torque, and predicted status.
"""


def _call_ollama_model(prompt: str) -> str:
    if ollama is None:
        raise RuntimeError("ollama package is not installed")

    try:
        client = ollama.Client()
    except Exception:
        client = ollama

    if hasattr(client, "interactions") and hasattr(client.interactions, "create"):
        response = client.interactions.create(model="llama3.2", input=prompt)
        return getattr(response, "output_text", str(response))

    if hasattr(client, "chat"):
        response = client.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
        if isinstance(response, dict):
            if "message" in response and isinstance(response["message"], dict):
                content = response["message"].get("content")
                if content:
                    return content
            if "content" in response:
                return response["content"]
        if hasattr(response, "message"):
            return getattr(response.message, "content", str(response))
        return str(response)

    if hasattr(client, "generate"):
        response = client.generate(model="llama3.2", prompt=prompt)
        if isinstance(response, dict):
            if "response" in response:
                return str(response["response"])
        return str(response)

    raise RuntimeError("No supported Ollama client API available")


def _fallback_response(machine_id: str, status: str, temperature: float, speed: float, torque: float) -> str:
    return (
        f"Status Summary: Machine {machine_id} is currently {status}. "
        f"The operating profile indicates elevated thermal and speed conditions with temperature {temperature} °C, "
        f"speed {speed} rpm, and torque {torque} Nm.\n\n"
        f"Immediate Recommended Action: Inspect the machine promptly, verify the current operating state against safe limits, "
        f"and schedule maintenance review before continued operation.\n\n"
        f"Safety Alert: Do not continue unrestricted operation while the machine remains in a {status} state; follow standard safety procedures and lockout/inspection workflows before restarting."
    )


def _parse_sections(text: str) -> dict:
    cleaned = text.strip()
    headings = [
        "Status Summary",
        "Immediate Recommended Action",
        "Safety Alert",
    ]

    values = {}
    for index, heading in enumerate(headings):
        next_heading = headings[index + 1] if index < len(headings) - 1 else None
        if next_heading is None:
            pattern = rf"(?:\*\*|\*)?\s*{re.escape(heading)}\s*(?:\*\*|\*)?\s*[:\-]?\s*(.*)$"
        else:
            pattern = (
                rf"(?:\*\*|\*)?\s*{re.escape(heading)}\s*(?:\*\*|\*)?\s*[:\-]?\s*"
                rf"(.*?)(?=(?:\*\*|\*)?\s*{re.escape(next_heading)}\s*(?:\*\*|\*)?\s*[:\-]?)"
            )

        match = re.search(pattern, cleaned, flags=re.IGNORECASE | re.DOTALL)
        if match:
            value = match.group(1).strip()
            value = re.sub(r"^\s*(?:\*+|-+)\s*", "", value)
            value = re.sub(r"\s*(?:\*+|-+)\s*$", "", value)
            values[heading] = value.strip()

    if values:
        return {
            "status_summary": values.get("Status Summary", "").strip(),
            "recommended_action": values.get("Immediate Recommended Action", "").strip(),
            "safety_alert": values.get("Safety Alert", "").strip(),
        }

    return {
        "status_summary": "The machine is operating under monitored conditions.",
        "recommended_action": "Inspect the machine and follow standard maintenance procedures before resuming normal operation.",
        "safety_alert": "Do not bypass safety protocols or continue operation in an unsafe state.",
    }


def generate_maintenance_recommendation(machine_id: str, status: str, temperature: float, speed: float, torque: float) -> dict:
    user_input = f"""
Machine ID: {machine_id}
Predicted Status: {status}

Current Sensors:
- Temperature: {temperature} °C
- Rotational Speed: {speed} rpm
- Torque: {torque} Nm

Please generate the maintenance recommendation.
"""

    prompt = SYSTEM_PROMPT + "\n\n" + user_input

    try:
        response = _call_ollama_model(prompt)
        parsed = _parse_sections(response)
    except Exception:
        response = _fallback_response(machine_id, status, temperature, speed, torque)
        parsed = _parse_sections(response)

    if not parsed.get("status_summary") or not parsed.get("recommended_action") or not parsed.get("safety_alert"):
        parsed = {
            "status_summary": f"Machine {machine_id} is currently in {status} condition with temperature {temperature} °C, speed {speed} rpm, and torque {torque} Nm.",
            "recommended_action": "Inspect the machine promptly and verify the current operating state against safe limits before continued operation.",
            "safety_alert": f"Do not continue unrestricted operation while the machine remains in a {status} state; follow standard safety procedures and lockout/inspection workflows before restarting.",
        }

    return parsed
