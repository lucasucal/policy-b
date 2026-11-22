import json
import threading
import traceback

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from mypolicy.model import Controller

app = FastAPI()
state = {"model": None, "error": None}


def _load():
    try:
        state["model"] = Controller()
        print("model loaded", flush=True)
    except Exception:
        traceback.print_exc()
        state["error"] = "load failed"


@app.get("/healthz")
def healthz():
    return {}


@app.get("/readyz")
def readyz():
    body = {"ready": state["model"] is not None, "contract": "v0", "schema_version": 2,
            "capabilities": {"chunking": False, "caching": False}}
    if state["error"]:
        body["error"] = state["error"]
    return body


@app.post("/act")
async def act(request: Request):
    if state["model"] is None:
        return JSONResponse({"error": "not ready"}, status_code=503)
    obs = json.loads(await request.body())
    a = state["model"].act(obs)
    eid, step = obs["episode_id"], obs["step"]
    return JSONResponse({"episode_id": eid, "step": step, "action": a},
                        headers={"X-RC-Episode-Id": str(eid), "X-RC-Step": str(step)})


if __name__ == "__main__":
    threading.Thread(target=_load, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=8000)
