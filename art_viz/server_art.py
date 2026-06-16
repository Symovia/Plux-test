"""Art-viz server for jie's 5-track data. Port 8001.

Standalone from the acquisition server (port 8000). Loads pre-computed
per-second tracks (tracks_*.csv) and serves them as JSON for browser viz.
"""
import asyncio
import sys
from pathlib import Path

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

TRACKS_DIR = Path(__file__).resolve().parent.parent / "tracks"
STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI()


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/conditions")
async def list_conditions():
    """Return available conditions (one per tracks_*.csv)."""
    files = sorted(TRACKS_DIR.glob("tracks_*.csv"))
    # Exclude combined
    conds = [f.stem.replace("tracks_", "")
             for f in files if "combined" not in f.stem]
    return {"conditions": conds}


@app.get("/api/tracks/{cond}")
async def get_tracks(cond: str):
    p = TRACKS_DIR / f"tracks_{cond}.csv"
    if not p.exists():
        return JSONResponse({"error": f"no such condition: {cond}"},
                            status_code=404)
    df = pd.read_csv(p)
    # Compact JSON: array of objects
    return JSONResponse({
        "condition": cond,
        "n": len(df),
        "data": df.to_dict(orient="records"),
    })


app.mount("/static", StaticFiles(directory=STATIC), name="static")


if __name__ == "__main__":
    import uvicorn
    print("============================================")
    print(" Art Viz Server")
    print(" Open: http://127.0.0.1:8001")
    print("============================================")
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="warning")
