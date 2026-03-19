from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

from src.pipeline.planning_pipeline import PlanningPipeline


app = FastAPI(title="NetPlanner AI")

BASE_DIR = Path(__file__).resolve().parent
MAP_FILE_PATH = BASE_DIR / "tower_upgrade_map.html"
PRIORITY_CSV_PATH = BASE_DIR / "priority_sites.csv"


@app.get("/", response_class=HTMLResponse)
async def welcome():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>NetPlanner AI</title>
        <style>
            body {
                margin: 0;
                min-height: 100vh;
                display: grid;
                place-items: center;
                font-family: Georgia, serif;
                background: linear-gradient(135deg, #f2efe7 0%, #d7e3d4 100%);
                color: #1f2a1f;
            }
            .card {
                width: min(560px, 92vw);
                padding: 40px;
                border-radius: 24px;
                background: rgba(255, 255, 255, 0.9);
                box-shadow: 0 20px 60px rgba(31, 42, 31, 0.15);
                text-align: center;
            }
            h1 {
                margin: 0 0 12px;
                font-size: clamp(2rem, 4vw, 3rem);
            }
            p {
                margin: 0 0 28px;
                line-height: 1.6;
                font-size: 1.05rem;
            }
            .button {
                display: inline-block;
                padding: 14px 24px;
                border-radius: 999px;
                background: #2f5d50;
                color: #ffffff;
                text-decoration: none;
                font-weight: 700;
            }
            .button:hover {
                background: #24483e;
            }
            .actions {
                display: flex;
                gap: 12px;
                justify-content: center;
                flex-wrap: wrap;
            }
            .button.secondary {
                background: #6d7f76;
            }
            .button.secondary:hover {
                background: #566760;
            }
            .button.tertiary {
                background: #9a6b2f;
            }
            .button.tertiary:hover {
                background: #7e5624;
            }
        </style>
    </head>
    <body>
        <section class="card">
            <h1>Welcome to NetPlanner AI</h1>
            <p>
                Open the latest tower upgrade map and review the highest-priority
                network upgrade locations.
            </p>
            <div class="actions">
                <a class="button" href="/tower-upgrade-map">Open Tower Upgrade Map</a>
                <a class="button secondary" href="/generate-tower-upgrade-map">Generate Tower Upgrade Map</a>
                <a class="button tertiary" href="/export-priority-sites">Export Priority Sites CSV</a>
            </div>
        </section>
    </body>
    </html>
    """


@app.get("/generate-tower-upgrade-map")
async def generate_tower_upgrade_map():
    pipeline = PlanningPipeline()
    pipeline.generate_upgrade_map(output_path=MAP_FILE_PATH)
    return RedirectResponse(url="/tower-upgrade-map", status_code=303)


@app.get("/export-priority-sites")
async def export_priority_sites():
    pipeline = PlanningPipeline()
    csv_path = pipeline.generate_priority_sites_csv(output_path=PRIORITY_CSV_PATH)
    return FileResponse(
        path=csv_path,
        media_type="text/csv",
        filename=Path(csv_path).name,
    )


@app.get("/tower-upgrade-map")
async def tower_upgrade_map():
    if not MAP_FILE_PATH.exists():
        return HTMLResponse(
            content=f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8" />
                <meta name="viewport" content="width=device-width, initial-scale=1.0" />
                <title>Map Not Found</title>
            </head>
            <body style="font-family: Georgia, serif; padding: 32px;">
                <h1>Tower upgrade map not found</h1>
                <p>
                    Expected file:
                    <code>{MAP_FILE_PATH}</code>
                </p>
                <p>Run the NetPlanner AI training pipeline first to generate the HTML map.</p>
                <p><a href="/">Back to welcome screen</a></p>
            </body>
            </html>
            """,
            status_code=404,
        )

    return HTMLResponse(MAP_FILE_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
