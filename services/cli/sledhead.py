import typer
from sledhead_imu.config import MODEL_READY
app = typer.Typer(help="CLI for Sled-Head IMU pipeline")

@app.command()
def check():
    print(f"Model-ready path: {MODEL_READY}")

if __name__ == "__main__":
    app()
