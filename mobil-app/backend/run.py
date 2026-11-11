"""
Run the Alzheimer Classification API server.

Usage (with yolo-env):
    C:\\Users\\berkk\\Desktop\\yolo\\yolo-env\\Scripts\\python.exe run.py
    C:\\Users\\berkk\\Desktop\\yolo\\yolo-env\\Scripts\\python.exe run.py --host 0.0.0.0 --port 8000

Or activate the environment first:
    C:\\Users\\berkk\\Desktop\\yolo\\yolo-env\\Scripts\\activate.bat
    python run.py
"""
import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Alzheimer Classification API")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on changes")
    args = parser.parse_args()

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
