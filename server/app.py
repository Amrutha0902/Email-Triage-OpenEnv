import uvicorn
from env.environment import app

def start():
    """Entry point for the [project.scripts] server command"""
    uvicorn.run("env.environment:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    start()
