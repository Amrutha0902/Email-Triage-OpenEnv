import uvicorn
from env.environment import app

def main():
    """Mandatory main function for the OpenEnv validator"""
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
