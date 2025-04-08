import uvicorn
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    # Start the FastAPI server on port 8000 to match the frontend proxy configuration
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)