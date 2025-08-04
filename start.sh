#!/bin/bash
# start.sh

# Ensure .env variables are loaded if needed
if [ -f ".env" ]; then
  export $(cat .env | xargs)
fi

# Start the FastAPI app
uvicorn main:app --host 0.0.0.0 --port 10000
