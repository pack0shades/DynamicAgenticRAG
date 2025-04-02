#!/bin/bash
# Configure guardrails at runtime
guardrails configure --enable-metrics --enable-remote-inferencing --token "$GUARDRAILS_API_KEY"

# Start the application
uvicorn fast_api_server:app --port 8001 --host 0.0.0.0
