#!/bin/sh
source .venv/bin/activate
export FLASK_APP=main.py
export FLASK_DEBUG=1
# Use PORT if it's set, otherwise default to 8080
flask run --host=0.0.0.0 --port=${PORT:-8080}
