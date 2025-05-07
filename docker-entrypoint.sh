#!/bin/sh
set -e

echo "Starting app with profile: $PROFILE"

if [ "$PROFILE" = "dev" ]; then
    exec uv run fastapi dev --host 0.0.0.0 --port 8000
else
    exec uv run fastapi run --host 0.0.0.0 --port 8000
fi