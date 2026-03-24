#!/usr/bin/env bash
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
cd "$(dirname "$0")"
exec .venv/bin/uvicorn dashboard:app --host 0.0.0.0 --port "${PORT:-8000}"
