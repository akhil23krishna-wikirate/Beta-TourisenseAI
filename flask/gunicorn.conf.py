# gunicorn.conf.py — Production server configuration
# Start with: gunicorn -c gunicorn.conf.py app:app

import multiprocessing

# ── Workers ───────────────────────────────────────────────
# 2-4 workers is enough; models are loaded once per process.
# Keep low to avoid multiplying RAM usage (each worker loads all PKL models).
workers     = 2
worker_class = "sync"
timeout      = 120          # allow up to 2 min for a full-day prediction
keepalive    = 5

# ── Binding ───────────────────────────────────────────────
bind = "0.0.0.0:5000"

# ── Logging ───────────────────────────────────────────────
loglevel     = "info"
accesslog    = "-"          # stdout
errorlog     = "-"          # stderr
capture_output = True

# ── Preload ───────────────────────────────────────────────
# Loads app (and all PKL models) before forking workers.
# All workers share the same memory pages → saves RAM.
preload_app = True
