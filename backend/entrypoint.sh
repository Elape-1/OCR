#!/bin/sh
set -e

# Run migrations only when RUN_MIGRATIONS is set to "true" (defaults to true).
# This allows disabling migrations in development by setting RUN_MIGRATIONS=false.
if [ "${RUN_MIGRATIONS:-true}" = "true" ] && { [ "$1" = "uvicorn" ] || [ -z "$1" ]; }; then
  alembic upgrade head
fi

exec "$@"