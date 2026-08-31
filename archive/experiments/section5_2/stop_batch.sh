#!/usr/bin/env bash
# Stop a batch SAFELY, by wrapper pid recorded at launch -- never by matching a
# script name against process command lines. A broad name filter also matches
# the Claude Code wrapper shell that is running the filter, which kills the
# session's own shell and every other batch whose wrapper happens to carry the
# same text. That has happened; do not reintroduce it.
#
# Usage: bash experiments/section5_2/stop_batch.sh logs/ablations3_<timestamp>
set -u
D=${1:?usage: stop_batch.sh <log-dir>}
PIDFILE="$D/batch.pid"
[ -f "$PIDFILE" ] || { echo "no $PIDFILE"; exit 1; }
WRAP=$(cat "$PIDFILE")
echo "wrapper pid $WRAP"
CHILDREN=$(ps -W 2>/dev/null | awk -v p="$WRAP" '$2==p {print $1}')
echo "child pids  ${CHILDREN:-none}"
kill -9 "$WRAP" 2>/dev/null && echo "killed wrapper"
for c in $CHILDREN; do kill -9 "$c" 2>/dev/null && echo "killed child $c"; done
sleep 2
echo "--- still alive under that wrapper ---"
ps -W 2>/dev/null | awk -v p="$WRAP" '$1==p || $2==p'
echo "(empty above = stopped)"
