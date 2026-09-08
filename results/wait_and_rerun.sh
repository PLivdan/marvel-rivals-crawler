#!/bin/bash
# Wait until the crawl reaches 500k matches, then refit on the larger sample.
cd /Users/dlivdan/projects/marvel-rivals-crawler
while true; do
  n=$(python3 -c "
import sqlite3
c=sqlite3.connect('file:data/rivals.db?mode=ro',uri=True,timeout=120)
print(c.execute('SELECT COUNT(*) FROM matches').fetchone()[0])" 2>/dev/null)
  [ -n "$n" ] && [ "$n" -ge 500000 ] && break
  sleep 300
done
echo "reached $n matches at $(date '+%H:%M'); refitting"
python3 results/run_apm.py --attribution W1 --tag W1_500k
