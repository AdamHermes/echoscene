#!/bin/bash
# (Re)start the qualitative viewer in the background: ./run.sh [extra server.py args, e.g. --rebuild]
# Log: server.log next to this script. Stop: ./run.sh stop
cd "$(dirname "$0")"
[ -f server.pid ] && kill "$(cat server.pid)" 2>/dev/null && sleep 1
rm -f server.pid
[ "$1" = stop ] && exit 0
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then source ~/miniconda3/etc/profile.d/conda.sh; conda activate echoscene; fi
nohup nice -n 10 python server.py --config config_lhanubu.json "$@" > server.log 2>&1 < /dev/null &
echo $! > server.pid
echo "started (pid $!), log: $(pwd)/server.log, open http://localhost:8765"
