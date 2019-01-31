#!/bin/bash
# kill all processes with same pattern
# Usage: ./batchkill.sh python balabala...

pattern=$@
pids=`ps aux |grep "$pattern" | awk '{print $2}'`
for pid in $pids
do
    kill -9 $pid
done
