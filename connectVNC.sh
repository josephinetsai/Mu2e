#!/usr/bin/env bash
    kill -9 `lsof -i -n | egrep '\<ssh\>' | grep 59<n> |  awk '{print $2}'`
    ssh -XKA -v -f -X -N -L 59<n>:localhost:59<n> <username>@mu2egpvm<m>.fnal.gov

# <n> number used in start
# <m> numer of machine used
# <username> your mu2e username
