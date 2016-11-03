#!/usr/bin/env bash

echo "Test global UniFlex controller in mininet:"

python2 ./mininet_script.py

if [ "$?" != "0" ]; then
  echo "UniFlex mininet failed !!!!"
fi

echo "cleaning up ..."
sudo mn -c 2>/dev/null
