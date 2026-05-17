#!/bin/sh
set -eu
while true; do
  '/home/belki/.hermes/node/bin/node' '/mnt/c/Users/belki/AppData/Roaming/npm/node_modules/agent-workgraph/bin/workgraph.js' 'watch' '/mnt/c/Users/belki/Desktop/Experiment/Javis' '--source' 'claude' '--session' 'latest' '--provider' 'local' '--once'
  sleep 300
done
