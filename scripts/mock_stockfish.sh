#!/usr/bin/env bash
set -euo pipefail

# Minimal UCI-compatible mock engine for local optimizer smoke tests.
# It acknowledges setup commands and always returns "bestmove 0000", which
# the optimizer treats as "no move / draw" for fail-fast integration testing.

while IFS= read -r line; do
  case "$line" in
    uci)
      echo "id name MockStockfish"
      echo "id author FieldEngine"
      echo "uciok"
      ;;
    isready)
      echo "readyok"
      ;;
    ucinewgame)
      ;;
    position\ fen\ *)
      ;;
    go\ movetime\ *)
      # Small sleep simulates compute time and exercises timeout handling.
      sleep 0.02
      echo "bestmove 0000"
      ;;
    quit)
      exit 0
      ;;
    stop)
      ;;
    setoption\ name\ *)
      ;;
    *)
      ;;
  esac
done
