#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/package_optimize_with_stockfish.sh /path/to/stockfish [output.tar.gz]

This copies the Stockfish binary into bin/stockfish and creates a deployable
archive containing optimize.jl and its required sources.
EOF
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 1
fi

stockfish_src="$1"
out_path="${2:-field_engine_stockfish_bundle.tar.gz}"
root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dest_bin="$root_dir/bin/stockfish"
src_abs="$(cd "$(dirname "$stockfish_src")" && pwd)/$(basename "$stockfish_src")"

if [[ ! -f "$stockfish_src" ]]; then
  echo "error: stockfish binary not found at: $stockfish_src" >&2
  exit 1
fi

mkdir -p "$root_dir/bin"
if [[ "$src_abs" != "$dest_bin" ]]; then
  cp "$stockfish_src" "$dest_bin"
fi
chmod +x "$dest_bin"

tar -czf "$out_path" -C "$root_dir" \
  Project.toml \
  Manifest.toml \
  src/optimize.jl \
  src/state.jl \
  src/fields.jl \
  src/energy.jl \
  bin/stockfish

echo "Created archive: $out_path"
