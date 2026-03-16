#!/usr/bin/env bash
# setup_stockfish.sh — Download a pre-compiled Stockfish binary to bin/stockfish
#
# Run this once from the project root before using the optimizer:
#   bash scripts/setup_stockfish.sh
#
# The binary is saved to bin/stockfish (or bin/stockfish.exe on Windows).
# bin/stockfish is in .gitignore — it won't be committed.
#
# Supported platforms:
#   macOS  (Apple Silicon arm64, Intel x86_64)
#   Linux  (x86_64, aarch64/ARM64 — covers AWS Graviton)
#   Windows (x86_64)
#
# To install a different version, set:
#   SF_VERSION=sf_17.1 bash scripts/setup_stockfish.sh

set -euo pipefail

SF_VERSION="${SF_VERSION:-sf_17.1}"
BIN_DIR="$(cd "$(dirname "$0")/.." && pwd)/bin"
mkdir -p "$BIN_DIR"
DEST="$BIN_DIR/stockfish"

# ── Detect platform ───────────────────────────────────────────────
OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
  Darwin)
    case "$ARCH" in
      arm64)  ASSET="stockfish-macos-m1-apple-silicon" ;;
      x86_64) ASSET="stockfish-macos-x86-64-modern"   ;;
      *)      echo "Unsupported macOS arch: $ARCH" && exit 1 ;;
    esac
    ;;
  Linux)
    case "$ARCH" in
      x86_64)          ASSET="stockfish-ubuntu-x86-64-modern" ;;
      aarch64|arm64)   ASSET="stockfish-ubuntu-aarch64"       ;;
      *)               echo "Unsupported Linux arch: $ARCH" && exit 1 ;;
    esac
    ;;
  MINGW*|CYGWIN*|MSYS*)
    ASSET="stockfish-windows-x86-64-modern"
    DEST="$BIN_DIR/stockfish.exe"
    ;;
  *)
    echo "Unsupported OS: $OS"
    exit 1
    ;;
esac

BASE_URL="https://github.com/official-stockfish/Stockfish/releases/download/${SF_VERSION}"
ARCHIVE=""
URL=""

probe_url() {
  local candidate_url="$1"
  if command -v curl &>/dev/null; then
    curl -fsIL "$candidate_url" >/dev/null
  else
    wget --spider -q "$candidate_url"
  fi
}

for ext in tar zip; do
  candidate_archive="${ASSET}.${ext}"
  candidate_url="${BASE_URL}/${candidate_archive}"
  if probe_url "$candidate_url"; then
    ARCHIVE="$candidate_archive"
    URL="$candidate_url"
    break
  fi
done

if [ -z "$ARCHIVE" ]; then
  echo "Error: could not find a downloadable asset for ${ASSET} under ${BASE_URL}."
  exit 1
fi

# ── Download and extract ──────────────────────────────────────────
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

echo "Downloading Stockfish ${SF_VERSION} for ${OS}/${ARCH}..."
echo "  URL: $URL"

if command -v curl &>/dev/null; then
  curl -fL --progress-bar -o "$TMP_DIR/$ARCHIVE" "$URL"
elif command -v wget &>/dev/null; then
  wget -q --show-progress -O "$TMP_DIR/$ARCHIVE" "$URL"
else
  echo "Error: neither curl nor wget found."
  echo "  Amazon Linux / RHEL:  sudo yum install -y curl"
  echo "  Debian / Ubuntu:      sudo apt-get install -y curl"
  exit 1
fi

echo "Extracting..."
mkdir -p "$TMP_DIR/extracted"
case "$ARCHIVE" in
  *.zip)
    if ! command -v unzip &>/dev/null; then
      echo "Error: unzip not found."
      echo "  Amazon Linux / RHEL:  sudo yum install -y unzip"
      echo "  Debian / Ubuntu:      sudo apt-get install -y unzip"
      exit 1
    fi
    unzip -q "$TMP_DIR/$ARCHIVE" -d "$TMP_DIR/extracted"
    ;;
  *.tar)
    tar -xf "$TMP_DIR/$ARCHIVE" -C "$TMP_DIR/extracted"
    ;;
  *)
    echo "Error: unsupported archive format: $ARCHIVE"
    exit 1
    ;;
esac

# Releases may contain either a plain `stockfish` binary or a
# platform-specific executable like `stockfish-macos-m1-apple-silicon`.
# Find the first executable that looks like the engine binary.
SF_BIN="$(find "$TMP_DIR/extracted" -type f \( -name "stockfish" -o -name "stockfish.exe" -o -name "stockfish-*" \) -perm -111 | head -1)"
if [ -z "$SF_BIN" ]; then
  echo "Error: could not find stockfish binary in downloaded archive."
  ls -R "$TMP_DIR/extracted"
  exit 1
fi

cp "$SF_BIN" "$DEST"
chmod +x "$DEST"

# ── Verify ────────────────────────────────────────────────────────
echo ""
echo "Installed: $DEST"
echo ""
VERSION_LINE="$("$DEST" <<< 'uci' 2>/dev/null | grep '^id name' | head -1 || true)"
if [ -n "$VERSION_LINE" ]; then
  echo "Verified: $VERSION_LINE"
else
  echo "Binary installed but version check skipped (may need execution permission)."
fi

echo ""
echo "Done. Run the optimizer with:"
echo "  julia --threads auto src/optimize.jl 5 32 100"
echo ""
echo "Or with explicit path:"
echo "  julia --threads auto src/optimize.jl 5 32 100 --stockfish $DEST"
