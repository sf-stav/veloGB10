#!/bin/sh
# veloGB10 one-line installer — downloads the latest release of the veloGB10
# inference engine for NVIDIA GB10 (DGX Spark) and installs the single binary
# into the user's bin directory (~/.local/bin).
#
# Install:
#   curl -fsSL https://raw.githubusercontent.com/sf-stav/veloGB10/main/install.sh | sh
#
# Requirements: an aarch64 (ARM) Linux box with an NVIDIA GB10 GPU (DGX Spark),
# curl or wget, tar, sha256sum. Nothing is built; no root needed.

set -eu

REPO="sf-stav/veloGB10"
BIN="gb10_inference"
DEST="${HOME}/.local/bin"

# ---------- ANSI colors (disabled when not a TTY) ----------
if [ -t 1 ]; then
  C_B="\033[1m"; C_G="\033[32m"; C_Y="\033[33m"; C_R="\033[31m"; C_C="\033[36m"; C_D="\033[2m"; C_0="\033[0m"
else
  C_B=""; C_G=""; C_Y=""; C_R=""; C_C=""; C_D=""; C_0=""
fi

say()  { printf "%b\n" "$*"; }
err()  { printf "%b\n" "${C_R}error:${C_0} $*" >&2; exit 1; }
step() { printf "%b\n" "${C_C}::${C_0} $*"; }

say ""
say "${C_B}veloGB10 installer${C_0} ${C_D}(NVIDIA GB10 / DGX Spark inference engine)${C_0}"
say ""

# ---------- platform guard ----------
case "$(uname -m)" in
  aarch64|arm64) ;;
  *) err "this is an aarch64 (ARM) build. $(uname -m) detected. veloGB10 runs on NVIDIA GB10 (DGX Spark) boxes only." ;;
esac

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | grep -qi 'GB10'; then
  GPU_OK=1
else
  say "${C_Y}note:${C_0} could not confirm an NVIDIA GB10 GPU on this box (nvidia-smi missing or no GB10 found)."
  say "      veloGB10 runs ${C_B}only${C_0} on GB10 (DGX Spark) hardware."
  printf "%b" "continue anyway? [y/N] "
  read -r ANSWER
  case "$ANSWER" in y|Y|yes|YES) ;; *) say "aborted (nothing was installed)"; exit 1 ;; esac
fi

# ---------- tools ----------
FETCH=""
if command -v curl >/dev/null 2>&1; then FETCH="curl -fsSL"
elif command -v wget >/dev/null 2>&1; then FETCH="wget -qO-"
else err "need curl or wget to download the release."; fi
command -v tar >/dev/null 2>&1 || err "need tar to unpack the release."
command -v sha256sum >/dev/null 2>&1 || err "need sha256sum to verify the release."

# ---------- latest release ----------
step "looking up the latest release…"
TAG=$( $FETCH "https://api.github.com/repos/${REPO}/releases/latest" \
       | sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p' | head -1 )
[ -n "$TAG" ] || err "could not determine the latest release (network or GitHub API limit? try again shortly)."
say "  latest release: ${C_B}${TAG}${C_0}"

# ---------- download + verify ----------
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
TARBALL="velogb10-${TAG}-gb10-sm121.tar.gz"
URL="https://github.com/${REPO}/releases/download/${TAG}/${TARBALL}"

step "downloading ${TARBALL}…"
$FETCH -o "$TMP/$TARBALL" "$URL" || err "download failed: $URL"
$FETCH -o "$TMP/SHA256SUMS" "https://github.com/${REPO}/releases/download/${TAG}/SHA256SUMS" \
  || say "${C_Y}note:${C_0} no SHA256SUMS asset published; skipping integrity check."

if [ -s "$TMP/SHA256SUMS" ]; then
  step "verifying sha256…"
  grep "$TARBALL" "$TMP/SHA256SUMS" | grep -q . || err "checksum line for ${TARBALL} not found in SHA256SUMS."
  ( cd "$TMP" && echo "$( grep "$TARBALL" SHA256SUMS | awk '{print $1}' )  $TARBALL" | sha256sum -c - >/dev/null ) \
    || err "sha256 mismatch — the download is corrupt. Re-run the installer."
  say "  ${C_G}checksum ok${C_0}"
fi

# ---------- install ----------
step "installing to ${DEST}…"
mkdir -p "$DEST"
tar -xzf "$TMP/$TARBALL" -C "$TMP"
BINPATH=$( find "$TMP" -type f -name "$BIN" | head -1 )
[ -n "$BINPATH" ] || err "the tarball does not contain a '${BIN}' binary — please report this."
install -m 0755 "$BINPATH" "$DEST/$BIN"

case ":$PATH:" in
  *":$DEST:"*) PATH_OK=1 ;;
  *) PATH_OK=0 ;;
esac

# ---------- done ----------
VERSION=$( "$DEST/$BIN" --version 2>/dev/null | head -1 || echo "$TAG" )
say ""
say "${C_G}${C_B}installed:${C_0} ${DEST}/${BIN}  ${C_D}($VERSION)${C_0}"
[ "$PATH_OK" = 1 ] || say "${C_Y}note:${C_0} ${DEST} is not on your PATH. Add it with:"
say "      ${C_C}echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.bashrc && source ~/.bashrc${C_0}"
say ""
say "${C_B}Quick start${C_0} ${C_D}(TP=1, the Qwen3.5-0.8B tester model)${C_0}:"
say ""
say "  ${C_C}gb10_inference --server --model-dir Qwen/Qwen3.5-0.8B --port 9000${C_0}"
say ""
say "  ${C_D}--model-dir takes a local directory or a Hugging Face repo id (a local snapshot${C_0}"
say "  ${C_D}is resolved from ~/.cache/huggingface/hub; download first: huggingface-cli download <id>).${C_0}"
say ""
say "${C_B}Multi-box (tensor parallel) users:${C_0} install ${C_B}the same version${C_0} on ${C_B}every node${C_0} —"
say "a head and a node from different builds refuse to start together (by design)."
say ""
