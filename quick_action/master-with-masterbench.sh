#!/bin/zsh
# Finder Quick Action: "Master with MasterBench".
# Receives the selected audio files as arguments and masters each one against its genre's
# default reference. A notification appears when each finishes.
#
# MASTERBENCH_REPO must point at this repository's checkout.

MASTERBENCH_REPO="${MASTERBENCH_REPO:-$HOME/Developer/mystro}"
export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$PATH"

if [[ $# -gt 1 ]]; then
  uv run --project "$MASTERBENCH_REPO" masterbench batch "$@" --notify
else
  uv run --project "$MASTERBENCH_REPO" masterbench master "$1" --notify
fi
