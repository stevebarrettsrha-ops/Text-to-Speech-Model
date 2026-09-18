#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

find_python() {
  PY=""
  for cand in python3.13 python3.12 python3.11 python3.10 python3 python; do
    if command -v "$cand" >/dev/null 2>&1 && \
       "$cand" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
      PY="$cand"; return 0
    fi
  done
  return 1
}

if ! find_python; then
  echo "  Python 3.10 or newer was not found." >&2

  # python3-venv is a separate package on Debian and Ubuntu, and Script
  # Builder cannot build ComfyUI's environment without it.
  if command -v brew >/dev/null 2>&1; then
    INSTALL=(brew install python)
  elif command -v apt-get >/dev/null 2>&1; then
    INSTALL=(sudo apt-get install -y python3 python3-venv python3-pip)
  elif command -v dnf >/dev/null 2>&1; then
    INSTALL=(sudo dnf install -y python3 python3-pip)
  elif command -v pacman >/dev/null 2>&1; then
    INSTALL=(sudo pacman -S --noconfirm python python-pip)
  else
    echo "  No package manager was found. Install Python 3.10+, then run this again." >&2
    exit 1
  fi

  echo "  This can be done for you with: ${INSTALL[*]}"
  printf "  Install Python now? [y/N] "
  read -r reply </dev/tty || reply=""
  case "$reply" in
    [yY]*) ;;
    *) echo "  Nothing was installed. Install Python 3.10+, then run this again." >&2
       exit 1 ;;
  esac

  if ! "${INSTALL[@]}"; then
    echo "  That did not finish. Install Python 3.10+ by hand, then run this again." >&2
    exit 1
  fi

  if ! find_python; then
    echo "  Python is installed but this shell cannot see it yet." >&2
    echo "  Open a new terminal and run ./run.sh again." >&2
    exit 1
  fi
fi

echo "  Using: $("$PY" -c 'import sys;print(sys.executable)')"

# Script Builder's own packages go into a virtual environment beside this
# script, never into the Python that was found. Debian, Ubuntu and Homebrew all
# mark their Python as externally managed and pip refuses to install into it
# (PEP 668) — which is exactly the Python the offer above installs — and even
# where pip would allow it, putting Flask into the system Python is not our
# business. ComfyUI's own packages are a separate matter: those go into the
# interpreter ComfyUI runs on, which bootstrap.py works out.
VENV=".venv"
if [ -e "$VENV" ] && ! "$VENV/bin/python" -c 'import sys' >/dev/null 2>&1; then
  echo "  The existing environment no longer runs. Building it again."
  rm -rf "$VENV"
fi
if [ ! -x "$VENV/bin/python" ]; then
  echo "  Setting up Script Builder's packages (first run only)..."
  if ! "$PY" -m venv "$VENV"; then
    echo "  Could not create the environment. On Debian and Ubuntu the venv" >&2
    echo "  module ships separately:  sudo apt-get install -y python3-venv" >&2
    echo "  Install it, then run this again." >&2
    exit 1
  fi
fi

"$VENV/bin/python" -m pip install --disable-pip-version-check --quiet \
    -r requirements.txt
exec "$VENV/bin/python" server.py
