#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Started from the file manager (right-click → Run as a Program) there is no
# terminal: nothing this script says can be seen, and a failure just looks like
# nothing happening. On a Linux desktop, open a terminal and run in it instead.
# Anything run with SCRIPT_BUILDER_NO_TERMINAL=1 (a service, an IDE task) stays
# where it was started. Not INVOCATION_ID: a file manager that runs as
# a systemd service hands it to everything it starts.
if [ "$(uname)" = "Linux" ] && [ -z "${SCRIPT_BUILDER_IN_TERMINAL:-}" ] \
   && [ ! -t 0 ] && [ ! -t 1 ] && [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ] \
   && [ -z "${SCRIPT_BUILDER_NO_TERMINAL:-}" ]; then
  export SCRIPT_BUILDER_IN_TERMINAL=1
  SELF="$(pwd)/$(basename "$0")"
  for term in gnome-terminal ptyxis kgx konsole xfce4-terminal mate-terminal \
              tilix x-terminal-emulator xterm; do
    command -v "$term" >/dev/null 2>&1 || continue
    case "$term" in
      gnome-terminal|ptyxis|kgx) "$term" -- bash "$SELF" && exit 0 ;;
      xfce4-terminal|mate-terminal) "$term" -x bash "$SELF" && exit 0 ;;
      *) "$term" -e bash "$SELF" && exit 0 ;;
    esac
  done
  # No terminal found: carry on without one. The page still opens.
fi

# In a window we opened, keep it open on failure so the reason can be read.
# This is also why server.py is not started with `exec` below: exec would
# replace this shell, and the trap with it.
pause_on_error() {
  local code=$?
  if [ -n "${SCRIPT_BUILDER_IN_TERMINAL:-}" ] && [ "$code" -ne 0 ] \
     && [ "$code" -ne 130 ] && [ -t 0 ]; then
    printf "\n  Script Builder stopped (exit code %s). Press Enter to close." "$code"
    read -r _ || true
  fi
}
trap pause_on_error EXIT

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
#
# Healthy means pip runs, not merely python: without python3-venv, `-m venv`
# makes .venv/bin/python and then fails at ensurepip, and a check of `import
# sys` kept that pip-less folder for good — every later run died on "No
# module named pip", even after the package the message asks for was in.
VENV=".venv"
if [ -e "$VENV" ] && ! "$VENV/bin/python" -m pip --version >/dev/null 2>&1; then
  echo "  The existing environment is incomplete. Building it again."
  rm -rf "$VENV"
fi
if [ ! -x "$VENV/bin/python" ]; then
  echo "  Setting up Script Builder's packages (first run only)..."
  if ! "$PY" -m venv "$VENV"; then
    rm -rf "$VENV"
    echo "  Could not create the environment. On Debian and Ubuntu the venv" >&2
    echo "  module ships separately:  sudo apt-get install -y python3-venv" >&2
    echo "  Install it, then run this again." >&2
    exit 1
  fi
fi

"$VENV/bin/python" -m pip install --disable-pip-version-check --quiet \
    -r requirements.txt
# The server takes this script's place (exec), so whatever stops the
# script stops the server. Only in a window this script opened does
# the shell stay, so the trap above can keep a failure on screen.
if [ -n "${SCRIPT_BUILDER_IN_TERMINAL:-}" ]; then
  "$VENV/bin/python" server.py
else
  exec "$VENV/bin/python" server.py
fi
