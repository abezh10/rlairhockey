# source env.sh  -> activates venv and puts ffmpeg/ffprobe on PATH
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/.venv/bin/activate"
export PATH="$ROOT/tools/bin:$PATH"
export UV_CACHE_DIR="$ROOT/.uv-cache"
export HF_HOME="$ROOT/models/hf"
# Roboflow keys etc. (git-ignored; see .env.example)
if [ -f "$ROOT/.env" ]; then set -a; source "$ROOT/.env"; set +a; fi
