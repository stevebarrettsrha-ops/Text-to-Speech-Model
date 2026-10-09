"""Reuse local model folders without copying or deleting checkpoint data."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import stat

WEIGHTS = {'.safetensors', '.bin', '.pt', '.pth'}
PARTIAL = {'.part', '.incomplete'}


def whole(folder: Path) -> bool:
    """Cheap structural check, not a checksum or an inference guarantee."""
    try:
        config = folder / 'config.json'
        if not config.is_file() or not config.stat().st_size:
            return False
        files = list(folder.rglob('*')) if folder.is_dir() else []
        if any(p.suffix in PARTIAL for p in files):
            return False
        # A nested audio tokenizer alone does not make its parent a model.
        weights = [p for p in files if p.parent == folder and p.suffix in WEIGHTS]
        if not weights:
            return False
        for p in [p for p in files if p.suffix in WEIGHTS]:
            if not p.is_file() or not p.stat().st_size:
                return False
            with p.open('rb') as f:
                if f.read(80).startswith(b'version https://git-lfs.github.com/spec/'):
                    return False
        for index in (p for p in files if p.name.endswith(('.safetensors.index.json', '.bin.index.json'))):
            manifest = json.loads(index.read_text(encoding='utf-8'))
            shards = manifest.get('weight_map')
            if not isinstance(shards, dict) or not shards:
                return False
            for name in set(shards.values()):
                if not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts:
                    return False
                shard = index.parent / name
                if not shard.is_file() or not shard.stat().st_size:
                    return False
        return True
    except (OSError, ValueError, TypeError):
        return False


def hub_cache() -> Path:
    home = os.environ.get('HF_HOME') or str(Path(os.environ.get('XDG_CACHE_HOME') or Path.home() / '.cache') / 'huggingface')
    return Path(os.environ.get('HF_HUB_CACHE') or os.environ.get('HUGGINGFACE_HUB_CACHE') or Path(home) / 'hub').expanduser()


def cached_snapshot(repo: str) -> Path | None:
    """Use main's local snapshot, or a sole complete snapshot; no network."""
    root = hub_cache() / ('models--' + repo.replace('/', '--'))
    try:
        ref = root / 'refs' / 'main'
        if ref.is_file():
            revision = ref.read_text(encoding='utf-8').strip()
            if not revision or Path(revision).name != revision or revision in ('.', '..'):
                return None
            candidates = [root / 'snapshots' / revision]
        else:
            candidates = list((root / 'snapshots').iterdir()) if (root / 'snapshots').is_dir() else []
        complete = [p for p in candidates if (p / 'config.json').is_file() and whole(p)]
        return complete[0] if len(complete) == 1 else None
    except OSError:
        return None


def extra_roots(comfy_dir: Path, engine: str) -> list[Path]:
    """Read ComfyUI's normal extra paths with its base_path semantics."""
    path = comfy_dir / 'extra_model_paths.yaml'
    if not path.is_file():
        return []
    import yaml
    with path.open(encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise RuntimeError(f'{path} must contain a mapping of model path groups.')
    out = []
    keys = ('qwen-tts', 'TTS') if engine == 'qwen' else ('moss-tts',)
    for group in config.values():
        if not isinstance(group, dict):
            continue
        base = Path(os.path.expandvars(os.path.expanduser(str(group.get('base_path') or path.parent))))
        if not base.is_absolute():
            base = path.parent / base
        for key in keys:
            value = group.get(key, '')
            if not isinstance(value, str):
                raise RuntimeError(f'{path}: {key} paths must be a string.')
            for line in value.splitlines():
                if line.strip():
                    out.append((base / line.strip()).absolute())
    return out


def link_directory(source: Path, target: Path) -> None:
    """Symlink, or an unprivileged Windows junction; never replace a folder."""
    source = source.resolve()
    if os.path.lexists(target):
        if target.resolve() == source:
            return
        # A stale link has no checkpoint data to replace. Repair the link
        # after its source moved; leave real partial folders untouched.
        junction = (os.name == 'nt' and getattr(target.lstat(), 'st_reparse_tag', 0)
                    == getattr(stat, 'IO_REPARSE_TAG_MOUNT_POINT', 0xA0000003))
        if not target.exists() and target.is_symlink():
            target.unlink()
        elif not target.exists() and junction:
            target.rmdir()
        else:
            raise RuntimeError(f'{target} already exists; it was left unchanged. Existing weights are at {source}.')
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.symlink_to(source, target_is_directory=True)
        return
    except OSError as exc:
        if os.name == 'nt':
            # cmd expands percent signs even inside quotes. Refuse such paths
            # rather than interpreting any part of a user-configured filename.
            values = (str(target.absolute()), str(source))
            if not any(any(c in value for c in '%"\r\n') for value in values):
                command = f'mklink /J "{values[0]}" "{values[1]}"'
                result = subprocess.run(['cmd.exe', '/d', '/v:off', '/s', '/c', command],
                                        capture_output=True, text=True)
                if result.returncode == 0 and target.is_dir():
                    return
        raise RuntimeError(f'Already-downloaded weights were found at {source}, but could not be linked to {target}: {exc}. Set the models folder to the existing location or enable directory links, then Recheck. No second copy was downloaded.') from exc
