"""Download pinned public model files and verify publisher SHA256 digests."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from download_file import download


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        while block := source.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadata', required=True, type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--kind', required=True, choices=['mert', 'omni'])
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    metadata = json.loads(args.metadata.read_text())
    repo, revision = metadata['id'], metadata['sha']
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.kind == 'mert':
        names = ['config.json', 'configuration_MERT.py', 'modeling_MERT.py',
                 'preprocessor_config.json', 'README.md', 'pytorch_model.bin']
    else:
        names = ['config.json', 'generation_config.json', 'preprocessor_config.json',
                 'chat_template.json', 'tokenizer.json', 'tokenizer_config.json',
                 'special_tokens_map.json', 'added_tokens.json', 'vocab.json',
                 'merges.txt', 'LICENSE', 'README.md', 'model.safetensors.index.json',
                 'model-00001-of-00003.safetensors', 'model-00002-of-00003.safetensors',
                 'model-00003-of-00003.safetensors']
    files = {f['rfilename']: f for f in metadata['siblings']}
    for name in names:
        info = files[name]
        dest = args.output_dir / name
        url = f'https://huggingface.co/{repo}/resolve/{revision}/{name}'
        if info['size'] >= 8 * 1024 * 1024:
            download(url, dest, info['size'], args.workers)
        elif not dest.exists() or dest.stat().st_size != info['size']:
            subprocess.run(['curl', '--silent', '--show-error', '--fail', '--location',
                            '--retry', '3', '--connect-timeout', '30', '--max-time', '180',
                            '--output', str(dest), url], check=True)
        if dest.stat().st_size != info['size']:
            raise RuntimeError(f'Incorrect downloaded size: {name}')
        expected = info.get('lfs', {}).get('sha256')
        if expected and sha256(dest) != expected:
            raise RuntimeError(f'Publisher SHA256 mismatch: {name}')
        print('Verified', name, flush=True)
    (args.output_dir / 'source_revision.json').write_text(json.dumps({
        'repository': repo, 'revision': revision, 'files': names,
    }, indent=2))


if __name__ == '__main__':
    main()
