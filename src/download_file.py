"""Resumable public-file downloader using bounded HTTP byte ranges and curl.

No credentials are read. Completed parts are retained until the final file is
assembled. The caller supplies the expected size obtained from the publisher.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import subprocess
import time


def download(url: str, output: Path, size: int, workers: int = 6,
             chunk_bytes: int = 8 * 1024 * 1024) -> None:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size == size:
        print(f"Already complete: {output.name}", flush=True)
        return
    parts = output.with_name(output.name + '.parts')
    parts.mkdir(exist_ok=True)
    plan = [(i, start, min(start + chunk_bytes, size) - 1)
            for i, start in enumerate(range(0, size, chunk_bytes))]

    def get_part(item):
        i, start, end = item
        final = parts / f'{i:06d}'
        length = end - start + 1
        if final.exists() and final.stat().st_size == length:
            return length
        tmp = final.with_suffix('.partial')
        headers = final.with_suffix('.headers')
        error = ''
        for attempt in range(5):
            cmd = ['curl', '--silent', '--show-error', '--fail', '--location',
                   '--connect-timeout', '30', '--max-time', '300',
                   '--max-filesize', str(length), '--range', f'{start}-{end}',
                   '--dump-header', str(headers), '--output', str(tmp), url]
            result = subprocess.run(cmd, capture_output=True, text=True)
            header = headers.read_text(errors='replace') if headers.exists() else ''
            expected = f'content-range: bytes {start}-{end}/{size}'
            if result.returncode == 0 and tmp.exists() and tmp.stat().st_size == length:
                if expected in header.lower() or (start == 0 and length == size):
                    tmp.replace(final)
                    return length
            error = result.stderr[-1000:] + ' ' + header[-250:]
            time.sleep(min(2 ** attempt, 15))
        raise RuntimeError(f'Range {start}-{end} failed for {output.name}: {error}')

    started = time.monotonic()
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(get_part, item) for item in plan]
        for n, future in enumerate(as_completed(futures), 1):
            completed += future.result()
            if n % 8 == 0 or n == len(plan):
                print(f'{output.name}: {n}/{len(plan)} parts, '
                      f'{completed/1e6:.1f} MB, {time.monotonic()-started:.0f}s', flush=True)
    assembled = output.with_name(output.name + '.assembling')
    with assembled.open('wb') as dest:
        for i, _, _ in plan:
            with (parts / f'{i:06d}').open('rb') as src:
                while block := src.read(1024 * 1024):
                    dest.write(block)
    if assembled.stat().st_size != size:
        raise RuntimeError('Assembled size mismatch')
    assembled.replace(output)
    print(f'Finished {output.name}: {size} bytes', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--size', required=True, type=int)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    download(args.url, args.output, args.size, args.workers)


if __name__ == '__main__':
    main()
