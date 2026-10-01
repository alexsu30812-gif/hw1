"""Extract the two course archives safely, preserving already verified files."""
from __future__ import annotations
import argparse
from pathlib import Path
import zipfile
import zlib


def extract(archive: Path, destination: Path):
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            path = (destination / info.filename).resolve()
            if not path.is_relative_to(destination):
                raise ValueError('Unsafe archive path')
            if info.is_dir():
                path.mkdir(parents=True, exist_ok=True)
                continue
            if path.is_file() and path.stat().st_size == info.file_size:
                if zlib.crc32(path.read_bytes()) & 0xffffffff == info.CRC:
                    continue
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix+'.extracting')
            with z.open(info) as source, temporary.open('wb') as target:
                while block := source.read(1024 * 1024):
                    target.write(block)
            temporary.replace(path)
    print(f'Extracted and CRC-checked {archive.name}', flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True)
    a=p.parse_args()
    extract(a.archive,a.destination)
