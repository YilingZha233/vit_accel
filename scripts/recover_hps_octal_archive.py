"""Recover a gzip tar from BusyBox od -v -b PuTTY logging, validating offsets and CRC."""
import argparse
import gzip
import io
from pathlib import Path
import re
import tarfile


def decode(text):
    lines = text.splitlines()
    starts = [i for i, s in enumerate(lines) if s.strip() == 'HPS_OCT_BEGIN']
    if not starts:
        raise ValueError('Missing HPS_OCT_BEGIN')
    start = starts[-1]
    ends = [i for i in range(start+1, len(lines)) if lines[i].strip() == 'HPS_OCT_END']
    if not ends:
        raise ValueError('Missing HPS_OCT_END')
    data = bytearray()
    for line in lines[start+1:ends[0]]:
        parts = line.split()
        if not parts or not re.fullmatch(r'[0-7]{7,}', parts[0]):
            continue
        offset = int(parts[0], 8)
        if offset != len(data):
            raise ValueError(f'Offset mismatch {offset} != {len(data)}; preserve log and retransmit archive')
        for x in parts[1:]:
            if not re.fullmatch(r'[0-7]{3}', x) or int(x, 8) > 255:
                raise ValueError('Invalid octal byte')
            data.append(int(x, 8))
    gzip.decompress(data)  # verifies CRC and complete gzip stream before writing anything
    return bytes(data)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--log', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    data = decode(a.log.read_text(errors='replace'))
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out/'returned.tar.gz').write_bytes(data)
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as t:
        t.extractall(a.out, filter='data')
    print('Archive integrity check passed; recovered', len(data), 'bytes to', a.out)


if __name__ == '__main__':
    main()
