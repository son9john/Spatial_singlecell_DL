"""Download the 12 public Maynard et al. Visium sections from their authors."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import file_digest
from pathlib import Path
import csv
import shutil
from urllib.request import urlopen

SECTIONS = tuple(map(str, (*range(151507, 151511), *range(151669, 151677))))
HUMANPILOT = '044446d6bd8fc154aa74f7be62ec67effb1ec376'
GITHUB = f'https://raw.githubusercontent.com/LieberInstitute/HumanPilot/{HUMANPILOT}/10X'
S3 = 'https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5'
ROOT = Path(__file__).resolve().parent


def fetch(item):
    destination, url = item
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + '.download')
        with urlopen(url, timeout=60) as source, temporary.open('wb') as target:
            shutil.copyfileobj(source, target)
        temporary.replace(destination)
    with destination.open('rb') as source:
        digest = file_digest(source, 'sha256').hexdigest()
    return [str(destination.relative_to(ROOT)), url, destination.stat().st_size, digest]


if __name__ == '__main__':
    data = ROOT / 'data' / 'dlpfc'
    items = [(data / 'layers.tsv', f'{GITHUB}/barcode_level_layer_map.tsv')]
    for section in SECTIONS:
        items.extend([
            (data / section / 'counts.h5', f'{S3}/{section}_filtered_feature_bc_matrix.h5'),
            (data / section / 'positions.csv', f'{GITHUB}/{section}/tissue_positions_list.txt'),
        ])
    with ThreadPoolExecutor(max_workers=6) as executor:
        rows = list(executor.map(fetch, items))
    with (ROOT / 'data' / 'source_manifest.csv').open('w', newline='') as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(['path', 'source_url', 'size_bytes', 'sha256'])
        writer.writerows(rows)
    print(f'Downloaded or verified {len(rows)} files ({sum(row[2] for row in rows):,} bytes).')
