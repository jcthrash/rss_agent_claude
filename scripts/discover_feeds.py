#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests

from common_paths import OUTPUT_DIR, ensure_project_dirs

QUERY_TERMS = [
    'microbial oceanography',
    'marine microbiology',
    'biogeochemistry',
    'microbial genomics',
    'genome scale metabolic modeling',
    'metagenomics',
    'microbial ecology',
]
URL = 'https://api.crossref.org/works'


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Discover candidate journals via Crossref queries')
    p.add_argument('--outfile', type=Path, default=OUTPUT_DIR / 'reports' / 'discovered_journals.json')
    return p.parse_args()


def main() -> None:
    ensure_project_dirs()
    args = parse_args()
    journals = {}
    for q in QUERY_TERMS:
        print(f'Searching Crossref for: {q}')
        r = requests.get(URL, params={'query': q, 'rows': 100}, timeout=30)
        r.raise_for_status()
        data = r.json()
        for item in data['message']['items']:
            title = item.get('container-title', [])
            issn = item.get('ISSN', [])
            publisher = item.get('publisher', '')
            if title:
                name = title[0]
                journals[name] = {'journal': name, 'publisher': publisher, 'issn': issn}
    out = sorted(journals.values(), key=lambda x: x['journal'].lower())
    args.outfile.parent.mkdir(parents=True, exist_ok=True)
    args.outfile.write_text(json.dumps(out, indent=2))
    print(f'Saved {len(out)} candidate journals to {args.outfile}')


if __name__ == '__main__':
    main()
