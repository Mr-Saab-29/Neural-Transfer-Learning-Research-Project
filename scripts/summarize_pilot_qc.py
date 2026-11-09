#!/usr/bin/env python3
"""Summarize the most recent QC record per scan (not cumulative attempts)."""
import csv
from collections import Counter
from pathlib import Path

root = Path(__file__).resolve().parents[1]
with (root / 'data/manifests/pilot_cohort.csv').open() as stream:
    expected = {row['hf_path'] for row in csv.DictReader(stream)}
latest = {}
with (root / 'data/manifests/preprocessing_qc.csv').open() as stream:
    for row in csv.DictReader(stream):
        if row['hf_path'] in expected:
            latest[row['hf_path']] = row
print('Latest status:', dict(Counter(row['status'] for row in latest.values())))
print('Not attempted:', len(expected - latest.keys()))
for path, row in sorted(latest.items()):
    if row['status'] != 'pass':
        print(path, row['status'], 'retention=' + row['foreground_retention'], row['error'])
