"""Shared integrity checks and small full-volume CNN for the follow-up study."""
import io
import json
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aomic_validation_batch import read, csv_bytes, folder_for
from final_aomic_test import atomic_write, lock, pipeline_hash
from preprocess_pilot import digest, validate_tensor
import preprocess_brain_pilot as brain
from src.data.dataset import project_path
from build_followup_protocol import build_files

PROTOCOL = ROOT / 'data/manifests/aomic_followup_v1'
OUT = ROOT / 'results/aomic_followup_v1'
ARCHITECTURE = 'whole64-conv8-16-32-gn-adaptive2-v1'


def load_protocol(directory=PROTOCOL):
    source = ROOT / 'data/manifests/aomic_downstream_splits_v1.csv'
    expected = build_files(read(source), digest(source))
    for name, data in expected.items():
        if (directory / name).read_bytes() != data:
            raise ValueError(f'Frozen follow-up file differs: {name}')
    return json.loads(expected['protocol.json']), read(directory / 'cohort.csv')


def code_hashes():
    paths = [ROOT / 'scripts' / name for name in (
        'followup_common.py', 'prepare_followup_cohort.py', 'train_wholebrain.py',
        'build_followup_protocol.py', 'final_aomic_test.py')]
    return {str(p.relative_to(ROOT)): digest(p) for p in paths}


def approved_artifact(row, review, preprocessing):
    if review['decision'] != 'accept' or review['status'] not in ('pass', 'review'):
        raise ValueError(f"Visual acceptance required: {row['participant_id']}")
    if review['status'] == 'review' and not review['notes'].strip():
        raise ValueError('Flagged acceptance requires notes')
    folder = folder_for(row)
    metadata = folder / 'metadata.json'
    if digest(metadata) != review['metadata_sha256']:
        raise ValueError('Stale visual approval')
    saved = json.loads(metadata.read_text())
    if saved['provenance']['hf_path'] != row['hf_path'] or any(
            saved['provenance'][k] != v for k, v in preprocessing.items()):
        raise ValueError('Incompatible preprocessing')
    for name in brain.ARTIFACTS:
        if digest(folder / name) != saved['sha256'][name]:
            raise ValueError(f'Artifact changed: {folder / name}')
    return dict(row, tensor_path=str((folder / 'tensor.pt').relative_to(ROOT)),
                tensor_sha256=saved['sha256']['tensor.pt'], metadata_sha256=digest(metadata), decision='accept')


def whole_image(row):
    path = project_path(ROOT, row['tensor_path'])
    if digest(path) != row['tensor_sha256'] or digest(path.parent / 'metadata.json') != row['metadata_sha256']:
        raise ValueError('Approved input changed')
    image = torch.load(path, map_location='cpu', weights_only=True)
    validate_tensor(image)
    # Average all 2x2x2 neighborhoods: full field of view, no central cropping.
    return F.avg_pool3d(image.unsqueeze(0), 2).squeeze(0)


def encoder():
    layers = []
    previous = 1
    for channels in (8, 16, 32):
        layers.extend([nn.Conv3d(previous, channels, 3, stride=2, padding=1),
                       nn.GroupNorm(4, channels), nn.GELU()])
        previous = channels
    return nn.Sequential(*layers)


class WholeBrainCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = encoder()
        self.head = nn.Sequential(nn.AdaptiveAvgPool3d(2), nn.Flatten(), nn.Linear(256, 1))

    def forward(self, image):
        return self.head(self.encoder(image)).flatten()


class ReconstructionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = encoder()
        self.decoder = nn.Sequential(nn.ConvTranspose3d(32, 16, 4, 2, 1), nn.GELU(),
                                     nn.ConvTranspose3d(16, 8, 4, 2, 1), nn.GELU(),
                                     nn.ConvTranspose3d(8, 1, 4, 2, 1))

    def forward(self, image):
        return self.decoder(self.encoder(image))


def save_torch(path, state):
    buffer = io.BytesIO()
    torch.save(state, buffer)
    atomic_write(path, buffer.getvalue())


def json_write(path, value):
    atomic_write(path, (json.dumps(value, indent=2, allow_nan=False) + '\n').encode())


def load_config():
    load_protocol()
    path = OUT / 'ready/config.json'
    config = json.loads(path.read_text())
    if config['code_sha256'] != code_hashes() or config['pipeline_sha256'] != pipeline_hash():
        raise ValueError('Implementation changed after freeze; preserve this run and version the setup explicitly')
    if digest(PROTOCOL / 'protocol.json') != config['protocol_sha256']:
        raise ValueError('Protocol changed')
    for name, checksum in config['files_sha256'].items():
        if digest(OUT / 'ready' / name) != checksum:
            raise ValueError(f'Frozen setup changed: {name}')
    if digest(OUT / 'review.csv') != config['review_sha256']:
        raise ValueError('QC decisions changed after freeze')
    return config, digest(path)
