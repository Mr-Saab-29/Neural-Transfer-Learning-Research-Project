"""Synthetic scale-cohort integrity and transfer checks; no real-data execution."""
from pathlib import Path
import sys
import unittest

import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from train_pretraining_scale import validate_sources,foreground_mask,transfer_backbone
from followup_common import WholeBrainCNN,ReconstructionCNN


def cohort(size):
    return [dict(dataset=source,participant_id=f'{source}_{i}',hf_path=f'{source}/{i}',
                 tensor_path=f'{source}/{i}.pt',tensor_sha256='a'*64,metadata_sha256='b'*64,
                 decision='accept',split='train') for source in ('PT005_IXI','PT015_NKI') for i in range(size//2)]


class ScaleTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_nested_sources_and_rejections(self):
        small,medium,large=cohort(22),cohort(60),cohort(120)
        validate_sources(medium,small,medium,60)
        validate_sources(large,medium,large,120)
        changed=[dict(r) for r in medium]; changed[0]['tensor_sha256']='c'*64
        with self.assertRaises(ValueError): validate_sources(changed,small,medium,60)
        changed=[dict(r) for r in medium]; changed[0]['decision']='pending'
        with self.assertRaises(ValueError): validate_sources(changed,small,medium,60)
        changed=[dict(r) for r in medium]; changed[0]['dataset']='AOMIC'
        with self.assertRaises(ValueError): validate_sources(changed,small,medium,60)
        with self.assertRaises(ValueError): validate_sources(medium[:-1]+[medium[0]],small,medium,60)

    def test_mask_reproducibility_and_background(self):
        target=torch.ones(1,1,64,64,64); target[:,:,:16]=0
        a=foreground_mask(target,torch.Generator().manual_seed(43))
        b=foreground_mask(target,torch.Generator().manual_seed(43))
        self.assertTrue(torch.equal(a,b)); self.assertTrue(a.any())
        self.assertFalse(a[:,:,:16].any())
        with self.assertRaises(ValueError): foreground_mask(torch.zeros_like(target),torch.Generator().manual_seed(43))

    def test_transfer_keeps_identical_seeded_head(self):
        torch.manual_seed(42); source=ReconstructionCNN()
        torch.manual_seed(43); reference=WholeBrainCNN()
        torch.manual_seed(43); transferred=WholeBrainCNN()
        transfer_backbone(transferred,source.encoder.state_dict())
        for k,v in reference.head.state_dict().items():
            self.assertTrue(torch.equal(v,transferred.head.state_dict()[k]))
        for k,v in source.encoder.state_dict().items():
            self.assertTrue(torch.equal(v,transferred.encoder.state_dict()[k]))
        bad={k:v.clone() for k,v in source.encoder.state_dict().items()}
        bad['0.weight'].fill_(float('nan'))
        with self.assertRaises(ValueError): transfer_backbone(transferred,bad)


if __name__=='__main__': unittest.main()
