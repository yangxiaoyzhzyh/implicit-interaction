# -*- coding: utf-8 -*-
"""encoder.py — 共享分子编码器（v(im) 与 v(ex) 两条路线共用）

实验 2/4/5/7 的公共模型组件（本身不产出数据）
GIN×3(64) → Linear(64,256) → TFM×2(256, nhead 4) → 逐分子 mean pooling

`GINConv`, `Encoder`, `mol_to_graph` and `DEVICE` are the backbone behind every reported
extrapolation run: a GIN over the molecular graph followed by a transformer inside the
component (see the supplementary materials, M1). They were previously defined inside
`probe_transfer_matrix.py`, which also carried an exploratory transfer-matrix probe that the
paper does not report; that probe has been moved out of the released package, and the shared
pieces live here so the experiment scripts import a single definition.
"""
import csv
import os

import numpy as np
import torch
from rdkit import Chem, RDLogger
from torch_geometric.utils import scatter, to_dense_batch

RDLogger.DisableLog('rdApp.*')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GINConv(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(d, d), torch.nn.BatchNorm1d(d), torch.nn.ReLU(),
            torch.nn.Linear(d, d), torch.nn.BatchNorm1d(d), torch.nn.ReLU())
        self.eps = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x, ei):
        r, c = ei
        o = scatter(x[c], r, dim=0, dim_size=x.size(0), reduce='sum')
        return torch.nn.functional.relu(self.mlp((1 + self.eps) * x + o))


class Encoder(torch.nn.Module):
    """Shared backbone for the chemical tasks; `atom_embed` returns per-atom vectors.

    pad_from: input feature width (CheMixHub models use 32 = 31 + mole fraction;
               odor models use 31, no mole fraction)
    normalize: L2-normalise the pooled vector (cosine-head geometry, odor models)
    """

    def __init__(self, dim=256, nhead=4, pad_from=32, normalize=False):
        super().__init__()
        self.pad_from = pad_from
        self.normalize = normalize
        self.pad = torch.nn.ZeroPad2d((0, 64 - pad_from, 0, 0))
        self.convs = torch.nn.ModuleList([GINConv(64) for _ in range(3)])
        self.proj = torch.nn.Linear(64, dim)
        enc = torch.nn.TransformerEncoderLayer(
            d_model=dim, nhead=nhead, dim_feedforward=2 * dim,
            batch_first=True, dropout=0.1)
        self.tfm = torch.nn.TransformerEncoder(enc, 2)

    def atom_embed(self, x, ei, batch):
        x = self.pad(x)
        for c in self.convs:
            x = c(x, ei)
        x = self.proj(x)
        x, m = to_dense_batch(x, batch)
        return self.tfm(x, src_key_padding_mask=~m if m is not None else None)[m]

    def pooled(self, x, ei, batch):
        xa = self.atom_embed(x, ei, batch)
        v = scatter(xa, batch, dim=0,
                    dim_size=int(batch.max()) + 1, reduce='mean')
        if self.normalize:
            v = torch.nn.functional.normalize(v, dim=1)
        return v


TARGET_ELEMENTS = [6, 7, 8, 9, 16, 17, 35, 53]
CHIRAL_TAGS = [Chem.ChiralType.CHI_TETRAHEDRAL_CW,
               Chem.ChiralType.CHI_TETRAHEDRAL_CCW]
_gc = {}


def mol_to_graph(smi):
    """31 atom features + edge index; cached per SMILES."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    HYBRIDS = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2,
               Chem.HybridizationType.SP3, Chem.HybridizationType.SP3D,
               Chem.HybridizationType.SP3D2]
    nf = []
    for a in mol.GetAtoms():
        f = [1 if a.GetAtomicNum() == x else 0 for x in TARGET_ELEMENTS]
        f += [1 if a.GetChiralTag() == x else 0 for x in CHIRAL_TAGS]
        deg = min(a.GetDegree(), 5)
        f += [1.0 if deg == i else 0.0 for i in range(6)]
        h = min(a.GetTotalNumHs(), 4)
        f += [1.0 if h == i else 0.0 for i in range(5)]
        f += [1.0 if a.GetHybridization() == hyb else 0.0 for hyb in HYBRIDS]
        f += [1.0 if a.GetIsAromatic() else 0.0]
        fc = int(a.GetFormalCharge())
        f += [1.0 if fc < 0 else 0.0, 1.0 if fc == 0 else 0.0,
              1.0 if fc > 0 else 0.0]
        f += [1.0 if a.IsInRing() else 0.0]
        nf.append(f[:31])
    ei = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        ei += [[i, j], [j, i]]
    if not ei:
        ei = [[0, 0]]
    return {'x': np.array(nf, dtype=np.float32),
            'ei': np.array(ei, dtype=np.int64).T}


def load_compounds(data_root):
    """{compound_id: SMILES} from a CheMixHub-style compounds.csv."""
    cmp_smiles = {}
    with open(os.path.join(data_root, 'compounds.csv')) as f:
        for r in csv.DictReader(f):
            cmp_smiles[int(float(r['compound_id']))] = r['smiles']
    return cmp_smiles
