# -*- coding: utf-8 -*-
"""fp_matrix.py —— 从 odor_mixture/eval_encoders.py 抽出，去掉对气味模型的依赖

归档目的：E5（§8 迁移实验）只需要分子指纹，不需要气味编码器。
原脚本 eval_encoders.py 会 import encode_odorants（气味模型），
在归档目录下不成立，故把 fp_matrix 单独复制一份，函数体与原文件逐字一致。
"""
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import AllChem, MACCSkeys, Descriptors

RDLogger.DisableLog('rdApp.*')


def fp_matrix(smiles, kind):
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            rows.append(np.zeros(1, np.float32))
            continue
        if kind == 'ecfp4':
            v = AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)
        elif kind == 'ecfp6':
            v = AllChem.GetMorganFingerprintAsBitVect(m, 3, 2048)
        elif kind == 'maccs':
            v = MACCSkeys.GenMACCSKeys(m)
        elif kind == 'avalon':
            v = pyAvalonTools.GetAvalonFP(m, 1024)
        elif kind == 'rdkit2d':
            vals = []
            for _, fn in Descriptors.descList:
                try:
                    x = fn(m)
                except Exception:
                    x = np.nan
                vals.append(x if isinstance(x, (int, float)) else np.nan)
            rows.append(np.nan_to_num(np.array(vals, np.float32),
                                      nan=0.0, posinf=0.0, neginf=0.0))
            continue
        else:
            raise ValueError(kind)
        rows.append(np.array(v, np.float32))
    return np.stack(rows)
