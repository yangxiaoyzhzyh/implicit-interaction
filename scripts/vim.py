# -*- coding: utf-8 -*-
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'common'))

"""vim.py —— v(im)：一阶聚合 + 塑形编码器（唯一一份）

任务由 --tag 选（见 common/spec.py），划分由 --split-file 选（make_splits.py 产出）。
早停开关由 --no-early-stop 控制：组分数轴关闭，新分子轴与极值轴保持默认开启。
"""
import engine  # noqa: E402

if __name__ == '__main__':
    sys.argv = [sys.argv[0], '--model', 'vim'] + sys.argv[1:]
    engine.main()
