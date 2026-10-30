# -*- coding: utf-8 -*-
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'common'))

"""vex.py —— v(ex)：跨组分注意力（唯一一份）

与 vim.py 共用引擎与划分文件，只把 --model 固定为 vex。
"""
import engine  # noqa: E402

if __name__ == '__main__':
    sys.argv = [sys.argv[0], '--model', 'vex'] + sys.argv[1:]
    engine.main()
