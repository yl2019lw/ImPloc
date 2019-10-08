#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
from util import datautil
from util import constant as c

DST = "/tmp/testimgs"

test_genes = datautil.get_test_gene_list(size=0)
for g in test_genes:
    pics = datautil.get_gene_pics(g)
    dst_dir = os.path.join(DST, g)
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    for pic in pics:
        src = os.path.join(c.QDATA_DIR, g, pic)
        dst = os.path.join(dst_dir, pic)
        shutil.copy(src, dst)
