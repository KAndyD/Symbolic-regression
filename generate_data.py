"""Явно пересоздать входные данные из config.json (перезаписывает data)."""
from model import generate
from common import ROOT,write_csv
import json
cfg=json.loads((ROOT/'config.json').read_text())
write_csv(ROOT/'data/regression.csv',generate(cfg['data_seed']))
