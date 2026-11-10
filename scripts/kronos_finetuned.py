from pathlib import Path
import sys
import torch
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "Kronos"))
from model import Kronos, KronosTokenizer, KronosPredictor

DEVICE = "cpu"  # inference: keep CPU, or "mps" after model.eval()
CKPT = ROOT / "outputs" / "kronos_ft_small"

tokenizer = KronosTokenizer.from_pretrained(str(CKPT / "tokenizer" / "best_model"))
model = Kronos.from_pretrained(str(CKPT / "basemodel" / "best_model"))
tokenizer.eval()
model.eval()

predictor = KronosPredictor(model, tokenizer, device=DEVICE, max_context=512)
