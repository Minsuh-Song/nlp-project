# -*- coding: utf-8 -*-
"""추론. 여기 TODO 하나가 여러분이 채울 두 번째 부분입니다.

    python predict.py --ckpt runs/klue_roberta-base_seed0

evaluate.py 와 error_analysis.py 가 이 함수를 불러 씁니다.
"""
import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

import nsmc_data as D
from tokenize_util import NSMCDataset, build_tokenizer


# ================================================================ TODO 2 / 2
@torch.no_grad()
def predict_loop(model, loader, device):
    """전체 배치를 돌면서 예측 라벨과 확률을 반환하세요.

    반환값은 두 개의 numpy 배열입니다.
        pred  (N,)   0 또는 1
        prob  (N, 2) softmax 확률.  prob[:, 1] 이 긍정 확률

    순서대로 하시면 됩니다.

      1. model.eval() 로 바꿉니다. (dropout 이 꺼집니다. 빠뜨리면 결과가 흔들립니다.)
      2. loader 를 돌면서:
           labels 가 있으면 빼 두고, 나머지를 device 로 옮깁니다.
           logits = model(**batch).logits
           p = torch.softmax(logits, dim=-1)
           예측은 p.argmax(-1)
      3. 배치별 결과를 이어 붙여서 numpy 로 반환합니다.
             torch.cat(...).cpu().numpy()

    주의: 함수 위에 @torch.no_grad() 가 이미 붙어 있습니다. 그래서 따로
    no_grad 로 감싸지 않으셔도 됩니다. 이걸 빼면 5만 건에서 메모리가 터집니다.
    """
    model.eval()
    all_pred, all_prob = [], []

    for batch in loader:
        batch.pop("labels", None)                           # 정답은 예측에 필요 없으니 제거
        batch = {k: v.to(device) for k, v in batch.items()} # 나머지를 GPU로 이동

        logits = model(**batch).logits                      # (배치크기, 2) 점수
        p = torch.softmax(logits, dim=-1)                   # 확률로 변환

        all_pred.append(p.argmax(-1).cpu())                 # 예측 라벨 (0 또는 1)
        all_prob.append(p.cpu())                            # 확률

    pred = torch.cat(all_pred).numpy()
    prob = torch.cat(all_prob).numpy()
    return pred, prob


# ================================================================ CLI (그대로 두셔도 됩니다)
def load(ckpt, device):
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(ckpt).to(device)
    tok = build_tokenizer(ckpt)
    model.eval()
    return model, tok


def predict_split(ckpt, split="test", max_len=64, bs=256, device=None):
    """체크포인트로 한 split 전체를 예측합니다. 다른 스크립트들이 이걸 부릅니다."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = load(ckpt, device)
    df = D.load(split)
    dl = DataLoader(NSMCDataset(df, tok, max_len), batch_size=bs, shuffle=False)
    pred, prob = predict_loop(model, dl, device)
    return df, pred, prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-len", dest="max_len", type=int, default=64)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    df, pred, prob = predict_split(a.ckpt, a.split, a.max_len)
    assert len(pred) == len(df), f"예측 {len(pred)}개, 데이터 {len(df)}개 — 개수가 안 맞습니다"
    assert np.isfinite(prob).all(), "확률에 NaN 이 있습니다"

    out = a.out or os.path.join(a.ckpt, f"pred_{a.split}.npz")
    np.savez(out, pred=pred, prob=prob)
    acc = float((pred == df["label"].values).mean())
    print(f"{len(df):,}건 예측 완료   정확도 {acc*100:.2f}%   ->  {out}")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
