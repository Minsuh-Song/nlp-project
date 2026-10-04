# -*- coding: utf-8 -*-
"""정확도·F1·혼동행렬을 재서 results.json 으로 떨굽니다. 보고서 숫자는 여기서 나옵니다.

    python evaluate.py --ckpt runs/klue_roberta-base_seed0
    python evaluate.py --ckpt runs/klue_roberta-base_seed0 runs/..._seed1 runs/..._seed2

TODO 가 없습니다. train.py 와 predict.py 를 채우고 나면 그대로 돕니다.
체크포인트를 여러 개 주면 시드 평균과 표준편차를 같이 계산합니다.
"""
import argparse
import json
import os

import metrics
from predict import predict_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--max-len", dest="max_len", type=int, default=64)
    ap.add_argument("--out", default="runs/results.json")
    a = ap.parse_args()

    per_ckpt, accs = {}, []
    for ck in a.ckpt:
        df, pred, _ = predict_split(ck, "test", a.max_len)
        m = metrics.classification_report(df["label"].values, pred)
        per_ckpt[ck] = m
        accs.append(m["accuracy"])
        print(f"{os.path.basename(ck)}")
        print(f"  정확도  {m['accuracy']*100:.2f}%")
        print(f"  F1      {m['f1']*100:.2f}%")
        metrics.print_confusion(m["confusion"], indent="  ")
        print()

    out = {"per_checkpoint": per_ckpt}
    if len(accs) > 1:
        s = metrics.summarize_seeds(accs)
        out["seeds"] = s
        print(f"시드 {len(accs)}개 평균  {s['mean']*100:.2f}% "
              f"± {s['sd']*100:.2f}%p")
        print("  (" + ", ".join(f"{x*100:.2f}" for x in s["runs"]) + ")")
    else:
        print("시드가 하나뿐입니다. 단일 실행은 결과가 아닙니다 — 최소 3개 돌리세요.")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\n-> {a.out}")
    print("\n정확도에서 멈추지 마세요. error_analysis.py 를 돌리셔야 합니다.")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
