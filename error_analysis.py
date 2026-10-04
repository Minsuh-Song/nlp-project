# -*- coding: utf-8 -*-
"""에러 분석. 여기서 나오는 표가 제출물이고, 비중이 가장 큰 항목입니다.

    python error_analysis.py --ckpt runs/klue_roberta-base_seed0

이 스크립트가 해 주는 것:
  1. 틀린 예측을 전부 모아서 runs/errors_<태그>.csv 로 저장
  2. 그중 모델이 가장 확신하며 틀린 100건을 뽑아 runs/errors_top100_<태그>.csv 로 저장
  3. 분류용 빈 칼럼(category, note)을 붙여 둠

여기서부터는 **사람이 읽는 일**입니다. 자동으로 못 합니다.
errors_top100_*.csv 를 열어서 category 칼럼을 채우세요. 강의 18번 슬라이드의 유형입니다.

  label_wrong     라벨이 잘못됨 (사람이 읽으면 답이 명백한데 라벨이 반대)
  irony           반어 · 비꼬기
  negation        부정 표현 오독
  too_short       너무 짧아 근거가 없음
  mixed           한 문장에 두 감정
  cast_only       배우 · 제작진 언급만
  other           위에 안 들어가는 것

다 채우고 나면 --summarize 로 분포를 뽑을 수 있습니다.

    python error_analysis.py --summarize runs/errors_top100_<태그>.csv

강의 19번 슬라이드의 구분을 잊지 마세요. label_wrong 은 고쳐서 재학습하면
정확도가 올라갑니다. 나머지는 모델 쪽 문제고 대응이 다릅니다.
"""
import argparse
import os

import numpy as np
import pandas as pd

CATEGORIES = ["label_wrong", "irony", "negation", "too_short", "mixed",
              "cast_only", "other"]


def build(ckpt, max_len=64, top_k=100):
    from predict import predict_split
    df, pred, prob = predict_split(ckpt, "test", max_len)
    y = df["label"].values
    wrong = pred != y

    e = df[wrong].copy()
    e["pred"] = pred[wrong]
    e["p_pos"] = prob[wrong, 1]
    # 모델이 얼마나 확신하며 틀렸는가. 클수록 '자신 있게 틀린' 경우입니다.
    e["confidence"] = np.max(prob[wrong], axis=1)
    e = e.sort_values("confidence", ascending=False).reset_index(drop=True)

    top = e.head(top_k).copy()
    top["category"] = ""
    top["note"] = ""
    return e, top, float(wrong.mean())


def summarize(path):
    df = pd.read_csv(path)
    if "category" not in df.columns:
        raise SystemExit("category 칼럼이 없습니다.")
    filled = df[df["category"].astype(str).str.strip().astype(bool)]
    if not len(filled):
        raise SystemExit(
            f"{path} 의 category 칼럼이 비어 있습니다. 먼저 손으로 채우셔야 합니다.")
    vc = filled["category"].value_counts()
    print(f"{len(filled)}/{len(df)}건 분류됨\n")
    print(f"{'유형':<14} {'건수':>5} {'비율':>7}")
    for k, v in vc.items():
        print(f"{k:<14} {v:>5} {v/len(filled)*100:>6.0f}%")
    unknown = set(vc.index) - set(CATEGORIES)
    if unknown:
        print(f"\n(참고: 정의에 없는 유형 {sorted(unknown)})")
    lw = vc.get("label_wrong", 0) / len(filled)
    print(f"\n라벨 오류 비율 {lw*100:.0f}% — 이만큼은 모델을 고쳐도 안 올라갑니다.")
    print("나머지가 여러분이 손댈 수 있는 부분입니다.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--summarize", metavar="CSV")
    ap.add_argument("--max-len", dest="max_len", type=int, default=64)
    ap.add_argument("--top-k", dest="top_k", type=int, default=100)
    ap.add_argument("--out", default="runs", help="csv 를 떨굴 폴더")
    a = ap.parse_args()

    if a.summarize:
        summarize(a.summarize)
        return
    if not a.ckpt:
        raise SystemExit("--ckpt 또는 --summarize 중 하나는 주셔야 합니다.")

    e, top, rate = build(a.ckpt, a.max_len, a.top_k)
    # 체크포인트 폴더 안이 아니라 runs/ 에 떨굽니다. 제출할 때 체크포인트는
    # 빼고 내야 하는데, csv 가 그 안에 있으면 같이 딸려 나가거나 빠집니다.
    tag = os.path.basename(a.ckpt.rstrip("/\\"))
    d = a.out
    os.makedirs(d, exist_ok=True)
    p_all = os.path.join(d, f"errors_{tag}.csv")
    p_top = os.path.join(d, f"errors_top100_{tag}.csv")
    e.to_csv(p_all, index=False, encoding="utf-8-sig")
    top.to_csv(p_top, index=False, encoding="utf-8-sig")

    print(f"틀린 예측 {len(e):,}건 ({rate*100:.2f}%)  ->  {p_all}")
    print(f"확신하며 틀린 상위 {len(top)}건        ->  {p_top}")
    print()
    print("이제 errors_top100.csv 를 엑셀로 열어서 category 칼럼을 채우세요.")
    print("쓸 수 있는 값: " + ", ".join(CATEGORIES))
    print()
    print("가장 확신하며 틀린 5건 미리보기:")
    for _, r in top.head(5).iterrows():
        doc = r["document"]
        doc = doc if len(doc) <= 42 else doc[:42] + "..."
        print(f"  정답 {r['label']} / 예측 {r['pred']} (확신 {r['confidence']:.2f})  \"{doc}\"")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
