# -*- coding: utf-8 -*-
"""TF-IDF + 로지스틱 회귀 베이스라인. 여러분이 넘어야 할 기준선입니다.

    python baseline.py

여러분이 고칠 파일이 아닙니다. 먼저 이걸 돌려서 기준점을 잡으세요.
기준선 없이 나온 숫자는 해석할 수가 없습니다.

주의할 점이 하나 있습니다. 어절(word) 단위와 문자(char) 단위의 차이가 큽니다.
한국어는 교착어라 어절로 자르면 '재미없었어요' 가 통째로 한 토큰이 되고,
조금만 달라져도 미등록어가 됩니다. 문자 n-gram 은 그 문제가 없습니다.

그래서 '베이스라인을 이겼다' 고 말하려면 약한 쪽이 아니라
**문자 n-gram** 을 기준으로 삼으셔야 합니다.

이 모델은 결정론적입니다. 시드를 바꿔도 결과가 같아요. '시드 3개' 규정은
신경망 학습에만 해당합니다.
"""
import argparse
import json
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import metrics
from nsmc_data import load

CONFIGS = [
    ("word 1-2gram",    dict(analyzer="word",    ngram_range=(1, 2))),
    ("char_wb 2-4gram", dict(analyzer="char_wb", ngram_range=(2, 4))),
]


def run(tr, te, vec_kw, C=4.0):
    vec = TfidfVectorizer(min_df=2, sublinear_tf=True, **vec_kw)
    t0 = time.time()
    Xtr = vec.fit_transform(tr["document"])
    Xte = vec.transform(te["document"])
    clf = LogisticRegression(C=C, max_iter=2000)
    clf.fit(Xtr, tr["label"])
    pred = clf.predict(Xte)
    return pred, {"n_features": Xtr.shape[1], "seconds": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/baseline.json")
    a = ap.parse_args()

    tr, te = load("train"), load("test")
    print(f"train {len(tr):,}  test {len(te):,}\n")

    out = {}
    for name, kw in CONFIGS:
        pred, info = run(tr, te, kw)
        m = metrics.classification_report(te["label"].values, pred)
        out[name] = {**m, **info}
        print(f"{name}")
        print(f"  특징 수     {info['n_features']:,}")
        print(f"  걸린 시간   {info['seconds']:.1f}초  (CPU)")
        print(f"  정확도      {m['accuracy']*100:.2f}%")
        print(f"  F1          {m['f1']*100:.2f}%")
        metrics.print_confusion(m["confusion"], indent="  ")
        print()

    import os
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"-> {a.out}")
    best = max(out.values(), key=lambda v: v["accuracy"])["accuracy"]
    print(f"\n여러분이 넘어야 할 선: {best*100:.2f}%")
    print("이걸 못 넘으면 사전학습 모델을 쓴 의미가 없습니다.")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
