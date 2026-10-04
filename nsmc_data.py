# -*- coding: utf-8 -*-
"""NSMC 데이터 로딩과 데이터 감사.

여러분이 고칠 파일이 아닙니다. 그냥 import 해서 쓰세요.

처음 실행하면 data/ 밑으로 ratings_train.txt, ratings_test.txt 를 받아 옵니다.
(약 19MB, 한 번만 받습니다.)

audit() 이 왜 있냐면, 이 데이터셋에는 강의에서 말씀드린 라벨 잡음이 실제로
들어 있고, 그 숫자를 직접 보고 시작하셔야 하기 때문입니다.
"""
import os

import pandas as pd

RAW = "https://raw.githubusercontent.com/e9t/nsmc/master/{}"
FILES = {"train": "ratings_train.txt", "test": "ratings_test.txt"}
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# ---------------------------------------------------------------- 다운로드
def fetch(split):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, FILES[split])
    if not os.path.exists(path):
        import requests
        url = RAW.format(FILES[split])
        print(f"내려받는 중: {url}")
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
    return path


def load(split, drop_na=True):
    """split 은 'train' 또는 'test'. id / document / label 3개 컬럼입니다.

    document 가 비어 있는 행이 train 5건, test 3건 있습니다. 그대로 두면
    토크나이저에서 터지므로 기본값으로 떨어뜨립니다.
    """
    df = pd.read_csv(fetch(split), sep="\t", quoting=3, dtype={"id": str})
    if drop_na:
        df = df.dropna(subset=["document"])
        df = df[df["document"].str.strip().astype(bool)]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------- 데이터 감사
def audit(verbose=True):
    """데이터 명세와 라벨 잡음을 실제로 세어 봅니다. 시작 전에 한 번 돌려 보세요.

    보고서에 '이 데이터의 상한은 얼마인가'를 쓰려면 이 숫자가 필요합니다.
    """
    tr, te = load("train"), load("test")
    both = pd.concat([tr, te])

    overlap = te["document"].isin(set(tr["document"]))
    m = tr.drop_duplicates(subset=["document"]).set_index("document")["label"]
    ov = te[overlap].copy()
    ov["train_label"] = ov["document"].map(m)
    agree = float((ov["label"] == ov["train_label"]).mean()) if len(ov) else 1.0

    g = both.groupby("document")["label"].agg(["nunique", "count"])
    contra = g[g["nunique"] > 1]

    info = {
        "train_rows": len(tr),
        "test_rows": len(te),
        "train_pos_rate": float(tr["label"].mean()),
        "test_pos_rate": float(te["label"].mean()),
        "max_chars": int(both["document"].str.len().max()),
        "pct_under_140": float((both["document"].str.len() <= 140).mean()),
        "test_in_train": int(overlap.sum()),
        "test_in_train_pct": float(overlap.mean()),
        "overlap_label_agreement": agree,
        "contradictory_docs": int(len(contra)),
        "contradictory_rows": int(contra["count"].sum()),
    }
    if verbose:
        print(f"train {info['train_rows']:,}행   test {info['test_rows']:,}행")
        print(f"긍정 비율  train {info['train_pos_rate']:.3f}  "
              f"test {info['test_pos_rate']:.3f}   (거의 5:5)")
        print(f"최대 길이 {info['max_chars']}자, "
              f"{info['pct_under_140']*100:.2f}%가 140자 이하  -> max_length=64로 충분")
        print()
        print("--- 라벨 잡음 ---")
        print(f"test 문서 중 train 에도 있는 것  {info['test_in_train']:,}건 "
              f"({info['test_in_train_pct']*100:.2f}%)")
        print(f"  그중 라벨이 일치하는 비율      {info['overlap_label_agreement']*100:.2f}%")
        print(f"긍정·부정 양쪽으로 달린 문장     {info['contradictory_docs']:,}개")
        print(f"거기 걸린 행                     {info['contradictory_rows']:,}행 "
              f"({info['contradictory_rows']/len(both)*100:.2f}%)")
        print()
        print("같은 문장인데 라벨이 갈리는 사례:")
        top = contra.sort_values("count", ascending=False).head(6)
        for doc, r in top.iterrows():
            pos = float(both[both["document"] == doc]["label"].mean())
            d = doc if len(doc) <= 30 else doc[:30] + "..."
            print(f"  {int(r['count']):>4}행  긍정 {pos*100:>3.0f}%   \"{d}\"")
        print()
        print("어떤 모델도 NSMC 에서 100% 에 도달할 수 없습니다.")
        print("에러 분석 때 '모델이 틀림' 과 '라벨이 틀림' 을 반드시 나누세요.")
    return info


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    audit()
