# -*- coding: utf-8 -*-
"""채점 대상 지표. 팀마다 다르게 재면 비교가 안 되므로 통일해서 드립니다.

여러분이 고칠 파일이 아닙니다.

정확도 하나로 끝내지 마세요. 강의에서 말씀드린 대로 클래스가 5:5라 정확도가
의미는 있지만, 어느 쪽으로 틀리는지는 혼동행렬을 봐야 알 수 있습니다.
"""
import numpy as np


def classification_report(y_true, y_pred):
    """정확도, F1(긍정 기준), 혼동행렬을 한 번에."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "precision": prec,
        "recall": rec,
        "f1": 2 * prec * rec / max(prec + rec, 1e-9),
        "confusion": [[tn, fp], [fn, tp]],
        "n": int(len(y_true)),
    }


def print_confusion(cm, indent=""):
    (tn, fp), (fn, tp) = cm
    print(f"{indent}혼동행렬            예측:부정  예측:긍정")
    print(f"{indent}  실제 부정        {tn:>9,} {fp:>10,}")
    print(f"{indent}  실제 긍정        {fn:>9,} {tp:>10,}")
    if fp + fn:
        print(f"{indent}  부정->긍정 {fp:,}건 vs 긍정->부정 {fn:,}건 "
              f"(한쪽으로 쏠렸으면 임계값이나 데이터가 치우친 것)")


def summarize_seeds(accs):
    """시드 여러 개의 평균과 표준편차. 단일 실행은 결과가 아닙니다."""
    a = np.asarray(accs, dtype=float)
    return {"mean": float(a.mean()), "sd": float(a.std()), "runs": a.tolist()}
