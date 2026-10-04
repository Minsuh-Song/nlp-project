# -*- coding: utf-8 -*-
"""파인튜닝. 여기 TODO 하나가 여러분이 채울 첫 번째 부분입니다.

    python train.py --model klue/roberta-base --seed 0
    python train.py --model beomi/KcELECTRA-base --seed 0

TODO를 채우기 전에 실행하면 무엇을 채워야 하는지 알려 주고 멈춥니다.

기본 설정은 강의 16번 슬라이드 그대로입니다. 바꾸기 전에 일단 이대로 한 번
돌려 보세요. lr 2e-5 / batch 32 / 2 epoch / max_length 64 / AdamW + warmup 10%.
"""
import argparse
import json
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import nsmc_data as D
from tokenize_util import NSMCDataset, build_tokenizer


# ================================================================ TODO 1 / 2
def training_step(model, batch):
    """배치 하나에 대한 손실을 계산해서 스칼라 텐서로 반환하세요.

    강의 15번 슬라이드가 말하는 그림입니다. 새로 붙는 건 선형 레이어 하나뿐이고,
    AutoModelForSequenceClassification 이 그 선형 레이어까지 포함해서 만들어 줍니다.

      1. batch 는 dict 입니다. 키는 input_ids, attention_mask, labels
         (모델에 따라 token_type_ids 가 더 있을 수 있습니다.)
      2. labels 를 빼내고 나머지를 모델에 넘깁니다.
             labels = batch.pop("labels")
             out = model(**batch)
      3. out.logits 는 (B, 2) 입니다. 손실은
             F.cross_entropy(out.logits, labels)

    사실 labels 를 그대로 넘기면 (`model(**batch)`) transformers 가 손실을
    알아서 계산해서 out.loss 에 넣어 줍니다. 그 방법도 정답입니다.
    다만 logits 에서 직접 계산해 보시는 쪽을 권합니다. 무슨 일이 일어나는지
    눈에 보이거든요.

    주의: batch 안의 텐서는 이미 올바른 device 에 올라와 있습니다.
    """
    raise NotImplementedError(
        "train.py 의 training_step 을 구현하세요. "
        "docstring 에 세 단계가 순서대로 적혀 있습니다.")


# ================================================================ 학습 루프 (그대로 두셔도 됩니다)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="klue/roberta-base")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", dest="max_len", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-train", type=int, default=0,
                    help="빠른 점검용. 0이면 전체 15만 건")
    ap.add_argument("--out", default="runs")
    a = ap.parse_args()

    from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    tag = f"{a.model.replace('/', '_')}_seed{a.seed}"

    tr = D.load("train")
    if a.limit_train:
        tr = tr.sample(a.limit_train, random_state=0).reset_index(drop=True)
    te = D.load("test")
    print(f"device={device}  model={a.model}  seed={a.seed}")
    print(f"train {len(tr):,}  test {len(te):,}")
    print(f"lr={a.lr}  bs={a.bs}  epochs={a.epochs}  max_len={a.max_len}\n")

    tok = build_tokenizer(a.model)
    dtr = DataLoader(NSMCDataset(tr, tok, a.max_len), batch_size=a.bs,
                     shuffle=True, drop_last=True)
    if len(dtr) == 0:
        raise SystemExit(
            f"배치가 하나도 만들어지지 않습니다. 학습 데이터가 {len(tr)}건인데 "
            f"batch size 가 {a.bs} 입니다 (drop_last=True). "
            f"--bs 를 줄이거나 --limit-train 을 늘리세요. "
            f"이대로 두면 학습을 한 번도 안 하고 체크포인트만 저장됩니다.")

    model = AutoModelForSequenceClassification.from_pretrained(
        a.model, num_labels=2).to(device)

    total = len(dtr) * a.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)

    t0, losses = time.time(), []
    for ep in range(a.epochs):
        model.train()
        run, nb = 0.0, 0
        for i, batch in enumerate(dtr):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            loss = training_step(model, batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            run += loss.item(); nb += 1
            if (i + 1) % 500 == 0:
                print(f"  ep{ep+1} step {i+1}/{len(dtr)}  loss {run/nb:.4f}  "
                      f"({time.time()-t0:.0f}s)", flush=True)
        losses.append(run / max(nb, 1))
        print(f"  epoch {ep+1}/{a.epochs} 끝  train_loss {losses[-1]:.4f}", flush=True)

    mins = (time.time() - t0) / 60
    ckpt = os.path.join(a.out, tag)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)
    json.dump({"losses": losses, "train_minutes": mins, "model": a.model,
               "seed": a.seed, "lr": a.lr, "bs": a.bs, "epochs": a.epochs,
               "max_len": a.max_len},
              open(ckpt + "_train.json", "w"), indent=2)
    print(f"\n{mins:.1f}분  ->  {ckpt}")
    print("이제 evaluate.py 로 정확도를 재세요. 학습 손실만 보고 판단하지 마세요.")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()
