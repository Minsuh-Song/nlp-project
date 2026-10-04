# -*- coding: utf-8 -*-
"""토크나이저와 Dataset. 여러분이 고칠 파일이 아닙니다.

강의 8번 슬라이드의 결론이 코드로는 이 파일입니다.
**토크나이저는 백본과 세트입니다.** 모델 이름과 토크나이저 이름이 다르면
성능이 조용히 떨어지고, 에러도 안 납니다. 그래서 여기서 한 이름으로 묶어 뒀습니다.
"""
import torch
from torch.utils.data import Dataset


def build_tokenizer(model_name):
    """모델과 같은 이름으로 토크나이저를 불러옵니다. 이 둘은 반드시 같아야 합니다."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name)


class NSMCDataset(Dataset):
    """문장을 미리 전부 토크나이즈해서 들고 있습니다. 15만 건이면 메모리에 올라갑니다.

    max_length=64 인 이유는 nsmc_data.audit() 이 보여 주는 대로
    리뷰의 99.9%가 140자 이하이고, 서브워드로 자르면 64토큰이면 충분하기 때문입니다.
    """

    def __init__(self, df, tok, max_len=64, with_labels=True):
        self.enc = tok(list(df["document"]), truncation=True, max_length=max_len,
                       padding="max_length", return_tensors="pt")
        self.with_labels = with_labels
        self.y = (torch.tensor(df["label"].values, dtype=torch.long)
                  if with_labels else None)

    def __len__(self):
        return self.enc["input_ids"].shape[0]

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        if self.with_labels:
            item["labels"] = self.y[i]
        return item
