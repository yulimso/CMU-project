#!/usr/bin/env python3
import os, json, math, argparse
from datasets import load_dataset
from PIL import Image

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--subset", default="large_random_1k")
    ap.add_argument("--export_dir", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--relative_prefix", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=-1)
    args = ap.parse_args()

    os.makedirs(args.export_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)

    ds = load_dataset(
        "poloclub/diffusiondb",
        args.subset,
        trust_remote_code=True,
        cache_dir=args.cache_dir
    )["train"]

    n = len(ds)
    start = max(0, args.start)
    end = n if args.limit < 0 else min(n, start + args.limit)

    # 1) 먼저 범위 내에서 "prompt 첫 등장"만 모으기 (순서 유지)
    seen = set()
    kept_indices = []
    for i in range(start, end):
        rec = ds[i]
        prompt = rec.get("prompt") or ""
        if prompt in seen:
            continue
        seen.add(prompt)
        kept_indices.append(i)

    # 저장 개수에 맞춰 패딩 계산
    k = len(kept_indices)
    pad = max(5, int(math.log10(max(1, k))) + 1)

    # 2) 저장 & JSONL 작성
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for out_idx, i in enumerate(kept_indices, start=1):
            rec = ds[i]
            img = rec["image"]
            prompt = rec.get("prompt") or ""

            fname = f"{out_idx:0{pad}d}.png"
            save_path = os.path.join(args.export_dir, fname)
            img.save(save_path)

            rel_path = os.path.join(args.relative_prefix, fname).replace("\\", "/")
            f.write(json.dumps({"image": rel_path, "caption1": prompt}, ensure_ascii=False) + "\n")

    print(f"[OK] wrote {k} unique-prompt samples to {args.out_jsonl} (from {end - start} scanned)")

if __name__ == "__main__":
    main()
