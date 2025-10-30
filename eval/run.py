#!/usr/bin/env python3
import argparse, os, json, torch
from typing import List, Any, Dict, Tuple
from PIL import Image
from datetime import datetime
from zoneinfo import ZoneInfo
from transformers import AutoTokenizer
from unsloth import FastVisionModel
from metrics import (
    eval_bleu_pyco, eval_rouge, eval_cider, eval_meteor_nltk,
    eval_clip_transformers, load_items_and_refs, load_gt_lines,
    build_inputs, clean_pred, flatten_refs_first, align_len
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="JSONL with images and (optional) captions.")
    ap.add_argument("--model_name", required=True, default="unsloth/Llama-3.2-11B-Vision-Instruct")
    # Save dir fixed to requested path; filename will be timestamp like MMDD-HHMM.json
    ap.add_argument("--save_dir", required=True, default="../results/llama32-11b",
                    help="Directory to save caption results (filename is auto timestamp).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--min_p", type=float, default=0.08)
    ap.add_argument("--prompt", default="Describe this image in one concise sentence.")
    ap.add_argument("--load_in_4bit", action="store_true", default=True)
    ap.add_argument("--eval_metric", default="bleu",
                    choices=["bleu","rouge","cider","meteor","clip","all"],
                    help="Choose evaluation metric.")
    ap.add_argument("--gt_file", default=None,
                    help="Optional path to GT captions file (one per line). Overrides JSONL refs.")
    args = ap.parse_args()

    # Build output path: /home/.../results/llama32-11b/MMDD-HHMM.json (America/Los_Angeles)
    os.makedirs(args.save_dir, exist_ok=True)
    ts = datetime.now(ZoneInfo("America/New_York")).strftime("%m%d-%H%M")
    out_path = os.path.join(args.save_dir, f"{ts}.json")

    # Load items and (optional) references
    paths, refs_from_jsonl = load_items_and_refs(args.jsonl)

    # Load model
    model, tokenizer = FastVisionModel.from_pretrained(
        args.model_name,
        load_in_4bit=bool(args.load_in_4bit),
        use_gradient_checkpointing="unsloth"
    )
    FastVisionModel.for_inference(model)
    model.to(args.device)

    # Generate captions
    captions = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        inputs = build_inputs(tokenizer, img, args.prompt).to(args.device)
        with torch.no_grad():
            toks = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                temperature=args.temperature,
                min_p=args.min_p
            )
        txt = tokenizer.decode(toks[0], skip_special_tokens=True)
        if "assistant\n\n" in txt:
            txt = txt.split("assistant\n\n", 1)[-1]
        captions.append(clean_pred(txt))

    # Save predictions (to timestamped file)
    output = [{"image_path": path, "generated_caption": cap} for path, cap in zip(paths, captions)]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)
    print(f"[OK] Generated captions saved to {out_path}")

    # Build GT (single-ref list and multi-ref for COCO metrics)
    if args.gt_file:
        gts_single = load_gt_lines(args.gt_file)
        gts_single, captions = align_len(gts_single, captions)
        refs_for_coco = gts_single  # each as single string
    else:
        gts_single = flatten_refs_first(refs_from_jsonl)   # for ROUGE/METEOR
        gts_single, captions = align_len(gts_single, captions)
        refs_for_coco = []
        for r in refs_from_jsonl[:len(captions)]:          # for BLEU/CIDEr (COCO style)
            if isinstance(r, list) and r:
                refs_for_coco.append(r)
            else:
                refs_for_coco.append("")  # fallback empty string

    # Evaluation
    need_gt = args.eval_metric in {"bleu","rouge","cider","meteor","all"}
    if need_gt and not any(len(x.strip()) for x in gts_single):
        raise RuntimeError("Ground-truth captions are required but not provided. "
                           "Add captions to JSONL or pass --gt_file.")

    if args.eval_metric == "bleu":
        score = eval_bleu_pyco(captions, refs_for_coco)
        print(f"BLEU: {score:.4f}")

    elif args.eval_metric == "rouge":
        r = eval_rouge(captions, gts_single)
        print(f"ROUGE: {r}")

    elif args.eval_metric == "cider":
        c = eval_cider(captions, refs_for_coco)
        print(f"CIDEr: {c:.4f}")

    elif args.eval_metric == "meteor":
        m = eval_meteor_nltk(captions, gts_single)
        print(f"METEOR: {m:.4f}")

    elif args.eval_metric == "clip":
        scores, avg = eval_clip_transformers(paths, captions, args.device)
        print(f"CLIP Scores (per-sample): {scores}")
        print(f"CLIP Average: {avg:.4f}")

    elif args.eval_metric == "all":
        b = eval_bleu_pyco(captions, refs_for_coco)
        r = eval_rouge(captions, gts_single)
        c = eval_cider(captions, refs_for_coco)
        m = eval_meteor_nltk(captions, gts_single)
        scores, avg = eval_clip_transformers(paths, captions, args.device)
        print(f"BLEU: {b:.4f}")
        print(f"ROUGE: {r}")
        print(f"CIDEr: {c:.4f}")
        print(f"METEOR: {m:.4f}")
        print(f"CLIP Average: {avg:.4f}")

if __name__ == "__main__":
    main()