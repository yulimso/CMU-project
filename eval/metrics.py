from typing import List, Any, Dict, Tuple
import argparse, os, json, torch

# ---------- I/O ----------
def load_items_and_refs(jsonl_path, base_path=None):
    """Load image paths and (optional) reference captions from JSONL."""
    paths, refs = [], []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            img = obj["image"]
            paths.append(os.path.join(base_path or "", img))

            # collect captions: caption1..caption5 -> caption
            cap_keys = sorted([k for k in obj.keys() if k.startswith("caption")])
            if cap_keys:
                refs.append([obj[k] for k in cap_keys])
            elif "caption" in obj:
                refs.append([obj["caption"]])
            else:
                refs.append(None)
    return paths, refs

def load_gt_lines(gt_file):
    """Load one caption per line as GT."""
    with open(gt_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]

def build_inputs(tokenizer, image, prompt):
    """Prepare multimodal input."""
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    return tokenizer(image, input_text, add_special_tokens=False, return_tensors="pt")

def clean_pred(s: str) -> str:
    return s.strip()

def flatten_refs_first(refs: List[Any]) -> List[str]:
    """Use first reference for single-ref metrics; fill empty if none."""
    out = []
    for r in refs:
        if isinstance(r, list) and len(r) > 0:
            out.append(r[0])
        else:
            out.append("")
    return out

def align_len(a: List, b: List) -> Tuple[List, List]:
    n = min(len(a), len(b))
    return a[:n], b[:n]

def to_coco_dicts(preds: List[str], refs: List[Any]) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Convert parallel lists to COCO dicts for pycocoevalcap."""
    preds, refs = align_len(preds, refs)
    gts, res = {}, {}
    for i, (p, r) in enumerate(zip(preds, refs)):
        img_id = str(i)
        res[img_id] = [p]
        if isinstance(r, list):
            gts[img_id] = r
        else:
            gts[img_id] = [r]
    return gts, res

def eval_bleu_pyco(preds: List[str], refs: List[Any], n: int = 4) -> float:
    """BLEU-n via pycocoevalcap (expects multiple refs)."""
    from pycocoevalcap.bleu.bleu import Bleu
    gts, res = to_coco_dicts(preds, refs)
    score, _ = Bleu(n=n).compute_score(gts, res)
    return float(score[-1] if isinstance(score, (list, tuple)) else score)

def eval_rouge(preds: List[str], refs_single: List[str]) -> float:
    """Average ROUGE-L (F1) only."""
    from rouge_score import rouge_scorer
    preds, refs_single = align_len(preds, refs_single)
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    total = 0.0
    n = max(1, len(preds))
    for p, r in zip(preds, refs_single):
        score = scorer.score(r, p)
        total += score['rougeL'].fmeasure
    return total / n

def eval_cider(preds: List[str], refs: List[Any]) -> float:
    """CIDEr via pycocoevalcap."""
    from pycocoevalcap.cider.cider import Cider
    gts, res = to_coco_dicts(preds, refs)
    score, _ = Cider().compute_score(gts, res)
    return float(score if not isinstance(score, (list,tuple)) else score[0])

def eval_meteor_nltk(preds: List[str], refs_single: List[str]) -> float:
    """Average NLTK METEOR over pairs."""
    import nltk
    from nltk.translate.meteor_score import meteor_score
    # resources (safe to call repeatedly; no-op after first)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

    preds, refs_single = align_len(preds, refs_single)
    scores = [meteor_score([r.split()], p.split()) for p, r in zip(preds, refs_single)]
    return sum(scores)/len(scores) if scores else 0.0

def eval_clip_transformers(image_paths: List[str], captions: List[str], device: str) -> Tuple[List[float], float]:
    """CLIP similarity via transformers (no torch==1.7.1 constraint)."""
    from transformers import CLIPModel, CLIPProcessor
    from PIL import Image as _Image
    import torch as _torch

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    scores = []
    for img_path, cap in zip(image_paths, captions):
        inputs = proc(text=[cap], images=_Image.open(img_path).convert("RGB"), return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with _torch.no_grad():
            out = model(**inputs)
            img_emb = out.image_embeds
            txt_emb = out.text_embeds
            sim = _torch.nn.functional.cosine_similarity(txt_emb, img_emb)
            scores.append(sim.item())
    avg = sum(scores)/len(scores) if scores else 0.0
    return scores, avg