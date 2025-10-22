# 🧠 CMU-Project: TBU

TBU (description).

---

## 📁 Repository Structure

```
CMU-project/
├─ data/
│  ├─ images/                # input images (may contain subfolders like sb/, yl/)
│  └─ captions.jsonl         # JSONL file with image paths and reference captions
│
├─ eval/
│  ├─ metrics.py             # evaluation utilities (BLEU, ROUGE, etc.)
│  └─ run_llama32.py         # main evaluation script
│
├─ results/
│  ├─ llama32-11b/           # auto-generated results for LLaMA-3.2-Vision
│  │  ├─ 1021-2129.json
│  │  └─ 1021-2149.json
│  └─ llava-7b/              
│
├─ scripts/
│  └─ run_llama32.sh         # shell script wrapper for run_llama32.py
│
├─ environment.yml           # conda environment definition
├─ requirements.txt          # pip requirements (identical dependencies)
└─ README.md                 # this documentation
```

---

## ⚙️ Environment Setup

### 1. Create the conda environment
```bash
conda env create -f environment.yml
conda activate unsloth_env
```


---

## 🧩 Input Format

### `data/captions.jsonl`
Each line represents one image and its reference captions.

```json
{"image": "yl/00001.png", "caption1": "A cat gives food to a man."}
{"image": "sb/00001.png", "caption1": "The dog takes a person for a walk."}
```

*Paths inside `"image"` must be relative to `data/images/`.*

---

## 🚀 Running Evaluation

### ✅ Option 1: Quick Run (recommended)
Use the shell script that activates the environment and runs evaluation automatically.

```bash
bash scripts/run_llama32.sh
bash scripts/run_llava7b.sh 
```

**What it does:**
1. Activates the conda environment `unsloth_env`
2. Calls the Python script `eval/run_llama32.py` or `eval/run_llava7b.py`
3. Saves generated captions to  
   `results/llama32-11b/<MMDD-HHMM>.json` or `results/llava-7b/<MMDD-HHMM>.json`
4. Prints selected metric scores in the terminal

Example output:
```
[OK] Generated captions saved to ../results/llama32-11b/1022-2337.json
BLEU: 0.4382
```

---

### ⚡ Option 2: Manual Python Execution
If you prefer direct control, run:
```bash
python3 eval/run_llama32.py \
  --jsonl data/captions.jsonl \
  --model_name unsloth/Llama-3.2-11B-Vision-Instruct \
  --save_dir results/llama32-11b \
  --eval_metric all
```

Available metrics:  
`bleu`, `rouge`, `cider`, `meteor`, `clip`, `all`

---

## 🧾 Output Files

Each run produces a timestamped JSON file:

```
results/llama32-11b/
├─ 1022-2337.json      
└─ 1022-2350.json     
```

Example content:
```json
[
   {
        "image_path": "../data/images/yl/00001.png",
        "generated_caption": "This image depicts a man sitting on the floor with a bowl of cat food, and a cat standing on its hind legs and holding the bowl with one paw, as if to beg for food."
    },
    {
        "image_path": "../data/images/yl/00002.png",
        "generated_caption": "The image depicts a man swimming underwater with a turtle perched on his back, showcasing a unique and intriguing scene."
    }
]
```

---

## 🧮 Evaluation Metrics

| Metric | Description |
|--------|--------------|
| **BLEU** | Measures n-gram precision against reference captions |
| **ROUGE-L** | Longest common subsequence overlap |
| **CIDEr** | Consensus-based metric used in COCO Captions |
| **METEOR** | Considers synonyms and word alignment |
| **CLIP Score** | Vision-language embedding similarity |

All metrics are implemented in `eval/metrics.py`.

---

## 🧪 Example Workflow

```bash
# Step 1: Activate environment
conda activate unsloth_env

# Step 2: Run evaluation
bash scripts/run_llama32.sh data/captions.jsonl

# Step 3: Inspect output
cat results/llama32-11b/<timestamp>.json
```

---

## 📈 Result Summary Template

When reporting results, you can log metrics like this:

| Model | BLEU-4 | ROUGE-L | CIDEr | METEOR | CLIP Avg |
|-------|--------|----------|--------|---------|-----------|
| LLaMA 3.2 11B Vision | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

---

## 👩‍💻 Author

*Carnegie Mellon University × IITP Program*

📧 seungbel@andrew.cmu.edu
**Seungbeen Lee**  
📧 subeenp@andrew.cmu.edu
**Subeen Park**  
📧 yujinle2@andrew.cmu.edu
**Yujin Lee**  
📧 yulims@andrew.cmu.edu
**Yulim So**  

---
