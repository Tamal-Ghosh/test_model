# Federated PEFT: Benchmarking Parameter-Efficient Fine-Tuning under Client Heterogeneity

এই রিপোজিটরিটি ফেডারেটেড লার্নিং (FL) পরিবেশে প্রি-ট্রেইন্ড ল্যাঙ্গুয়েজ মডেল (`roberta-base`) এবং **AG News** ডেটাসেটের ওপর ৪টি প্রধান **প্যারামিটার-এফিশিয়েন্ট ফাইন-টিউনিং (PEFT)** মেথডের পারফরম্যান্স এবং ডেটা হেটারোজেনিটি (Non-IID) সহনশীলতা মূল্যায়নের জন্য একটি সম্পূর্ণ ও মডুলার ফ্রেমওয়ার্ক।

---

## ১. এক্সপেরিমেন্টাল ম্যাট্রিক্স (Experimental Design)

```
                 ৪টি PEFT METHODS
          ┌──────┬──────┬──────┬──────┐
          │ LoRA │Adapt │Prefix│ IA³  │
          └──────┴──────┴──────┴──────┘
                    ×
             ৩টি PARAMETER BUDGETS
             ┌─────┬─────┬─────┐
             │100K │500K │ 1M  │
             └─────┴─────┴─────┘
                    ×
          ৪টি HETEROGENEITY LEVELS
          ┌─────┬─────┬─────┬──────┐
          │ 1.0 │ 0.5 │ 0.1 │ 0.01 │
          └─────┴─────┴─────┴──────┘
                  = ৪৮টি Configurations
                    × ৩টি Random Seeds (42, 43, 44)
                  = ১৪৪টি Training Runs
```

* **Pretrained Model:** `roberta-base` (১২টি লেয়ার, ৭৬৮ হিডেন ডাইমেনশন, মোট ১২৪.১৭ মিলিয়ন প্যারামিটার)
* **Dataset:** AG News (৪টি ক্লাস: World, Sports, Business, Sci/Tech)
* **Clients:** ১০ জন ক্লায়েন্ট (১০০% পার্টিসিপেশন, Cross-Silo)
* **FL Algorithm:** FedAvg (স্যাম্পল সাইজের অনুপাতে ওয়েটেড প্যারামিটার অ্যাগ্রিগেশন)
* **FL Rounds:** ৫০ রাউন্ডস
* **Non-IID Partition:** Dirichlet Distribution ($\alpha \in [1.0, 0.5, 0.1, 0.01]$)

---

## ২. প্যারামিটার বাজেট ক্যালিব্রেশন (Strict Budget Matching)

| PEFT Method | Low Budget (~100K Target) | Medium Budget (~500K Target) | High Budget (~1M Target) |
|---|---|---|---|
| **LoRA** | **113,668** (~113K) <br> *(rank $r=3$)* | **519,172** (~519K) <br> *(rank $r=14$)* | **998,404** (~998.4K) <br> *(rank $r=27$)* |
| **Adapter** (Houlsby) | **95,284** (~95K) <br> *(bottleneck $m=2$)* | **501,052** (~501K) <br> *(bottleneck $m=13$)* | **1,017,484** (~1.01M) <br> *(bottleneck $m=27$)* |
| **Prefix-Tuning** | **95,236** (~95K) <br> *(tokens $l=5$)* | **500,740** (~500K) <br> *(tokens $l=27$)* | **998,404** (~998.4K) <br> *(tokens $l=54$)* |
| **$\text{IA}^3$** | **76,804** (~77K) <br> *(full arch vectors)* | **76,804** (~77K) <br> *(architectural cap)* | **76,804** (~77K) <br> *(architectural cap)* |

> **Lightweight Head নোট:** RoBERTa-র সাধারণ ক্লাসিফায়ার হেডে প্রায় ৫৯৪K প্যারামিটার থাকে, যা লো-বাজেটকে নষ্ট করে দেয়। তাই আমরা একটি একক লিনিয়ার ক্লাসিফায়ার হেড ($768 \to 4 = \mathbf{3,076}$ params) ব্যবহার করেছি, যাতে বাজেটের প্রায় সম্পূর্ণ অংশ PEFT মডিউলের জন্য সংরক্ষিত থাকে।

---

## ৩. আর্কিটেকচার লেভেল থেকে প্যারামিটার সূত্র

1. **LoRA Formula:**
   $$N(r) = 12 \text{ layers} \times 2 \text{ matrices (q, v)} \times (2 \times 768 \times r) + 3,076 = 36,864 \times r + 3,076$$
2. **Bottleneck Adapter Formula (Houlsby):**
   $$N(m) = 24 \text{ adapters} \times (1,537 \times m + 768) + 3,076 = 36,888 \times m + 18,432 + 3,076$$
3. **Prefix-Tuning Formula:**
   $$N(l) = 2 \text{ (key/val)} \times 12 \text{ layers} \times 768 \times l + 3,076 = 18,432 \times l + 3,076$$
4. **$\text{IA}^3$ Formula:**
   $$N_{\text{IA}^3} = 12 \text{ layers} \times (768_{\text{k}} + 768_{\text{v}} + 768_{\text{dense}} + 768_{\text{dense}} + 3072_{\text{out}}) + 3,076 = \mathbf{76,804}$$

---

## ৪. ফাইল এবং ডিরেক্টরি পরিচিতি

* **`config.py`:** কেন্দ্রীয় কনফিগারেশন ক্লাস (`FLConfig`)।
* **`dataset.py`:** AG News ডেটাসেট ডাউনলোড, টোকেনাইজেশন ও PyTorch ডাটাসেট র‍্যাপার।
* **`partition.py`:** সিড-ভিত্তিক Dirichlet Non-IID ডেটা ডিস্ট্রিবিউটর।
* **`model.py`:** RoBERTa-base মডেল লোডার, Lightweight হেড সংযোজন ও বাজেট ক্যালিব্রেটর।
* **`peft_methods/`:**
  - `lora.py`: Low-Rank Adaptation (LoRA) কনফিগারেশন।
  - `adapter.py`: Houlsby Bottleneck Adapter র‍্যাপার ও ইনজেক্টর।
  - `prefix.py`: Prefix-Tuning কনফিগারেশন।
  - `ia3.py`: $\text{IA}^3$ ভেক্টর স্কেলিং কনফিগারেশন।
* **`client.py`:** ক্লায়েন্ট লোকাল ট্রেনিং লুপ ও শুধুমাত্র ট্রেইনেবল ওয়েট আপডেট রিটার্ন।
* **`server.py`:** সার্ভার কোঅর্ডিনেশন, FedAvg অ্যাগ্রিগেশন ও ক্লায়েন্ট-ওয়াইজ ইভ্যালুয়েশন।
* **`federated.py`:** ক্লায়েন্ট ডেটা সাইজের অনুপাতে ওয়েটেড FedAvg অ্যাগ্রিগেটর।
* **`logger.py`:** রাউন্ড-বাই-রাউন্ড Global Acc, F1, Loss ও Client stats CSV এবং JSON-এ সংরক্ষণ।
* **`run_experiments.py`:** অটোমেটিক ব্যাচ এক্সপেরিমেন্ট রানার (Resume ও Auto-Zip সুবিধা সহ)।
* **`plot_results.py`:** পেপারের Convergence গ্রাফ, Robustness ড্রপ কার্ভ এবং LaTeX সামারি টেবিল জেনারেটর।

---

## ৫. ইনস্টলেশন ও সেটআপ

```bash
git clone https://github.com/Tamal-Ghosh/test_model.git
cd test_model
pip install -r requirements.txt
pip install transformers datasets peft accelerate evaluate scikit-learn matplotlib seaborn
```

---

## ৬. এক্সপেরিমেন্ট চালানোর নিয়ম

### ক. দ্রুত ভেরিফিকেশন টেস্ট রান (২ মিনিট):
```bash
python -u run_experiments.py --methods lora --budgets low --alphas 1.0 --seeds 42 --rounds 2 --max_train_samples 200 --max_val_samples 100 --output_dir results_test
```

### ক. ফুল ডেটাসেটে মূল পেপার রান (সম্পূর্ণ ১,২০,০০০ স্যাম্পল):
ডিফল্টভাবে কোনো স্যাম্পল লিমিট না দিলে সম্পূর্ণ **১,২০,০০০ ট্রেইনিং স্যাম্পল** এবং **৭,৬০০ ভ্যালিডেশন স্যাম্পল** ব্যবহার হবে:
```bash
python -u run_experiments.py \
    --methods lora adapter prefix ia3 \
    --budgets low \
    --alphas 1.0 0.5 0.1 0.01 \
    --seeds 42 43 44 \
    --rounds 50 \
    --output_dir results_low
```

### খ. সম্পূর্ণ ৪৮ কনফিগারেশন (সব বাজেট, সব মেথড ও ফুল ডেটাসেট):
```bash
python -u run_experiments.py \
    --methods all \
    --budgets all \
    --alphas 1.0 0.5 0.1 0.01 \
    --seeds 42 43 44 \
    --rounds 50 \
    --output_dir results
```

### গ. ঐচ্ছিক: দ্রুত ভেরিফিকেশন বা ফাস্ট রান (Subsampled):
```bash
# ২ মিনিটের কুইক টেস্ট
python -u run_experiments.py --methods lora --budgets low --alphas 1.0 --seeds 42 --rounds 2 --max_train_samples 200 --max_val_samples 100 --output_dir results_test

# ফাস্ট রান (১৫,০০০ স্যাম্পলে)
python -u run_experiments.py --methods lora adapter prefix ia3 --budgets low --alphas 1.0 0.5 0.1 0.01 --seeds 42 --rounds 50 --max_train_samples 15000 --max_val_samples 1000 --output_dir results_low_fast
```

> **Resume সুবিধা:** মাঝপথে রান বন্ধ হলেও আগের সফল রানগুলো নিজে থেকেই স্কিপ হয়ে পরবর্তীগুলো শুরু হবে।

---

## ৭. পেপারের গ্রাফ ও LaTeX টেবিল তৈরি করা

এক্সপেরিমেন্ট শেষে মাত্র একটি কমান্ড দিন:
```bash
python plot_results.py --results_dir results_low --figures_dir results_low/figures
```

এটি স্বয়ংক্রিয়ভাবে নিচের ফাইলগুলো তৈরি করবে:
1. **`convergence_curves.png`:** প্রতি রাউন্ডে গ্লোবাল অ্যাকুরেসির বৃদ্ধি (৪টি মেথডের তুলনামূলক রেখাচিত্র)।
2. **`robustness_analysis.png`:** Non-IID বৃদ্ধির সাথে সাথে ($\alpha=1.0 \to 0.01$) কার অ্যাকুরেসি কতটা ড্রপ করে (Robustness Analysis)।
3. **`summary_table.md` ও `summary_table.tex`:** ৩টি সিডের গড় ও স্ট্যান্ডার্ড ডেভিয়েশন ($\text{Mean} \pm \text{Std}$) এবং সরাসরি পেপারের LaTeX-এ যুক্ত করার মতো ফর্ম্যাটেড কোড:
$$\text{Performance Drop} = \text{Accuracy}_{\alpha=1.0} - \text{Accuracy}_{\alpha=0.01}$$
