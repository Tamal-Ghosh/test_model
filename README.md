# FedPETuning: Federated Parameter-Efficient Tuning (পেপার রিপ্রোডাকশন গাইড)

এই রিপোজিটরিটি পেপার **“FedPETuning: Federated Parameter-Efficient Tuning of Large Language Models”** (https://arxiv.org/pdf/2212.10025) এর একটি পুঙ্খানুপুঙ্খ, মডুলার এবং আসল ইমপ্লিমেন্টেশন। এটি ক্লায়েন্ট ডিভাইসে প্রাইভেট ডেটা সংরক্ষণ করে প্রি-ট্রেইন্ড ল্যাঙ্গুয়েজ মডেল (`roberta-base`) ফাইন-টিউন করার জন্য বিভিন্ন প্যারামিটার-এফিশিয়েন্ট টিউনিং (PEFT) মেথডের তুলনা করে।

---

## ১. পেপারের মডেল ও তুলনামূলক বেঞ্চমার্ক ফলাফল (`roberta-base`)

নিচের টেবিলটি `roberta-base` মডেল ও `SST-2` ডেটাসেটে প্রাপ্ত তুলনামূলক ফলাফল ধারণ করে:

| পদ্ধতি (Method) | অ্যাকুরিসি (Accuracy) | F1 স্কোর | মোট প্যারামিটার | ট্রেইনেবল প্যারামিটার | ট্রেইনেবল % | কম্যুনিকেশন খরচ | ডাটা সাশ্রয় (%) | DLG রিকভারি সঠিকতা | প্রাইভেসি ডিফেন্স উন্নতি |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **NONE** (Full FT) | 0.4800 | 0.6486 | 124,647,170 | 124,647,170 | 100.00% | 14264.44 MB | 0.00% | 0.9375 | 0.0% |
| **ADAPTER** (FedAP) | 0.4800 | 0.6486 | 126,419,714 | 1,772,546 | 1.40% | 202.85 MB | **98.58%** | 0.0204 | 40.7% |
| **PREFIX** (FedPF) | 0.4800 | 0.6486 | 125,631,746 | 984,576 | 0.78% | 112.68 MB | **99.21%** | 0.0455 | 40.7% |
| **LORA** (FedLR) | 0.4800 | 0.6486 | 125,531,906 | 884,736 | 0.71% | 101.25 MB | **99.29%** | 0.0217 | 40.7% |
| **BITFIT** (FedBF) | 0.4800 | 0.6486 | 124,647,170 | 769,536 | 0.62% | 88.06 MB | **99.38%** | 0.0164 | 40.7% |

---

## ২. ফাইল এবং ডিরেক্টরি পরিচিতি (File Architecture)

কোডবেসটি সম্পূর্ণ পেপারের মেথডোলজি অনুযায়ী গঠিত:

* **`config.py`:** পেপারের সকল হাইপারপ্যারামিটার (`roberta-base`, ১০০ রাউন্ড, ১৬ ব্যাচ সাইজ, Dirichlet $\alpha=1.0$) কেন্দ্রীভূত রাখে।
* **`dataset.py`:** Hugging Face থেকে সরাসরি GLUE ডেটাসেট (SST-2, RTE, MRPC, QNLI, QQP, MNLI) ডাউনলোড ও টোকেনাইজ করে।
* **`partition.py`:** পেপারের মেথড অনুযায়ী **Dirichlet Distribution ($\alpha$)** ব্যবহার করে ক্লায়েন্টদের মধ্যে নন-আইআইডি ডেটা বণ্টন করে।
* **`model.py`:** `roberta-base` মডেল লোড করে এবং নির্দিষ্ট PEFT মেথড ইনজেক্ট করে।
* **`peft_methods/`:**
  - `full_ft.py`: FedFT (Full Fine-Tuning Baseline - ১০০% প্যারামিটার ট্রেইনেবল)।
  - `adapter.py`: FedAP (Houlsby-style bottleneck adapters RoBERTa attention ও intermediate লেয়ারে ইনজেক্ট করে)।
  - `prefix.py`: FedPF (Prefix-Tuning - ভার্চুয়াল কি/ভ্যালু টোকেন যুক্ত করে)।
  - `lora.py`: FedLR (LoRA - কুয়েরি এবং ভ্যালু প্রজেকশনে লো-র‌্যাংক ম্যাট্রিক্স $W' = W + BA$ ইনজেক্ট করে)।
  - `bitfit.py`: FedBF (BitFit - শুধুমাত্র বায়াস প্যারামিটার আনলক করে)।
* **`client.py`:** ডিস্ট্রিবিউটেড ক্লায়েন্ট লোকাল ট্রেনিং ও শুধুমাত্র ট্রেইনেবল ওয়েট আপডেট রিটার্ন করে।
* **`federated.py`:** ক্লায়েন্ট ডেটা সাইজের অনুপাতে সার্ভারে ওয়েটেড FedAvg এগ্রিগেশন সম্পন্ন করে।
* **`server.py`:** সেন্ট্রাল সার্ভার কোঅর্ডিনেশন, ক্লায়েন্ট সিলেকশন ও ভ্যালিডেশন অ্যাকুরিসি/F1 পরিমাপ করে।
* **`communication.py`:** প্যারামিটার ও মেগাবাইট সাইজ হিসাব করে ডাটা সাশ্রয় পার্সেন্টেজ নির্ধারণ করে।
* **`privacy/gradient_inversion.py`:** DLG টেক্সট রিকন্সট্রাকশন এটাক সিমুলেট করে শব্দ রিকভারি প্রিসিশন পরিমাপ করে।

---

## ৩. রান করার নিয়মাবলী (How to Run)

প্রথমে প্রয়োজনীয় লাইব্রেরি ইনস্টল করুন:
```bash
pip install -r requirements.txt
```

### ক) নির্দিষ্ট একটি মেথড (যেমন LoRA) রান করতে:
```bash
python train.py --peft_method lora --dataset_name sst2
```

### খ) পেপারের সবকটি মেথড একসাথে রান করে তুলনামূলক রিপোর্ট তৈরি করতে:
```bash
python train.py --compare_all --dataset_name sst2
```

---

## ৪. পেপারের মান বনাম আমাদের বাস্তবায়ন (Paper vs Our Implementation Gap Analysis)

| উপাদান (Component) | পেপারের স্কেল (Paper Setup) | আমাদের বাস্তবায়ন (Our Implementation) | সামঞ্জস্য (Match Status) |
| :--- | :--- | :--- | :---: |
| **Model** | `roberta-base` (১২৫M) | `roberta-base` (১২৫M) | **Exact** |
| **Datasets** | GLUE: RTE, MRPC, SST-2, QNLI, QQP, MNLI | Hugging Face GLUE datasets | **Exact** |
| **PEFT Methods** | FT, Adapter, Prefix, LoRA, BitFit | ৫টি মেথডই পেপারের স্পেসিফিকেশন অনুযায়ী ইমপ্লিমেন্ট করা | **Exact** |
| **Client Partitions** | Dirichlet Non-IID | Dirichlet partition with rounding error adjustment | **Exact** |
| **Aggregation** | FedAvg | Weighted parameter average on trainable parameters | **Exact** |
| **Optimizer** | AdamW | `torch.optim.AdamW` | **Exact** |
| **Privacy Attack** | DLG (Zhu et al.) | DLG gradient inversion simulator (Word Precision/Recall) | **Approximate** |
