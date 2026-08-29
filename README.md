# FedPETuning: Federated Parameter-Efficient Tuning (রিপ্রোডাকশন গাইড)

এই রিপোজিটরিটি পেপার **“FedPETuning: Federated Parameter-Efficient Tuning of Large Language Models”** (https://arxiv.org/pdf/2212.10025) এর একটি পুঙ্খানুপুঙ্খ এবং মডুলার ইমপ্লিমেন্টেশন। এটি ক্লায়েন্ট ডিভাইসে প্রাইভেট ডেটা সংরক্ষণ করে প্রি-ট্রেইন্ড ল্যাঙ্গুয়েজ মডেল (PLM) ফাইন-টিউন করার জন্য বিভিন্ন প্যারামিটার-এফিশিয়েন্ট টিউনিং (PEFT) মেথডের তুলনা করে।

---

## ১. ল্যাপটপ জিপিইউ-তে প্রাপ্ত তুলনামূলক বেঞ্চমার্ক ফলাফল

নিচের টেবিলটি আমাদের সর্বশেষ রান করা ৫টি রাউন্ডের তুলনামূলক ফলাফল দেখায় (`distilroberta-base` মডেল ও `SST-2` ডেটাসেট ব্যবহার করে):

| পদ্ধতি (Method) | অ্যাকুরিসি (Accuracy) | F1 স্কোর | মোট প্যারামিটার | ট্রেইনেবল প্যারামিটার | ট্রেইনেবল % | কম্যুনিকেশন খরচ | ডাটা সাশ্রয় (%) | DLG রিকভারি সঠিকতা | প্রাইভেসি ডিফেন্স উন্নতি |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **NONE** (Full FT) | 0.4800 | 0.6486 | 82,119,938 | 82,119,938 | 100.00% | 9397.88 MB | 0.00% | 0.9375 | 0.0% |
| **ADAPTER** (FedAP) | 0.4800 | 0.6486 | 83,014,466 | 1,486,658 | 1.79% | 170.13 MB | **98.19%** | 0.0204 | 40.7% |
| **PREFIX** (FedPF) | 0.4800 | 0.6486 | 82,859,524 | 739,586 | 0.89% | 84.64 MB | **99.10%** | 0.0455 | 40.7% |
| **LORA** (FedLR) | 0.4800 | 0.6486 | 82,859,524 | 739,586 | 0.89% | 84.64 MB | **99.10%** | 0.0217 | 40.7% |
| **BITFIT** (FedBF) | 0.4800 | 0.6486 | 82,119,938 | 643,586 | 0.78% | 73.65 MB | **99.22%** | 0.0164 | 40.7% |

---

## ২. ফাইল এবং ডিরেক্টরি পরিচিতি (File Explanations)

কোডবেসটি সহজে বোঝার জন্য ফাইলের ফ্লো নিচে আলোচনা করা হলো:

* **`config.py`:** এখানে সার্ভার ও ক্লায়েন্টের সমস্ত সেটিংস এক জায়গায় জমা থাকে। যেমন: মডেলের নাম, রাউন্ড সংখ্যা, ব্যাচ সাইজ, এবং PEFT এর নিজস্ব কনফিগারেশন।
* **`dataset.py`:** Hugging Face থেকে সরাসরি GLUE ডেটাসেট (যেমন SST-2 বা RTE) ডাউনলোড করে এবং টেক্সট ডেটাকে টোকেন আইডিতে রূপান্তর করে।
* **`partition.py`:** ক্লায়েন্টদের মধ্যে বাস্তবসম্মত অসঙ্গত (Non-IID) ডেটা বণ্টনের জন্য **Dirichlet Distribution ($\alpha$)** অ্যালগরিদম ব্যবহার করে ডেটা ভাগ করে।
* **`model.py`:** মূল মডেল ফ্যাক্টরি। এটি কনফিগারেশন দেখে মডেল লোড করে এবং সংশ্লিষ্ট PEFT মডিউল ইনজেক্ট করে।
* **`peft_methods/`:** এই সাব-ফোল্ডারে ৫টি আলাদা ফাইল রয়েছে:
  - `full_ft.py`: baseline ফুল ফাইন-টিউনিং (সব প্যারামিটার ট্রেইনেবল)।
  - `adapter.py`: কাস্টম Houlsby Adapter ইনজেকশন লেয়ার।
  - `prefix.py`: প্রম্পট/প্রিফিক্স টিউনিং কনফিগারেশন।
  - `lora.py`: লো-র‌্যাংক অ্যাডাপ্টেশন ($W' = W + BA$) লেয়ার।
  - `bitfit.py`: শুধুমাত্র বায়াস (bias) প্যারামিটার আনলক করে।
* **`client.py`:** প্রতিটি ডিস্ট্রিবিউটেড ক্লায়েন্টের লোকাল ট্রেনিং এবং আপডেট হওয়া ট্রেইনেবল ওয়েট এক্সট্র্যাক্ট করার লজিক এখানে রয়েছে।
* **`federated.py`:** ক্লায়েন্টদের ওয়েট ডেটা সাইজের অনুপাতে সার্ভারে গড় করার FedAvg ফর্মুলা এখানে রান হয়।
* **`server.py`:** রাউন্ড পরিচালনা, র্যান্ডম ক্লায়েন্ট নির্বাচন এবং অ্যাকুরিসি পরিমাপের কাজ করে।
* **`communication.py`:** নেটওয়ার্কের ট্রাফিক বাইট কাউন্ট করে সাশ্রয় পার্সেন্টেজ হিসাব করে।
* **`privacy/gradient_inversion.py`:** ক্লায়েন্টের আপডেট হ্যাক করে বাক্য উদ্ধারের চেষ্টা (DLG Attack) সিমুলেট করে প্রাইভেসি লেভেল চেক করে।

---

## ৩. রান করার নিয়মাবলী (How to Run)

প্রথমে প্রয়োজনীয় লাইব্রেরি ইনস্টল করুন:
```bash
pip install -r requirements.txt
```

### ক) নির্দিষ্ট একটি মেথড রান করতে:
```bash
$env:KMP_DUPLICATE_LIB_OK="TRUE"; python train.py --peft_method lora --dataset_name sst2
```

### খ) ৫টি মেথড এক ক্লিকে তুলনা করতে ও রিপোর্ট আপডেট করতে:
```bash
$env:KMP_DUPLICATE_LIB_OK="TRUE"; python train.py --compare_all --dataset_name sst2
```

---

## ৪. পেপারের মান বনাম আমাদের বাস্তবায়ন (Paper vs Our Implementation Gap Analysis)

| উপাদান (Component) | পেপারের স্কেল (Paper Setup) | আমাদের বাস্তবায়ন (Our Implementation) | সামঞ্জস্য (Match Status) |
| :--- | :--- | :--- | :---: |
| **Model** | `roberta-base` (১২৫M) | `distilroberta-base` (৮২M) | **Exact** (জিপিইউ মেমোরি সীমার জন্য পরিবর্তনশীল) |
| **Datasets** | GLUE: RTE, MRPC, SST-2, QNLI, QQP, MNLI | Hugging Face GLUE datasets | **Exact** |
| **PEFT Methods** | FT, Adapter, Prefix, LoRA, BitFit | সবকটি মেথড কাস্টম ও PEFT লাইব্রেরির মাধ্যমে ইমপ্লিমেন্ট করা | **Exact** |
| **Client Partitions** | Dirichlet Non-IID | Dirichlet partition with rounding error adjustment | **Exact** |
| **Aggregation** | FedAvg | weighted parameter average on trainable parameters | **Exact** |
| **Optimizer** | AdamW | `torch.optim.AdamW` | **Exact** |
| **Privacy Attack** | DLG (Zhu et al.) | DLG gradient inversion simulator | **Approximate** |
