# ✍️ Text Generation with Markovify

A flexible Python script for generating synthetic text using **n-gram Markov chains**. Supports both **word-level** and **character-level** modeling, combining multiple text corpora (e.g., classic literature) into a unified generative model.

---

## ⚡ Quickstart

### 1. Installation
Install `markovify` and its dependencies directly inside your environment:

```bash
python -m pip install markovify
```

### 2. Required Data Files
Ensure your project directory contains the text sources used for training:
* **`1342-0.txt`** (e.g., *Pride and Prejudice*)
* **`84-0.txt`** (e.g., *Frankenstein*)

---

## ⚙️ Configuration Parameters

| Parameter | Type | Description | Default Value |
| :--- | :--- | :--- | :--- |
| `level` | `str` | Processing granularity: `"word"` or `"char"` | `"char"` |
| `order` | `int` | The $n$-gram length (higher values preserve more original structure) | `7` |
| `output_n` | `int` | Total number of generated lines/sentences | `14` |
| `weights` | `list` | Influence ratio between source models `[Text_A, Text_B]` | `[0.5, 0.5]` |
| `length_limit` | `int` | Maximum allowed character length per output sentence | `280` |

---

## 💻 Complete Implementation Code

```python
import sys
import markovify

# 1. Custom class for character-level modeling
class SentencesByChar(markovify.Text):
    def word_split(self, sentence):
        return list(sentence)
    def word_join(self, words):
        return "".join(words)

# 2. Load text corpora
try:
    text_a = open("1342-0.txt", encoding="utf-8").read()
    text_b = open("84-0.txt", encoding="utf-8").read()
    print("Text files loaded successfully.")
except FileNotFoundError:
    print("Error: One or both text files not found.")
    text_a, text_b = "", ""

# 3. Configure parameters
level = "char"
order = 7
output_n = 14
weights = [0.5, 0.5]
length_limit = 280

# 4. Generate text
if text_a and text_b:
    model_cls = markovify.Text if level == "word" else SentencesByChar
    
    # Instantiate individual models
    gen_a = model_cls(text_a, state_size=order)
    gen_b = model_cls(text_b, state_size=order)
    
    # Combine models with custom weighting
    gen_combo = markovify.combine([gen_a, gen_b], weights)
    
    print("\n--- GENERATED TEXT ---\n")
    for i in range(output_n):
        out = gen_combo.make_short_sentence(length_limit, test_output=False)
        if out:
            out = out.replace("\n", " ")
            print(f"{out}\n")
        else:
            print(f"Could not generate sentence {i+1} with current parameters.")
```

---

## 📝 Sample Outputs (Character-Level, Order = 7)
```
He walked up and donations more general Terms of Use parts former conduct in the honours us during what Charlotte and despair.

During my landing, I removed at first unable to allude to a family made me shuddered through the astonishment.

She soon lessening as any affront to Colonel Fitzwilliam seemed it cannot, which, in its contention, that was too poor Lydia returned from any symptoms?

I understand the Foundations and Chamounix; I took the old man.

Sir William and Ernest!

The old lady is she?

The compliment to hear the obliged to Miss Bingley.

The shutting that preceded my father; my imagined the rest, I resolved to revealed, laughing.

You throw them in the most intolerably sensibly, and have commission.

But in spite of those most improbable than many hours round.

I wish you equal she was impossible.

And now my blood boils at the names.

Great God!

My ancestors had never been very sorry for her; or, with rage as the improved, her head.
```
