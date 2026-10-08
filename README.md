# Lab-4.3-LSTMs-GRUs-on-Text

## 📌 Project Overview

This lab demonstrates how to build a **character-level text generation model** using PyTorch.

The project covers:

* Character-level text preprocessing
* Building a character vocabulary
* Converting characters into numerical IDs
* Creating input and target sequences
* Building an LSTM model
* Building a GRU model
* Training an LSTM using Cross-Entropy Loss
* Generating new text from a trained model

The main model used in this implementation is a **Character-Level LSTM**.

---

## 🎯 Objectives

The objectives of this lab are to:

1. Understand character-level text representation.
2. Create a vocabulary from raw text.
3. Convert characters into integer IDs.
4. Prepare sequential training data.
5. Understand how LSTM and GRU networks process sequences.
6. Train an LSTM text-generation model.
7. Generate text using the trained model.

---

## 🧠 Technologies Used

* **Python**
* **PyTorch**
* **LSTM**
* **GRU**
* **Word/Character Embeddings**
* **Cross-Entropy Loss**
* **Adam Optimizer**

---

## 📂 Project Structure

```text
Lab-4.3-LSTMs-GRUs-on-Text/
│
├── lab_4_3.py
└── README.md
```

---

# 1. Dataset Preparation

The example uses a small text dataset:

```python
text = "hello world this is a character level lstm example"
```

Since this is a character-level model, every unique character becomes part of the vocabulary.

```python
chars = sorted(set(text))
```

For example, the vocabulary contains characters such as:

```text
a, c, e, h, i, l, m, o, r, s, t, ...
```

The character vocabulary is converted into numerical IDs:

```python
char_to_idx = {ch: i for i, ch in enumerate(chars)}
```

This allows the neural network to process characters as numbers.

---

# 2. Converting Text to Tensor

The complete text is converted into integer IDs:

```python
seq = torch.tensor(
    [char_to_idx[ch] for ch in text],
    dtype=torch.long
)
```

`torch.long` is important because PyTorch's `nn.Embedding` layer requires integer indices.

---

# 3. Creating Training Sequences

The sequence length is set to:

```python
seq_len = 25
```

The dataset is divided into overlapping input and target sequences.

```python
for i in range(len(seq) - seq_len):
    X.append(seq[i:i + seq_len])
    Y.append(seq[i + 1:i + seq_len + 1])
```

The target sequence is shifted by one character.

For example:

```text
Input:
hello

Target:
ello
```

Conceptually, the model learns:

```text
Current character/sequence → Next character/sequence
```

The resulting lists are converted into tensors:

```python
X = torch.stack(X)
Y = torch.stack(Y)
```

---

# 4. LSTM Model

The first model is a character-level LSTM.

```python
class CharLSTM(nn.Module):
    def __init__(self, vocab, hidden=128):
        super().__init__()

        self.emb = nn.Embedding(vocab, 32)
        self.lstm = nn.LSTM(32, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, vocab)

    def forward(self, x):
        o, _ = self.lstm(self.emb(x))
        return self.fc(o)
```

### Model Components

### Embedding

```python
self.emb = nn.Embedding(vocab, 32)
```

Converts each character ID into a 32-dimensional vector.

### LSTM

```python
self.lstm = nn.LSTM(32, hidden, batch_first=True)
```

The LSTM processes the character sequence and maintains information from previous time steps.

The hidden size is:

```text
128
```

### Fully Connected Layer

```python
self.fc = nn.Linear(hidden, vocab)
```

Converts the LSTM output into scores for every character in the vocabulary.

---

# 5. GRU Model

The lab also defines a GRU model:

```python
class CharGRU(nn.Module):
    def __init__(self, vocab, hidden=128):
        super().__init__()

        self.emb = nn.Embedding(vocab, 32)
        self.gru = nn.GRU(32, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, vocab)

    def forward(self, x):
        o, _ = self.gru(self.emb(x))
        return self.fc(o)
```

The GRU follows a similar architecture to the LSTM but uses a GRU recurrent layer instead.

### LSTM vs GRU

| Feature                | LSTM                  | GRU           |
| ---------------------- | --------------------- | ------------- |
| Memory mechanism       | More complex          | Simpler       |
| Gates                  | Input, forget, output | Update, reset |
| Parameters             | More                  | Fewer         |
| Training               | Can be slower         | Often faster  |
| Long-term dependencies | Strong                | Strong        |
| Implementation         | More complex          | Simpler       |

Although both models are defined in this lab, the current training code trains the **LSTM model**.

---

# 6. Creating the Model

The LSTM is initialized using the vocabulary size:

```python
model = CharLSTM(len(chars))
```

The Adam optimizer is used:

```python
opt = torch.optim.Adam(
    model.parameters(),
    lr=1e-2
)
```

The learning rate is:

```text
0.01
```

---

# 7. Training Loop

The model is trained for five epochs:

```python
for ep in range(5):

    logits = model(X)

    loss = F.cross_entropy(
        logits.reshape(-1, len(chars)),
        Y.reshape(-1)
    )

    opt.zero_grad()
    loss.backward()
    opt.step()

    print("epoch", ep, "loss", loss.item())
```

### Training Process

Each epoch performs the following steps:

1. Pass `X` through the model.
2. Generate predictions.
3. Calculate Cross-Entropy Loss.
4. Clear previous gradients.
5. Perform backpropagation.
6. Update model parameters.
7. Display the loss.

---

# 8. Loss Function

The project uses:

```python
F.cross_entropy()
```

Cross-Entropy Loss is suitable because the model predicts one character from multiple possible characters in the vocabulary.

The output is reshaped:

```python
logits.reshape(-1, len(chars))
```

The targets are also flattened:

```python
Y.reshape(-1)
```

This allows PyTorch to compare each predicted character with the correct target character.

---

# 9. Character Mappings for Generation

Two dictionaries are created:

```python
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
```

### `stoi`

Means:

```text
String → Integer
```

Example:

```text
"h" → 3
```

### `itos`

Means:

```text
Integer → String
```

Example:

```text
3 → "h"
```

These mappings are required during text generation.

---

# 10. Text Generation

The `generate()` function generates new characters from a starting character.

```python
def generate(model, start="r", length=50):
```

The default starting character is:

```text
r
```

and the model generates 50 additional characters.

The starting character is converted into an ID:

```python
idx = torch.tensor(
    [[stoi[ch] for ch in start]],
    dtype=torch.long
)
```

The character is passed through the embedding and LSTM:

```python
emb = model.emb(idx)
logits, _ = model.lstm(emb)
```

The final output is passed through the fully connected layer:

```python
model.fc(logits[:, -1, :])
```

The character with the highest predicted probability is selected:

```python
nxt = torch.argmax(...).item()
```

The predicted character is then added to the output:

```python
out += itos[nxt]
```

The process repeats until the requested number of characters has been generated.

---

# 11. Running the Project

Install PyTorch if it is not already installed:

```bash
pip install torch
```

Run the Python file:

```bash
python lab_4_3.py
```

Or run the cells in:

* Jupyter Notebook
* Google Colab
* VS Code

---

# 12. Expected Output

During training, you should see output similar to:

```text
Vocabulary size: 17
X shape: torch.Size([30, 25])
Y shape: torch.Size([30, 25])

epoch 0 loss ...
epoch 1 loss ...
epoch 2 loss ...
epoch 3 loss ...
epoch 4 loss ...
```

The exact loss values can vary depending on the dataset and model initialization.

After training, the generation function produces text such as:

```text
hello world this is a character...
```

Because the training dataset is very small and the model is trained for only five epochs, the generated text may not be grammatically meaningful.

---

# 13. Important Notes

### Small Dataset

The project uses a very small text sample:

```python
text = "hello world this is a character level lstm example"
```

Therefore, the model has very limited information to learn from.

### Short Training

Only five epochs are used:

```python
for ep in range(5):
```

Increasing the number of epochs can improve memorization of the training text, but it can also lead to overfitting.

### Character-Level Model

This is a **character-level** model rather than a word-level model.

For example:

```text
hello
```

is processed as:

```text
h → e → l → l → o
```

rather than treating `"hello"` as one token.

### Starting Character

The generation function uses:

```python
start="r"
```

The character `"r"` must exist in `chars`.

Otherwise, Python will raise:

```text
KeyError: 'r'
```

---

# 14. LSTM Architecture

The overall architecture is:

```text
Raw Text
    ↓
Character Vocabulary
    ↓
Character IDs
    ↓
Input Sequences
    ↓
Embedding Layer
    ↓
LSTM
    ↓
Fully Connected Layer
    ↓
Character Predictions
    ↓
Generated Text
```

---

# 15. Learning Outcomes

After completing this lab, you should understand:

* How character-level datasets are created.
* How characters are converted into numerical IDs.
* How embeddings represent characters.
* How LSTM networks process sequential data.
* How GRUs differ from LSTMs.
* How Cross-Entropy Loss is used for character prediction.
* How backpropagation trains an RNN-based model.
* How a trained LSTM can generate text.
* How `stoi` and `itos` mappings are used during generation.

---

# 16. Possible Improvements

The project can be extended by:

1. Using a larger text dataset.
2. Training for more epochs.
3. Comparing LSTM and GRU performance.
4. Adding temperature-based sampling.
5. Saving and loading the trained model.
6. Generating longer text.
7. Using multiple LSTM layers.
8. Increasing the hidden size.
9. Adding dropout.
10. Using a real book or text corpus.

---

## 🏁 Conclusion

This lab demonstrates the fundamentals of **LSTM and GRU-based character-level text generation using PyTorch**.

The workflow starts with raw text, creates a character vocabulary, converts characters into numerical IDs, prepares sequential training data, trains an LSTM model, and finally generates new text character by character.

The same concepts form the foundation for more advanced sequence-processing systems, including language models, text prediction systems, chatbots, and other Natural Language Processing applications.
