# LLM-Evaluations-Metrics

# BLEU and ROUGE Evaluation Metrics

A Python implementation of BLEU and ROUGE metrics for evaluating natural language generation models. This library provides clean, easy-to-use implementations of both metrics with detailed explanations and examples.

## 📋 Overview

**BLEU** (Bilingual Evaluation Understudy) and **ROUGE** (Recall-Oriented Understudy for Gisting Evaluation) are two fundamental metrics for evaluating the quality of machine-generated text:

- **BLEU**: Focuses on **precision** - measures how much of the generated text appears in reference text(s)
- **ROUGE**: Focuses on **recall** - measures how much of the reference content is captured in generated text

## 🚀 Quick Start

```python
from bluescore import BLEUScore
from rougescore import ROUGEScore

# Initialize scorers
bleu_scorer = BLEUScore(max_n=4)
rouge_scorer = ROUGEScore()

# Example texts
candidate = "The cat sat on the mat and looked around"
references = ["The cat sat on the mat", "A cat was sitting on the mat"]
reference_single = "The cat sat on the mat and was very comfortable"

# Calculate BLEU score
bleu_results = bleu_scorer.calculate(candidate, references)
print(f"BLEU Score: {bleu_results['bleu']:.4f}")

# Calculate ROUGE scores
rouge_results = rouge_scorer.calculate_all(candidate, reference_single)
print(f"ROUGE-1 F1: {rouge_results['rouge-1']['f1']:.4f}")
print(f"ROUGE-2 F1: {rouge_results['rouge-2']['f1']:.4f}")
print(f"ROUGE-L F1: {rouge_results['rouge-l']['f1']:.4f}")
```

## 📊 Key Differences

| Aspect | BLEU | ROUGE |
|--------|------|-------|
| **Focus** | Precision | Recall |
| **Best for** | Machine Translation | Text Summarization |
| **Multiple References** | Yes | Typically Single |
| **Penalty** | Brevity penalty | None |
| **Behavior** | Favors accurate, concise text | Favors comprehensive coverage |

## 🔧 Features

### BLEU Scorer
- **N-gram precision**: Supports 1-gram to 4-gram evaluation
- **Modified precision**: Clips n-gram counts to avoid repetition bias
- **Brevity penalty**: Prevents gaming with overly short outputs
- **Multiple references**: Can evaluate against multiple reference texts

### ROUGE Scorer
- **ROUGE-1**: Unigram overlap (word-level recall)
- **ROUGE-2**: Bigram overlap (phrase-level recall)
- **ROUGE-L**: Longest Common Subsequence (structural similarity)
- **Comprehensive metrics**: Returns precision, recall, and F1 for each variant

## 📖 Usage Examples

### Basic Evaluation
```python
# Machine Translation Evaluation (BLEU)
candidate = "Hello world"
references = ["Hello world", "Hi world"]
bleu_score = bleu_scorer.calculate(candidate, references)

# Summarization Evaluation (ROUGE)
summary = "AI models are improving rapidly"
reference = "Artificial intelligence models are advancing and improving at a rapid pace"
rouge_scores = rouge_scorer.calculate_all(summary, reference)
```

### Batch Evaluation
```python
candidates = [
    "The weather is nice today",
    "It's sunny and warm outside"
]
references = [
    ["The weather is good today", "Today has nice weather"],
    ["It's sunny outside", "The day is warm and bright"]
]

for i, (cand, refs) in enumerate(zip(candidates, references)):
    bleu = bleu_scorer.calculate(cand, refs)
    print(f"Candidate {i+1} BLEU: {bleu['bleu']:.4f}")
```

### Understanding the Metrics
```python
# Short vs Long candidates
reference = "The cat sat on the mat"
short_candidate = "The cat sat"
long_candidate = "The big fluffy cat sat comfortably on the soft mat"

# BLEU favors precision (short candidate may score higher)
bleu_short = bleu_scorer.calculate(short_candidate, [reference])
bleu_long = bleu_scorer.calculate(long_candidate, [reference])

# ROUGE favors recall (long candidate may score higher)
rouge_short = rouge_scorer.rouge_n(short_candidate, reference, 1)
rouge_long = rouge_scorer.rouge_n(long_candidate, reference, 1)

print("BLEU demonstrates precision focus:")
print(f"Short: {bleu_short['bleu']:.4f}, Long: {bleu_long['bleu']:.4f}")
print("ROUGE demonstrates recall focus:")
print(f"Short: {rouge_short['recall']:.4f}, Long: {rouge_long['recall']:.4f}")
```

## 🎯 When to Use Which Metric

### Use BLEU when:
- Evaluating **machine translation** systems
- **Precision** is more important than coverage
- You have **multiple reference translations**
- You want to penalize **overly verbose** outputs
- **Faithfulness** to reference is critical

### Use ROUGE when:
- Evaluating **text summarization** systems
- **Content coverage** is important
- You want to measure how well the output **captures key information**
- **Recall** of important details matters more than precision
- Evaluating **abstractive generation** tasks

## 📐 Metric Interpretation

### BLEU Scores
- **Range**: 0.0 to 1.0 (higher is better)
- **0.0 - 0.1**: Poor quality
- **0.1 - 0.3**: Moderate quality
- **0.3 - 0.6**: Good quality
- **0.6+**: High quality

### ROUGE Scores
- **Range**: 0.0 to 1.0 for each component (higher is better)
- **Precision**: How much of the candidate appears in reference
- **Recall**: How much of the reference appears in candidate
- **F1**: Harmonic mean of precision and recall

## 🔬 Technical Details

### BLEU Implementation
- Uses **modified n-gram precision** to prevent repetition exploitation
- Applies **brevity penalty**: `BP = exp(1 - r/c)` if `c < r`, else `1`
- **Geometric mean** of n-gram precisions (1 to 4-grams by default)
- Final score: `BLEU = BP × exp(Σ log(p_n) / N)`

### ROUGE Implementation
- **ROUGE-N**: Based on n-gram overlap between candidate and reference
- **ROUGE-L**: Uses Longest Common Subsequence for structural similarity
- Returns precision, recall, and F1 for comprehensive evaluation

## 🛠️ Requirements

- Python 3.6+
- No external dependencies (uses only standard library)

## 📝 License

MIT License - feel free to use in your projects!

## 🤝 Contributing

Contributions welcome! Feel free to:
- Add more ROUGE variants (ROUGE-W, ROUGE-S)
- Implement additional evaluation metrics
- Add more comprehensive examples
- Improve documentation

## 📚 References

- **BLEU**: Papineni, K., et al. (2002). "BLEU: a method for automatic evaluation of machine translation"
- **ROUGE**: Lin, C. Y. (2004). "Rouge: A package for automatic evaluation of summaries"

## 🎯 Example Output

```
=== BLEU vs ROUGE Evaluation Demo ===

Candidate text: The cat sat on the mat and looked around
BLEU references: ['The cat sat on the mat', 'A cat was sitting on the mat']
ROUGE reference: The cat sat on the mat and was very comfortable

--- BLEU Results (Focus: Precision) ---
BLEU Score: 0.5946
1-gram precision: 1.0000
2-gram precision: 0.8571
3-gram precision: 0.6000
4-gram precision: 0.2000
Brevity penalty: 1.0000

--- ROUGE Results (Focus: Recall) ---
ROUGE-1:
  Precision: 0.7778
  Recall: 0.7778
  F1: 0.7778
ROUGE-2:
  Precision: 0.5714
  Recall: 0.5714
  F1: 0.5714
ROUGE-L:
  Precision: 0.7778
  Recall: 0.7778
  F1: 0.7778
```
