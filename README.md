# TakeMeter: Reddit Comment Stance Classifier

TakeMeter is a fine-tuned text classifier for public Reddit comments from the Google Research GoEmotions corpus. The classifier predicts the discourse function or emotional stance of a comment: supportive reaction, critical reaction, uncertainty/questioning, or neutral observation.

## Community and Task

The community source is public Reddit discourse represented in the GoEmotions dataset. This is a good fit because Reddit comments are short, informal, and highly varied in tone. A useful classifier should distinguish whether a comment is mainly supportive, critical, uncertain, or neutral.

## Labels

| Label | Definition | Example 1 | Example 2 |
| --- | --- | --- | --- |
| `supportive_reaction` | A positive, affirming, appreciative, or encouraging response. | "You're welcome, glad to help!" | "That's a giant, adorable, super-good woofer you got right there!" |
| `critical_reaction` | A negative, annoyed, angry, disappointed, or disapproving response. | "All of Disney is cringey. It needs to be stopped" | "Ah yes three of my ~~best friends~~ worst enemies" |
| `uncertain_question` | A comment that asks for information, expresses confusion, registers surprise, or shows realization. | "How do they know how far away the FRBs are coming from" | "The internet never ceases to surprise" |
| `neutral_observation` | A mostly descriptive or informational comment without a primary supportive, critical, or uncertain stance. | "Migrants are here legally, illegals aren't. There is a huge difference." | "Avid Android user here. My Xiaomi Mi Mix 2 is by far the sexiest and fastest phone I have ever had." |

## Dataset

The dataset is derived from Google Research's public GoEmotions corpus. I downloaded the three raw GoEmotions CSV files and sampled 200 clear single-emotion rows into `data/takemeter_goemotions_200.csv`.

| Label | Count |
| --- | ---: |
| `supportive_reaction` | 50 |
| `critical_reaction` | 50 |
| `uncertain_question` | 50 |
| `neutral_observation` | 50 |

The final CSV includes `text`, `label`, `source_emotion`, `subreddit`, and `notes`. The notebook only requires `text` and `label`; the extra fields document how each example was mapped.

## Difficult Labeling Cases

| Example | Possible Labels | Decision |
| --- | --- | --- |
| "You completely missed the point of the meme..." | `uncertain_question`, `critical_reaction` | Labeled as `uncertain_question` because it identifies a misunderstanding, though it has a critical tone. |
| "That guy is an actor. Forgot his name but he's in imdb, played small roles." | `supportive_reaction`, `neutral_observation` | Kept according to the source emotion mapping, but noted as possible label noise because the text reads informational. |
| "Avid Android user here. My Xiaomi Mi Mix 2 is by far the sexiest and fastest phone I have ever had." | `neutral_observation`, `supportive_reaction` | Treated as `neutral_observation` because it is framed as a personal report, though it contains praise. |

## Fine-Tuning Approach

The fine-tuned model starts from `distilbert-base-uncased` using the starter notebook. The notebook splits the dataset into 70% training, 15% validation, and 15% test data, tokenizes comments with a maximum sequence length of 256, and trains a sequence classification head.

Default hyperparameters:

| Hyperparameter | Value | Reason |
| --- | ---: | --- |
| Epochs | 3 | Enough passes for a small dataset without intentionally overfitting. |
| Learning rate | 2e-5 | Standard starting point for BERT-family fine-tuning. |
| Train batch size | 16 | Fits comfortably on a Colab T4 GPU. |
| Weight decay | 0.01 | Adds mild regularization. |

## Baseline

The baseline is Groq `llama-3.3-70b-versatile` used zero-shot on the same test set. The prompt defines the four labels and requires the model to output only the label name.

## Evaluation Report

The fine-tuned model did not beat the zero-shot baseline. This is still useful because the failure pattern is specific: the fine-tuned model mostly collapsed into predicting `critical_reaction`, while Groq handled three of the four labels better but failed completely on `neutral_observation`.

| Model | Accuracy |
| --- | ---: |
| Zero-shot Groq baseline | 0.500 |
| Fine-tuned DistilBERT | 0.267 |

### Per-Class Metrics

Baseline, Groq `llama-3.3-70b-versatile`:

| Label | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| `supportive_reaction` | 0.50 | 0.62 | 0.56 | 8 |
| `critical_reaction` | 0.56 | 0.71 | 0.62 | 7 |
| `uncertain_question` | 0.56 | 0.62 | 0.59 | 8 |
| `neutral_observation` | 0.00 | 0.00 | 0.00 | 7 |
| Accuracy |  |  | 0.50 | 30 |
| Macro avg | 0.40 | 0.49 | 0.44 | 30 |
| Weighted avg | 0.41 | 0.50 | 0.45 | 30 |

Fine-tuned DistilBERT:

| Label | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| `supportive_reaction` | 0.00 | 0.00 | 0.00 | 8 |
| `critical_reaction` | 0.19 | 0.57 | 0.29 | 7 |
| `uncertain_question` | 0.44 | 0.50 | 0.47 | 8 |
| `neutral_observation` | 0.00 | 0.00 | 0.00 | 7 |
| Accuracy |  |  | 0.27 | 30 |
| Macro avg | 0.16 | 0.27 | 0.19 | 30 |
| Weighted avg | 0.16 | 0.27 | 0.19 | 30 |

### Fine-Tuned Confusion Matrix

The committed image version is `data/confusion_matrix.png`.

| True Label | Predicted supportive_reaction | Predicted critical_reaction | Predicted uncertain_question | Predicted neutral_observation |
| --- | ---: | ---: | ---: | ---: |
| `supportive_reaction` | 0 | 6 | 2 | 0 |
| `critical_reaction` | 0 | 4 | 3 | 0 |
| `uncertain_question` | 0 | 4 | 4 | 0 |
| `neutral_observation` | 0 | 7 | 0 | 0 |

The strongest pattern is that the fine-tuned model never predicted `supportive_reaction` or `neutral_observation`. It predicted `critical_reaction` for 21 of the 30 test examples and `uncertain_question` for the other 9. That explains the low accuracy and the zero F1 scores for two classes.

### Wrong Prediction Analysis

1. Text: "That's a giant, adorable, super-good woofer you got right there!"
   - True label: `supportive_reaction`
   - Predicted: `critical_reaction` with 0.29 confidence
   - Analysis: This is a clear supportive comment, but the model predicted the majority failure class. The sentence contains obvious positive words like "adorable" and "super-good," so this mistake suggests the fine-tuned model did not learn a stable positive-reaction boundary from only 140 training examples.

2. Text: "Avid Android user here. My Xiaomi Mi Mix 2 is by far the sexiest and fastest phone I have ever had. Such bliss when using it."
   - True label: `neutral_observation`
   - Predicted: `critical_reaction` with 0.29 confidence
   - Analysis: This example is genuinely tricky because the source mapping treats it as neutral, but the wording contains praise. The model's `critical_reaction` prediction is still wrong, but the example exposes a weakness in the derived dataset: some `neutral_observation` rows are not emotionally neutral in plain English.

3. Text: "Oh. It really is [RELIGION]. I was wondering if that was a bad gag."
   - True label: `uncertain_question`
   - Predicted: `critical_reaction` with 0.28 confidence
   - Analysis: The phrase "bad gag" carries a negative cue, while "I was wondering" signals uncertainty. The model appears to overweight the negative phrase and miss the question/realization function. This is one of the intended hard cases for the taxonomy.

4. Text: "Always go for someone you like and feel connected to."
   - True label: `supportive_reaction`
   - Predicted: `uncertain_question` with 0.29 confidence
   - Analysis: This is advice with a supportive tone, but it does not contain obvious gratitude or praise words. The model likely struggled because supportive reactions in the dataset are heterogeneous: some are thanks, some are compliments, and some are encouraging advice.

### Sample Classifications

| Text | Predicted Label | Confidence | Comment |
| --- | --- | ---: | --- |
| "All of Disney is cringey. It needs to be stopped" | `critical_reaction` | 0.29 | Correct. The comment is an explicit negative evaluation. |
| "Do you have the voting link? I went on the NHL website but I didn't see anything." | `uncertain_question` | 0.28 | Correct. The comment is directly asking for missing information. |
| "I was rather curious to see the True North Church receiving $1m from the lotteries commission." | `uncertain_question` | 0.28 | Correct. The main signal is curiosity. |
| "That's a giant, adorable, super-good woofer you got right there!" | `critical_reaction` | 0.29 | Incorrect. This should be supportive, but the model collapsed toward `critical_reaction`. |
| "Oh. It really is [RELIGION]. I was wondering if that was a bad gag." | `critical_reaction` | 0.28 | Incorrect. The model focused on negative wording instead of the uncertainty cue. |

### Colab Helper Note

After the main notebook printed only wrong predictions, I added a small helper cell to print correct predictions for the sample-classifications section. The helper did not change training, evaluation, or model outputs. It only read variables that already existed after Section 4:

```python
print("Correct predictions:\n")
correct_idx = np.where(ft_pred_ids == ft_true_ids)[0]

for i, idx in enumerate(correct_idx[:5]):
    text = test_df.iloc[idx]["text"]
    true_label = ID_TO_LABEL[ft_true_ids[idx]]
    pred_label = ID_TO_LABEL[ft_pred_ids[idx]]
    confidence = ft_probs[idx][ft_pred_ids[idx]]
    print(f"--- Correct #{i+1} ---")
    print(f"Text: {text[:300]}{'...' if len(text) > 300 else ''}")
    print(f"True: {true_label}")
    print(f"Predicted: {pred_label} (confidence: {confidence:.2f})")
    print()
```

The line `correct_idx = np.where(ft_pred_ids == ft_true_ids)[0]` finds every test-set position where the predicted class id matches the true class id. The loop then uses each index to retrieve the original comment from `test_df`, convert numeric class ids back into readable label names with `ID_TO_LABEL`, and pull the model's confidence score from `ft_probs`. This helper was needed only for reporting: the project requires 3-5 sample classifications and at least one explained correct prediction, while the starter notebook focuses on wrong predictions for error analysis.

## Reflection

The fine-tuned model learned much less than intended. I intended the classifier to learn four discourse functions: support, criticism, uncertainty, and neutral observation. Instead, the model mostly learned a weak distinction between `critical_reaction` and `uncertain_question`, while failing to predict `supportive_reaction` or `neutral_observation` at all on the test set.

The most likely cause is a combination of small data size and noisy derived labels. The dataset has only 200 examples, so the training split contains about 35 examples per class. That is very small for learning subtle stance boundaries. The source GoEmotions labels are also emotion annotations, not exactly the same thing as discourse-function labels. Some rows mapped to `neutral_observation` contain praise or criticism when read without the original Reddit context, which makes that class especially hard.

The zero-shot Groq baseline performed better overall, but it also failed on `neutral_observation`. That suggests the neutral class is the weakest part of the taxonomy and the dataset. If I continued this project, I would manually review all 200 examples, remove or relabel noisy neutral rows, add more supportive and neutral examples, and possibly merge `neutral_observation` with another class if it remained too ambiguous.

## Spec Reflection

The planning spec helped by forcing the labels to be mutually exclusive before training. It also made the evaluation failure easier to interpret because I had already predicted that `neutral_observation` would be a difficult boundary.

The main implementation divergence is that the dataset uses source emotion annotations from GoEmotions instead of fully manual labels. That made the project feasible under time pressure, but it introduced label noise because emotion labels do not perfectly equal discourse-function labels. The evaluation results show that this shortcut mattered: the model struggled most with the classes where the mapping was least clean.

## AI Usage

AI assistance was used in several specific ways:

1. I directed AI to review the project PDF and convert the requirements into a task list. I used that to identify the required deliverables: `planning.md`, a 200-example dataset, baseline comparison, fine-tuning results, confusion matrix, error analysis, sample classifications, reflection, and demo.
2. I directed AI to search for public datasets and compare options. I approved using Google Research's GoEmotions dataset because it is public, Reddit-based, and available as CSV files.
3. I directed AI to create a derived 200-row dataset from clear single-emotion GoEmotions examples. I reviewed the proposed label taxonomy and approved the sampling approach before the file was created.
4. I directed AI to analyze the evaluation outputs and help write the failure analysis. I accepted the conclusion that the fine-tuned model collapsed toward `critical_reaction`, and I revised the writeup to emphasize label noise and data size rather than overstating the model's usefulness.

Profanity or inappropriate quoted text from model outputs was either avoided or redacted in this README.
