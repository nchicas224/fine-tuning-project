# TakeMeter: Reddit Comment Stance Classifier

TakeMeter is a fine-tuned text classifier for public Reddit comments from the Google Research GoEmotions corpus. The classifier predicts the discourse function or emotional stance of a comment: supportive reaction, critical reaction, uncertainty/questioning, or neutral observation.

## Community and Task

The community source is public Reddit discourse represented in the GoEmotions dataset. This is a good fit because Reddit comments are short, informal, and highly varied in tone. A useful classifier should distinguish whether a comment is mainly supportive, critical, uncertain, or neutral.

## Labels

| Label | Definition | Example 1 | Example 2 |
| --- | --- | --- | --- |
| `supportive_reaction` | A positive, affirming, appreciative, or encouraging response. | "You?re welcome, glad to help!" | "That's a giant, adorable, super-good woofer you got right there!" |
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

Complete this section after running the Colab notebook.

| Model | Accuracy |
| --- | ---: |
| Zero-shot Groq baseline | TODO |
| Fine-tuned DistilBERT | TODO |

### Per-Class Metrics

Paste the baseline and fine-tuned classification reports here after running the notebook.

### Fine-Tuned Confusion Matrix

Write the confusion matrix as a markdown table here after running the notebook. Also commit `confusion_matrix.png`.

| True Label | Predicted supportive_reaction | Predicted critical_reaction | Predicted uncertain_question | Predicted neutral_observation |
| --- | ---: | ---: | ---: | ---: |
| supportive_reaction | TODO | TODO | TODO | TODO |
| critical_reaction | TODO | TODO | TODO | TODO |
| uncertain_question | TODO | TODO | TODO | TODO |
| neutral_observation | TODO | TODO | TODO | TODO |

### Wrong Prediction Analysis

After running the notebook, choose three wrong predictions and analyze why they failed.

1. TODO
2. TODO
3. TODO

### Sample Classifications

After running the fine-tuned model, add 3-5 sample classifications with predicted label and confidence.

| Text | Predicted Label | Confidence | Comment |
| --- | --- | ---: | --- |
| TODO | TODO | TODO | TODO |

## Reflection

Complete after evaluation. Focus on what the model actually learned versus what the label definitions intended. In particular, check whether it learned emotional keywords, subreddit/topic cues, or the intended discourse function.

## Spec Reflection

The planning spec helped by forcing the labels to be mutually exclusive before training. One likely divergence is that the final dataset uses source emotion annotations from GoEmotions rather than fully manual labels, which speeds up the project but introduces possible label noise that needs to be acknowledged.

## AI Usage

AI assistance was used to review the project requirements, select a public dataset, design a feasible label taxonomy, generate the derived dataset script, and draft project documentation. I reviewed and approved commands before data retrieval or file creation, and I revised the project direction to fit the available public dataset.

If additional AI assistance is used for failure analysis after training, document the exact prompt/task here and note which suggestions were accepted, revised, or rejected.
