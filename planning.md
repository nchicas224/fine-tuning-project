# TakeMeter Planning

## Community

This project uses public Reddit comments from the Google Research GoEmotions corpus. The corpus is a good fit for a classification task because it contains short, informal comments from hundreds of public subreddits, with a wide range of emotional and conversational stances. The discourse is varied enough to distinguish supportive reactions, critical reactions, uncertainty-seeking comments, and neutral observations.

## Label Taxonomy

The task is to classify the discourse function or emotional stance of a Reddit comment. Each example receives exactly one label.

### supportive_reaction

A positive, affirming, appreciative, or encouraging response toward another person, idea, outcome, or object.

Examples:
- "You?re welcome, glad to help!"
- "That's a giant, adorable, super-good woofer you got right there!"

### critical_reaction

A negative, annoyed, angry, disappointed, or disapproving response toward another person, idea, outcome, or object.

Examples:
- "All of Disney is cringey. It needs to be stopped"
- "Ah yes three of my ~~best friends~~ worst enemies"

### uncertain_question

A comment that asks for information, expresses confusion, registers surprise, or shows realization about something unclear or newly understood.

Examples:
- "How do they know how far away the FRBs are coming from"
- "The internet never ceases to surprise"

### neutral_observation

A mostly descriptive or informational comment that does not primarily express support, criticism, or uncertainty.

Examples:
- "Migrants are here legally, illegals aren't. There is a huge difference."
- "Avid Android user here. My Xiaomi Mi Mix 2 is by far the sexiest and fastest phone I have ever had. Such bliss when using it."

## Hard Edge Cases

Some comments combine emotional language with information. If the emotional stance is the main point of the comment, I label it supportive_reaction or critical_reaction. If the comment is mostly describing a fact, event, or personal experience without asking a question or clearly evaluating it, I label it neutral_observation.

Some sarcastic comments are difficult because they may look neutral on the surface but function as criticism. When sarcasm is clearly used to reject or mock something, I label the comment critical_reaction. If the sarcasm is too context-dependent to identify from the text alone, I prefer neutral_observation.

Some comments express surprise but also approval or criticism. If the surprise is the main communicative function, I label it uncertain_question. If the comment uses surprise only to intensify praise or criticism, I label it supportive_reaction or critical_reaction.

Difficult examples from the sampled dataset:
- "You completely missed the point of the meme..." could be realization or criticism. I label it uncertain_question when the main signal is pointing out a misunderstanding, but this is close to critical_reaction.
- "That guy is an actor. Forgot his name but he's in imdb, played small roles." was sourced from admiration, but the text reads mostly informational. This is a possible noisy source-label case.
- "Avid Android user here. My Xiaomi Mi Mix 2 is by far the sexiest and fastest phone I have ever had. Such bliss when using it." is sourced as neutral, but it contains praise. I keep it neutral_observation because it mainly reports personal experience.

## Data Collection Plan

I used the public GoEmotions dataset from Google Research, which contains Reddit comments annotated for emotions. I downloaded the three official raw CSV files and created a smaller project dataset of 200 clear single-emotion examples.

The project dataset is balanced across four labels:

| Label | Count |
| --- | ---: |
| supportive_reaction | 50 |
| critical_reaction | 50 |
| uncertain_question | 50 |
| neutral_observation | 50 |

The derived CSV is `data/takemeter_goemotions_200.csv`. It includes the required `text` and `label` columns plus `source_emotion`, `subreddit`, and `notes` for transparency. If one label had been underrepresented, I would have reduced the taxonomy or sampled additional source emotions related to that label.

## Evaluation Metrics

I will report overall accuracy for the fine-tuned model and the Groq zero-shot baseline because the dataset is balanced and each example has one correct label. Accuracy alone is not enough, so I will also report per-class precision, recall, and F1 to identify whether the model is over-predicting or missing specific labels.

I will include a confusion matrix for the fine-tuned model because the most important question is not just whether the model is wrong, but which boundaries it fails to learn. In this taxonomy, the most likely hard boundaries are critical_reaction vs. neutral_observation and uncertain_question vs. neutral_observation.

## Definition of Success

A useful classifier should beat the zero-shot Groq baseline on the same test set or come close while being cheaper and locally reproducible. I would consider the model successful if it reaches at least 0.70 overall accuracy and avoids a near-zero F1 score for any class. For a real community tool, I would want stronger performance, more reviewed labels, and explicit testing on sarcasm and context-dependent comments.

## AI Tool Plan

For label stress-testing, I used AI assistance to compare possible label taxonomies and identify edge cases before finalizing the four-label setup. I revised the labels to make them mutually exclusive and grounded in the GoEmotions source annotations.

For annotation assistance, I used the original GoEmotions emotion annotations to pre-map examples into the four TakeMeter labels, then inspected samples for noise and ambiguity. I did not treat the source labels as perfect; the README will disclose that this is a derived dataset and discuss likely source-label noise.

For failure analysis, after running the fine-tuned model, I will use AI assistance to inspect wrong predictions and suggest systematic error patterns. I will verify any proposed pattern by re-reading the misclassified examples before writing the final evaluation.
