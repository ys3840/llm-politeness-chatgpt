# Should We Be Polite to ChatGPT?  
Exploring how different user tones affect ChatGPT responses.

---

## 1. Project Overview

This project studies whether the **tone** of a user’s prompt changes how a large language model responds.

We focus on three tones:

- **Polite** – e.g., “Hi there, could you please… Thank you so much!”
- **Neutral** – direct instructions
- **Commanding** – e.g., “… I need you to do this immediately.”

We ask:

1. Does tone change the **style** of responses (length, structure, sentiment)?
2. Does tone change how **social** the model sounds (hedging, first-person pronouns)?
3. Does tone change **factual / math accuracy**?

All experiments are run with **gpt-4.1-mini** via the OpenAI API.

---

## 2. Repository Structure

```text
.
├── run_gpt.py                               # Script to call GPT and collect responses
├── analyze_responses.py                     # Script to score responses and summarize metrics
├── llm_politeness_responses.csv             # Raw responses (one row per model call)
├── llm_politeness_responses_scored.csv      # With sentiment, counts, math accuracy, etc.
├── llm_politeness_summary_by_tone_task.csv  # Aggregated by tone × task type
├── figures/
│   ├── word_count_by_tone.png
│   ├── sentiment_count_by_tone.png
│   ├── modal_verbs.png
│   ├── 1st_person.png
│   ├── taskxtone.png
│   └── math_accuracy.png
└── README.md
