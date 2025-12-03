import csv
import os
from collections import defaultdict
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ========= FILE NAMES =========
INPUT_CSV = "llm_politeness_responses.csv"  # from your generation script
OUTPUT_SCORED_CSV = "llm_politeness_responses_scored.csv"
OUTPUT_SUMMARY_CSV = "llm_politeness_summary_by_tone_task.csv"

# ========= LEXICONS / WORD LISTS =========
POLITENESS_WORDS = ["please", "thank", "thanks", "appreciate", "sorry"]
FIRST_PERSON_PRONOUNS = ["i", "me", "my", "mine", "we", "us", "our", "ours"]
MODAL_VERBS = ["can", "could", "would", "should", "might", "may", "must", "will", "shall"]

# Initialize VADER sentiment analyzer
vader_analyzer = SentimentIntensityAnalyzer()


# ========= BASIC TEXT METRICS =========

def word_count(text: str) -> int:
    return len(text.split())


def sentence_count(text: str) -> int:
    """
    Very rough sentence splitting based on ., ?, !.
    """
    tmp = text.replace("?", ".").replace("!", ".")
    sentences = [s.strip() for s in tmp.split(".") if s.strip()]
    return len(sentences)


def avg_sentence_length(text: str) -> float:
    wc = word_count(text)
    sc = sentence_count(text)
    if sc == 0:
        return 0.0
    return wc / sc


def sentiment_score(text: str) -> float:
    """
    VADER compound sentiment score in [-1, 1].
    -1 = very negative, 0 = neutral, 1 = very positive.
    """
    scores = vader_analyzer.polarity_scores(text)
    return scores["compound"]


def count_substrings(text: str, substr_list):
    t = text.lower()
    return sum(t.count(s) for s in substr_list)


def style_tone_features(text: str):
    """
    Heuristic style/tone proxies:
    - word count
    - sentence count
    - avg sentence length
    - exclamation and question marks
    - politeness words
    - first-person pronouns
    - modal verbs
    - VADER sentiment score
    """
    wc = word_count(text)
    sc = sentence_count(text)
    avg_len = avg_sentence_length(text)

    exclam_count = text.count("!")
    question_count = text.count("?")

    politeness_count = count_substrings(text, POLITENESS_WORDS)
    first_person_count = count_substrings(text, FIRST_PERSON_PRONOUNS)
    modal_count = count_substrings(text, MODAL_VERBS)

    sent = sentiment_score(text)

    return {
        "response_word_count": wc,
        "response_sentence_count": sc,
        "response_avg_sentence_length": avg_len,
        "response_exclamation_count": exclam_count,
        "response_question_count": question_count,
        "response_politeness_word_count": politeness_count,
        "response_first_person_count": first_person_count,
        "response_modal_verb_count": modal_count,
        "sentiment_score": sent,
    }


def politeness_markers(text: str):
    """
    More granular politeness markers.
    """
    t = text.lower()
    return {
        "please_count": t.count("please"),
        "thank_count": t.count("thank "),
        "thanks_count": t.count("thanks"),
        "appreciate_count": t.count("appreciate"),
        "sorry_count": t.count("sorry"),
    }


# ========= STEP 1: SCORE RESPONSES =========

def score_responses():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    with open(INPUT_CSV, "r", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        base_fieldnames = reader.fieldnames

        if "response_text" not in base_fieldnames:
            raise ValueError("Input CSV must contain a 'response_text' column.")

        extra_fields = [
            "response_word_count",
            "response_sentence_count",
            "response_avg_sentence_length",
            "response_exclamation_count",
            "response_question_count",
            "response_politeness_word_count",
            "response_first_person_count",
            "response_modal_verb_count",
            "sentiment_score",
            "please_count",
            "thank_count",
            "thanks_count",
            "appreciate_count",
            "sorry_count",
        ]
        fieldnames = base_fieldnames + extra_fields

        rows_scored = []

        for row in reader:
            response = row["response_text"]

            style_feats = style_tone_features(response)
            pm = politeness_markers(response)

            row.update(style_feats)
            row.update(pm)

            rows_scored.append(row)

    with open(OUTPUT_SCORED_CSV, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_scored:
            writer.writerow(row)

    print(f"[1] Scored responses written to: {OUTPUT_SCORED_CSV}")
    return rows_scored, fieldnames


# ========= STEP 2: SUMMARY BY TONE × TASK (AUTOMATIC FEATURES) =========

def summarize_scored(rows_scored):
    # aggregate by (task_type, tone)
    summary = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(int)

    metrics_to_avg = [
        "response_word_count",
        "response_sentence_count",
        "response_avg_sentence_length",
        "response_exclamation_count",
        "response_question_count",
        "response_politeness_word_count",
        "response_first_person_count",
        "response_modal_verb_count",
        "sentiment_score",
        "please_count",
        "thank_count",
        "thanks_count",
        "appreciate_count",
        "sorry_count",
    ]

    for row in rows_scored:
        task_type = row.get("task_type", "NA")
        tone = row.get("tone", "NA")
        key = (task_type, tone)
        counts[key] += 1

        for m in metrics_to_avg:
            summary[key][f"sum_{m}"] += float(row[m])

    # write summary
    fieldnames = [
        "task_type",
        "tone",
        "n_responses",
    ] + [f"avg_{m}" for m in metrics_to_avg]

    with open(OUTPUT_SUMMARY_CSV, "w", newline="", encoding="utf-8") as f_sum:
        writer = csv.DictWriter(f_sum, fieldnames=fieldnames)
        writer.writeheader()

        for (task_type, tone), sums in sorted(summary.items()):
            n = counts[(task_type, tone)] or 1
            row = {
                "task_type": task_type,
                "tone": tone,
                "n_responses": n,
            }
            for m in metrics_to_avg:
                row[f"avg_{m}"] = sums[f"sum_{m}"] / n
            writer.writerow(row)

    print(f"[2] Automatic summary by task_type × tone written to: {OUTPUT_SUMMARY_CSV}")


# ========= MAIN =========

def main():
    rows_scored, _ = score_responses()
    summarize_scored(rows_scored)


if __name__ == "__main__":
    main()
