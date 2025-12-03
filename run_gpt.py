import os
import csv
import time
from datetime import datetime
from typing import Dict, List
from openai import OpenAI

# ==============================
# CONFIG
# ==============================

# Set your API key (or rely on environment variable OPENAI_API_KEY)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MODEL_NAME = "gpt-4.1-mini"  
TEMPERATURE = 0.7
N_RUNS_PER_CELL = 3           # 3 runs per tone–task–prompt

OUTPUT_CSV = "llm_politeness_responses.csv"

# ----- 4 task types × 5 prompts each -----
# TODO: Replace these with your real stimuli.
BASE_PROMPTS: Dict[str, List[str]] = {
    "analytical": [
        "Analyze the potential economic impact of a sudden increase in energy prices on households and businesses.",
        "Analyze the relationship between interest rates and consumer spending, and explain under what conditions this relationship might weaken.",
        "Analyze how consumer behavior might change during periods of economic uncertainty and explain why.",
        "Explain how government policy changes—such as tax cuts or spending increases—can create short-term and long-term economic effects.",
        "Explain how technological innovation can affect labor markets differently across skill levels."
    ],
    "factual": [
        "Explain how photosynthesis works in plants.",
        "Explain the main causes of World War I.",
        "Explain how vaccines help protect the human body from disease.",
        "Consider a simple log-stochastic variance model: yt = σtϵt where ϵt is standard normal and log(σt2) = α + βlog(σt2−1) + ϵσt where ϵσt is i.i.d standard normal. Hint: you will need moments of the log-normal distribution.) (1) Compute the first and second unconditional moments of log(σt2). (2) Compute the first 4 moments of yt. Use these moments or a subset of these moments to set up a GMM estimator for the parameters of the model (α, β).",
        "Prove that the well-known test of the over-identifying restrictions in a GMM system is distributed χ2(r −q), where r is the number of orthogonality conditions and q is the number of parameters. (Hint: Show T[gT (ˆbT )]′Sˆ−1 T [gT (ˆbT )] L −→ χ2(r − q))"
    ],
    "advisory": [
        "Give me advice on how to study more effectively for exams.",
        "Give me advice on how to choose a college major.",
        "Give me advice on how to manage my time as a busy student.",
        "Give me advice on how to prepare for a job interview.",
        "Give me advice on how to stay motivated when working on long projects."
    ],
    "creative": [
        "Write a short story about a robot who wants to become human.",
        "Write a short poem about the first day of college.",
        "Write a short paragraph describing a city of the future.",
        "Write a short scene where two friends reunite after many years apart.",
        "Write a short story that starts with the sentence: 'The lights went out, and everything changed.'"
    ],
}

# 3 tone templates
TONE_TEMPLATES = {
    "polite":   "Hi there! If it isn’t too much trouble, could you please {content} I would really appreciate your help. Thank you so much!", 
    "neutral":  "{content}",
    "commanding": "{content} I need you to do this immediately. Do not delay."
}

TONES = ["polite", "neutral", "commanding"]


# ==============================
# HELPER FUNCTIONS
# ==============================

def build_prompt(tone: str, base_content: str) -> str:
    """Wrap base content in the chosen tone template."""
    return TONE_TEMPLATES[tone].format(content=base_content)


def call_chatgpt(client: OpenAI, prompt_text: str) -> str:
    """Call the Chat Completions API and return the response text."""
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        messages=[
            {"role": "user", "content": prompt_text}
        ]
    )
    return resp.choices[0].message.content


def init_csv(path: str):
    """Create CSV file with header if it doesn't already exist."""
    if os.path.exists(path):
        return

    header = [
        "timestamp_utc",
        "task_type",
        "tone",
        "prompt_index",      # 0–4
        "run_index",         # 0–2
        "model",
        "temperature",
        "base_prompt_text",
        "full_prompt_text",
        "response_text"
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)


def append_row(path: str, row: dict):
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            row["timestamp_utc"],
            row["task_type"],
            row["tone"],
            row["prompt_index"],
            row["run_index"],
            row["model"],
            row["temperature"],
            row["base_prompt_text"],
            row["full_prompt_text"],
            row["response_text"],
        ])


# ==============================
# MAIN EXPERIMENT LOOP
# ==============================

def main():
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is not set. Please set it as an environment variable before running this script."
        )
    client = OpenAI(api_key=OPENAI_API_KEY)

    init_csv(OUTPUT_CSV)

    n_task_types = len(BASE_PROMPTS)
    prompts_per_task = len(next(iter(BASE_PROMPTS.values())))  # assume all equal length
    total_responses = n_task_types * len(TONES) * prompts_per_task * N_RUNS_PER_CELL

    print(f"About to generate {total_responses} responses.")

    counter = 0
    for task_type, prompts in BASE_PROMPTS.items():
        for tone in TONES:
            for prompt_idx, base_content in enumerate(prompts):
                full_prompt = build_prompt(tone, base_content)

                for run_idx in range(N_RUNS_PER_CELL):
                    counter += 1
                    print(
                        f"[{counter}/{total_responses}] "
                        f"task={task_type}, tone={tone}, prompt#{prompt_idx+1}, run#{run_idx+1}"
                    )

                    try:
                        response_text = call_chatgpt(client, full_prompt)
                    except Exception as e:
                        print(f"API error: {e}")
                        response_text = f"[API_ERROR] {e}"

                    row = {
                        "timestamp_utc": datetime.utcnow().isoformat(),
                        "task_type": task_type,
                        "tone": tone,
                        "prompt_index": prompt_idx,
                        "run_index": run_idx,
                        "model": MODEL_NAME,
                        "temperature": TEMPERATURE,
                        "base_prompt_text": base_content,
                        "full_prompt_text": full_prompt,
                        "response_text": response_text,
                    }
                    append_row(OUTPUT_CSV, row)

                    # small delay to be gentle on rate limits
                    time.sleep(0.3)


if __name__ == "__main__":
    main()
