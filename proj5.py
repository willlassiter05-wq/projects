"""
analyze_feedback.py
-------------------
Analyzes Apple Vision Pro feedback in feedback.db using OpenAI GPT-5.

Outputs:
  • results.csv
  • sentiment_distribution.png
  • aspect_wordcloud.png
  • summary.txt
"""

import os
import sys
import sqlite3
import json
import time
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from tqdm import tqdm

# --- Step 1: Load API key ---
try:
    from apikey import OPENAI_API_KEY
    if not OPENAI_API_KEY.startswith("sk-"):
        raise ValueError("API key does not start with 'sk-'")
    print("🔑 API key loaded from apikey.py")
except (ModuleNotFoundError, ImportError, AttributeError, ValueError):
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if OPENAI_API_KEY and OPENAI_API_KEY.startswith("sk-"):
        print("🔑 API key loaded from environment variable")
    else:
        sys.exit(
            "❌ Could not load OpenAI API key. "
            "Create apikey.py with OPENAI_API_KEY='sk-...' "
            "or set environment variable OPENAI_API_KEY."
        )

# --- Step 2: Connect to OpenAI ---
try:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    print("✅ Connected to OpenAI (modern client)")
except Exception:
    try:
        import openai
        openai.api_key = OPENAI_API_KEY
        client = openai
        print("✅ Connected to OpenAI (legacy client)")
    except Exception as e:
        sys.exit(f"❌ Failed to connect to OpenAI: {e}")

# --- Config ---
DB_PATH = "feedback.db"
MODEL = "gpt-5"
REQUEST_DELAY = 0.35  # seconds between requests
OUTPUTS = {
    "csv": "results.csv",
    "chart": "sentiment_distribution.png",
    "wordcloud": "aspect_wordcloud.png",
    "summary": "summary.txt"
}

PROMPT = """
You are an expert sentiment analyst.
Analyze the customer review below and return ONLY valid JSON (no text before or after).

Review:
\"\"\"{text}\"\"\"

Your JSON must include:
- sentiment: "positive", "negative", or "neutral"
- confidence: number between 0 and 1
- aspects: list of {{"aspect": str, "sentiment": str, "quote": str}}

Example:
{{
  "sentiment": "positive",
  "confidence": 0.95,
  "aspects": [
    {{"aspect": "display", "sentiment": "positive", "quote": "The screen looks incredible"}},
    {{"aspect": "price", "sentiment": "negative", "quote": "It’s too expensive"}}
  ]
}}
"""

# --- Step 3: Load reviews ---
def load_reviews():
    print(f"🔹 Looking for feedback.db at: {os.path.abspath(DB_PATH)}")
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"❌ feedback.db not found at {os.path.abspath(DB_PATH)}. "
                                f"Please check the path or move the file there.")
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT id, review_text FROM reviews", con)
    con.close()
    return df


# --- Step 4: Analyze one review ---
def analyze_review(text):
    prompt = PROMPT.format(text=text.replace('"', '\\"'))
    for attempt in range(3):
        try:
            if hasattr(client, "responses"):
                resp = client.responses.create(model=MODEL, input=prompt, max_tokens=400, temperature=0)
                content = ""
                for item in resp.output:
                    if isinstance(item, dict) and "content" in item:
                        for c in item["content"]:
                            if c.get("type") == "output_text":
                                content += c.get("text", "")
                    elif isinstance(item, str):
                        content += item
            else:
                resp = client.ChatCompletion.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=400,
                )
                content = resp.choices[0].message.content.strip()

            start, end = content.find("{"), content.rfind("}")
            json_text = content[start:end+1] if start != -1 and end != -1 else content
            return json.loads(json_text)
        except Exception as e:
            print(f"Retry {attempt+1}: {e}")
            time.sleep(2 ** attempt)
    return {"sentiment": "neutral", "confidence": 0.0, "aspects": []}

# --- Step 5: Analyze all reviews ---
def analyze_all(df):
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing"):
        text = str(row["review_text"] or "").strip()
        if not text:
            results.append({
                "id": row["id"],
                "review_text": text,
                "sentiment": "neutral",
                "confidence": 0.0,
                "aspects": json.dumps([])
            })
            continue
        result = analyze_review(text)
        results.append({
            "id": row["id"],
            "review_text": text,
            "sentiment": result.get("sentiment", "neutral"),
            "confidence": result.get("confidence", 0.0),
            "aspects": json.dumps(result.get("aspects", []), ensure_ascii=False)
        })
        time.sleep(REQUEST_DELAY)
    return pd.DataFrame(results)

# --- Step 6: Visuals and summary ---
def save_outputs(df):
    df.to_csv(OUTPUTS["csv"], index=False)
    print(f"✅ Saved {OUTPUTS['csv']}")

    counts = Counter(df["sentiment"])
    labels = ["positive", "neutral", "negative"]
    values = [counts.get(x, 0) for x in labels]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values)
    plt.title("Sentiment Distribution")
    plt.tight_layout()
    plt.savefig(OUTPUTS["chart"])
    plt.close()
    print(f"✅ Saved {OUTPUTS['chart']}")

    # WordCloud of aspects
    aspects = []
    for s in df["aspects"]:
        try:
            for a in json.loads(s):
                if a.get("aspect"):
                    aspects.append(a["aspect"])
        except Exception:
            pass
    if aspects:
        text = " ".join(aspects)
        wc = WordCloud(width=800, height=400, collocations=False).generate(text)
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(OUTPUTS["wordcloud"])
        plt.close()
        print(f"✅ Saved {OUTPUTS['wordcloud']}")

    # Summary text
    pos = Counter()
    neg = Counter()
    for s in df["aspects"]:
        try:
            for a in json.loads(s):
                if a["sentiment"] == "positive":
                    pos[a["aspect"]] += 1
                elif a["sentiment"] == "negative":
                    neg[a["aspect"]] += 1
        except Exception:
            pass

    with open(OUTPUTS["summary"], "w", encoding="utf-8") as f:
        f.write("Apple Vision Pro Feedback Summary\n")
        f.write("=================================\n")
        f.write(f"Total reviews: {len(df)}\n\n")
        for lbl, val in zip(labels, values):
            f.write(f"{lbl.capitalize():<10}: {val}\n")

        f.write("\nTop Positive Aspects:\n")
        for asp, c in pos.most_common(10):
            f.write(f"  + {asp} ({c})\n")

        f.write("\nTop Negative Aspects:\n")
        for asp, c in neg.most_common(10):
            f.write(f"  - {asp} ({c})\n")

    print(f"✅ Saved {OUTPUTS['summary']}")

# --- Step 7: Main ---
def main():
    print("\n🔹 Loading reviews from feedback.db ...")
    df = load_reviews()
    if df.empty:
        return print("No reviews found!")
    print(f"Found {len(df)} reviews — running analysis...\n")
    df_results = analyze_all(df)
    save_outputs(df_results)
    print("\n🎉 Done! Files generated in this folder.")

if __name__ == "__main__":
    main()

