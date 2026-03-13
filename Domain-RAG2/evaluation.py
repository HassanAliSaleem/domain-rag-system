import json
import time
from datetime import datetime, timezone
from rag_pipeline import retrieve, reset
from llm_generator import generate_answer

QUESTIONS = [
    {"id": 1,  "type": "factual",   "q": "What is the company's annual leave policy?"},
    {"id": 2,  "type": "factual",   "q": "What are the standard working hours?"},
    {"id": 3,  "type": "factual",   "q": "What is the dress code policy?"},
    {"id": 4,  "type": "factual",   "q": "How many sick days are employees entitled to?"},
    {"id": 5,  "type": "factual",   "q": "What is the probation period for new employees?"},
    {"id": 6,  "type": "factual",   "q": "What is the remote work policy?"},
    {"id": 7,  "type": "factual",   "q": "How does the company handle overtime?"},
    {"id": 8,  "type": "factual",   "q": "What benefits does the company offer?"},
    {"id": 9,  "type": "factual",   "q": "What is the policy on workplace harassment?"},
    {"id": 10, "type": "factual",   "q": "How are performance reviews conducted?"},
    {"id": 11, "type": "multi-hop", "q": "If I use all sick days, can I use annual leave for medical reasons?"},
    {"id": 12, "type": "multi-hop", "q": "Can a remote worker request overtime pay?"},
    {"id": 13, "type": "multi-hop", "q": "How does probation period affect leave entitlement?"},
    {"id": 14, "type": "multi-hop", "q": "What happens if a harassment complaint is filed during probation?"},
    {"id": 15, "type": "multi-hop", "q": "Can performance review outcomes impact remote work eligibility?"},
    {"id": 16, "type": "multi-hop", "q": "How do working hours differ for part-time vs full-time employees?"},
    {"id": 17, "type": "multi-hop", "q": "What if I want to extend probation and take leave simultaneously?"},
    {"id": 18, "type": "multi-hop", "q": "Does the dress code apply when working remotely?"},
    {"id": 19, "type": "multi-hop", "q": "How are benefits affected during extended sick leave?"},
    {"id": 20, "type": "multi-hop", "q": "Can overtime during probation count toward performance evaluation?"},
    {"id": 21, "type": "tricky",    "q": "What is the CEO's favorite color?"},
    {"id": 22, "type": "tricky",    "q": "Can I bring my pet to the office?"},
    {"id": 23, "type": "tricky",    "q": "What is the company's stock price?"},
    {"id": 24, "type": "tricky",    "q": "Tell me about the company's competitors."},
    {"id": 25, "type": "tricky",    "q": "What will the leave policy be next year?"},
    {"id": 26, "type": "tricky",    "q": "Is the company planning layoffs?"},
    {"id": 27, "type": "tricky",    "q": "Summarize every policy in one sentence."},
    {"id": 28, "type": "tricky",    "q": "What if I disagree with the leave policy?"},
    {"id": 29, "type": "tricky",    "q": "Can I override the dress code with manager approval?"},
    {"id": 30, "type": "tricky",    "q": "What is the meaning of life?"},
]

BAD_PHRASES = ["i think", "probably", "i believe", "as far as i know", "generally speaking"]


def run():
    reset()
    records = []
    total_ms = 0

    print("=" * 65)
    print("EVALUATION REPORT —", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    print("=" * 65)

    for item in QUESTIONS:
        t0   = time.perf_counter()
        hits = retrieve(item["q"])
        resp = generate_answer(item["q"], hits)
        ms   = (time.perf_counter() - t0) * 1000
        total_ms += ms

        ans_lower    = resp["answer"].lower()
        has_citation = len(resp["citations"]) > 0
        grounded     = has_citation and resp["confidence"] != "low"
        hallucinated = any(p in ans_lower for p in BAD_PHRASES)

        records.append({
            "id": item["id"], "type": item["type"], "question": item["q"],
            "answer": resp["answer"][:200], "confidence": resp["confidence"],
            "citations": resp["citations"], "has_citations": has_citation,
            "grounded": grounded, "hallucination": hallucinated,
            "latency_ms": round(ms, 2),
        })

        icon = "✅" if grounded and not hallucinated else "⚠️"
        print(f"\n{icon} Q{item['id']:02d} [{item['type']:<9}] {item['q'][:55]}")
        print(f"   conf={resp['confidence']} | cited={has_citation} | grounded={grounded} | {ms:.0f}ms")

    print("\n" + "=" * 65)
    for qtype in ["factual", "multi-hop", "tricky"]:
        sub = [r for r in records if r["type"] == qtype]
        print(f"\n{qtype.upper()} ({len(sub)}):")
        print(f"  Grounded:    {sum(r['grounded'] for r in sub)}/{len(sub)}")
        print(f"  Hallucinate: {sum(r['hallucination'] for r in sub)}/{len(sub)}")
        print(f"  Citations:   {sum(r['has_citations'] for r in sub)}/{len(sub)}")
        print(f"  Avg Latency: {sum(r['latency_ms'] for r in sub)/len(sub):.0f} ms")

    print(f"\nOVERALL avg latency: {total_ms/len(records):.0f} ms")
    print("=" * 65)

    with open("evaluation_report.json", "w") as f:
        json.dump(records, f, indent=2)
    print("Report saved → evaluation_report.json")


if __name__ == "__main__":
    run()
