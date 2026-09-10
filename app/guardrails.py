import re
from typing import List, Tuple
from .models import Verdict

DISCLAIMER = (
    "Information from MOHFW Standard Treatment Guidelines for education only. "
    "Consult a qualified clinician for personal advice."
)

EMERGENCY_PATTERNS: List[str] = [
    r"suicid|\bkill myself\b|\bend my life\b|\bself[\s-]?harm\b",
    r"overdose|took too many pills",
    r"uncontrolled bleeding|bleeding heavily|severe bleeding|massive blood loss",
    r"heart attack|cardiac arrest|anaphylaxis|\bnot breathing\b|\bunconscious\b",
    r"poisoning|\bpoison\b.*\b(ingest|drink|swallow|exposure)\b",
    r"(i have|my|i am|i feel|me).*(chest pain|difficulty breathing|stroke|bleeding|poison|unconscious|overdose)",
    r"(chest pain|difficulty breathing|stroke|bleeding|shortness of breath).*(emergency|urgent|severe|help now|call.*ambulance|immediately)",
    r"emergency.*(chest pain|difficulty breathing|stroke|bleeding|unconscious)",
    r"face droop|slurred speech.*sudden|arm weakness.*sudden",
]

PERSONALIZED_PATTERNS: List[str] = [
    r"\bi have\b",
    r"\bmy (dose|dosage|symptoms?|fever|pain|bp|blood pressure|report|test|medicine|medication|prescription|case)\b",
    r"should i take",
    r"diagnose me",
    r"diagnose my",
    r"\bprescribe\b",
    r"dosage for me",
    r"what dose should i",
    r"what should i (take|do) for my",
    r"\bfor me\b",
    r"\bam i\b.*(diagnos|pregnant|sick)",
    r"\bmy (chest|stomach|head)\b.*pain",
    r"should i.*for my symptoms",
]

OUT_OF_SCOPE_PATTERNS: List[str] = [
    r"\bcricket\b",
    r"football.*score|world cup.*rules",
    r"python.*(code|program|script|function)",
    r"\bweather\b",
    r"quantum physics",
    r"lunar soil",
    r"\bmovie\b|\bstock price\b|\brecipe\b|\btravel visa\b|\bastrology\b",
    r"how to (code|cook|fix my car)",
]

INTENT_REGISTRY: List[Tuple[str, List[str]]] = [
    ("REFUSE_EMERGENCY", EMERGENCY_PATTERNS),
    ("REFUSE_PERSONALIZED", PERSONALIZED_PATTERNS),
    ("REFUSE_OUT_OF_SCOPE", OUT_OF_SCOPE_PATTERNS),
]

UNSAFE_OUTPUT_PATTERNS: List[str] = [
    r"you should take .*mg",
    r"you have (malaria|tuberculosis|diabetes|hypertension|cancer|stroke|rabies|tb|dengue|anemia)",
    r"you are diagnosed",
    r"your diagnosis is",
    r"take \d+\s?mg.*for your",
    r"i diagnose you",
    r"your prescribed dose is",
]

SAFETY_PATTERNS: List[str] = UNSAFE_OUTPUT_PATTERNS

REFUSAL_TEXT = {
    "REFUSE_EMERGENCY": (
        "This sounds like a possible emergency. "
        "Call your local emergency number or go to the nearest emergency department now. "
        "I cannot provide emergency care guidance."
    ),
    "REFUSE_PERSONALIZED": (
        "I cannot provide personal medical advice, diagnosis, or dosage for your case. "
        "Please consult a qualified clinician. I can summarize what the MOHFW guidelines say in general."
    ),
    "REFUSE_OUT_OF_SCOPE": (
        "This question is outside the MOHFW guideline corpus I can cite. "
        "I only answer questions grounded in the included guidelines."
    ),
    "REFUSE_LOW_CONFIDENCE": (
        "I don't have enough information in the guidelines to answer this."
    ),
}

_COMPILED_REGISTRY = [
    (verdict, [re.compile(p, re.IGNORECASE) for p in pats])
    for verdict, pats in INTENT_REGISTRY
]

_COMPILED_UNSAFE = [re.compile(p, re.IGNORECASE) for p in UNSAFE_OUTPUT_PATTERNS]


def classify_input(query: str) -> Verdict:
    text = query or ""
    for verdict, compiled in _COMPILED_REGISTRY:
        for rx in compiled:
            if rx.search(text):
                return verdict  # type: ignore[return-value]
    return "PROCEED"


def check_output(answer: str) -> bool:
    text = answer or ""
    for rx in _COMPILED_UNSAFE:
        if rx.search(text):
            return False
    return True
