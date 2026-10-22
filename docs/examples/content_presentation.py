"""Offline template illustration. No learning, model calls, or semantic scoring.

Run from the repository root: python docs/examples/content_presentation.py
"""

REGISTERS = ("introductory", "working", "technical")
FACTS = {"place": "Bay 3", "time": "18:45", "requirement": "printed ID"}


def render(facts, register="working"):
    """Change fixed wording while preserving the three supplied literal values.

    This does not verify the values against reality. Wrong input stays wrong.
    """
    if set(facts) != set(FACTS) or any(
        not isinstance(value, str) or not value.strip() for value in facts.values()
    ):
        raise ValueError("Provide exactly three nonblank facts: place, time, requirement")
    templates = {
        "introductory": "Go to {place} at {time}. Bring {requirement}.",
        "working": "Check-in: {place}, {time}. Required: {requirement}.",
        "technical": "Location={place}; time={time}; requirement={requirement}.",
    }
    if register not in templates:
        raise ValueError("Choose introductory, working, or technical")
    return templates[register].format(**facts)


def missing_literals(facts, text):
    """Only an exact-string checklist, not a natural-language entailment test."""
    return [value for value in facts.values() if value not in text]


def main():
    print("ILLUSTRATIVE TEMPLATES ONLY: no model or comprehension experiment")
    print("Fact ledger:", FACTS)
    for register in REGISTERS:
        text = render(FACTS, register)
        print(f"{register}: {text} | missing literals: {missing_literals(FACTS, text)}")
    shorter = "Check in this evening. Bring your ID."
    print(f"shorter but incomplete: {shorter}")
    print("Missing exact facts:", missing_literals(FACTS, shorter))
    wrong = dict(FACTS, place="Bay 9")
    print("Consistent but wrong input:", render(wrong))
    print("Missing relative to original ledger:", missing_literals(FACTS, render(wrong)))


if __name__ == "__main__":
    main()
