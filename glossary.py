# glossary.py
"""Plain-English F1 jargon for newer fans. No Streamlit import."""
import re

# term -> (aliases to look for in text, one-line explanation)
GLOSSARY = {
    "Safety Car": (
        ["safety car"],
        "An official car leads the field around the track at slow speed after an incident, "
        "bunching everyone up. Overtaking is not allowed, and teams often pit for cheap tyres.",
    ),
    "VSC": (
        ["vsc", "virtual safety car"],
        "Virtual Safety Car: everyone must slow to a set speed but stay in place, with no "
        "actual car leading. It is used for smaller incidents.",
    ),
    "Red flag": (
        ["red flag", "red-flag"],
        "The race is stopped completely and cars return to the pit lane. Teams can change "
        "tyres for free, so it can reshuffle the race.",
    ),
    "Undercut": (
        ["undercut"],
        "Pitting earlier than the car ahead, then using fresh tyres to set fast laps. If it "
        "works, you come out in front once they pit.",
    ),
    "Overcut": (
        ["overcut"],
        "The opposite of an undercut: staying out longer while the car ahead pits, "
        "hoping to go faster on clear track.",
    ),
    "Pit stop": (
        ["pit stop", "pit-stop", "pitted", "pits", "pit lane", "pit window"],
        "A short stop in the pit lane to change tyres, usually taking 2 to 3 seconds when "
        "stationary and around 20 seconds in total.",
    ),
    "Stint": (
        ["stint"],
        "The stretch of laps a driver does on one set of tyres between pit stops.",
    ),
    "Tyre compounds": (
        ["soft tyre", "soft tyres", "medium tyre", "medium tyres", "hard tyre", "hard tyres",
         "softs", "mediums", "hards", "compound"],
        "Soft tyres are fastest but wear out quickly, hard tyres last longest but are slower, "
        "and medium sits between. Intermediate and wet tyres are for rain.",
    ),
    "Tyre degradation": (
        ["degradation", "tyre wear", "tire wear", "tyre deg"],
        "Tyres lose grip as they wear, so lap times slowly get worse until the driver pits.",
    ),
    "Intermediates / Wets": (
        ["intermediate", "intermediates", "wet tyres", "full wets"],
        "Tyres with grooves for a damp (intermediates) or soaked (full wets) track.",
    ),
    "DRS": (
        ["drs"],
        "A flap on the rear wing that opens on certain straights to give a driver chasing "
        "within one second extra speed to overtake.",
    ),
    "Lapped": (
        ["lapped", "a lap down", "laps down"],
        "A driver is lapped when the leader catches and passes them, putting them a full "
        "lap behind.",
    ),
    "DNF": (
        ["dnf", "did not finish", "retired", "retirement"],
        "Did Not Finish: the driver's race ended early because of a crash or a car problem.",
    ),
    "Gap / Interval": (
        ["gap", "interval"],
        "Gap is the time behind the race leader. Interval is the time behind the car "
        "directly ahead.",
    ),
    "Fastest lap": (
        ["fastest lap"],
        "The quickest single lap of the race so far. At the end it can earn a bonus "
        "championship point for a driver in the top 10, if the rules in that season give it.",
    ),
    "Track status / flags": (
        ["yellow flag", "green flag", "track clear"],
        "Yellow means danger ahead so no overtaking in that area. Green means the track "
        "is clear and racing is back on.",
    ),
}


def _pattern(aliases):
    return re.compile(r"\b(" + "|".join(re.escape(a) for a in aliases) + r")\b", re.IGNORECASE)


_PATTERNS = {term: _pattern(aliases) for term, (aliases, _) in GLOSSARY.items()}


def find_terms(text):
    """Glossary terms mentioned in `text`, in order of first appearance."""
    if not text:
        return []
    hits = []
    for term, pat in _PATTERNS.items():
        m = pat.search(text)
        if m:
            hits.append((m.start(), term))
    hits.sort()
    return [(term, GLOSSARY[term][1]) for _, term in hits]


def glossary_for_prompt():
    return "\n".join(f"- {term}: {definition}" for term, (_, definition) in GLOSSARY.items())