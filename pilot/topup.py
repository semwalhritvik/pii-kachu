"""Append 14 fictional top-up cases (mixed-fault, detected=false, quasi) to data/pilot/pilot.jsonl. Idempotent."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.judge.derive import derive_verdict, load_purposes  # noqa: E402
from src.judge.render import DETECTOR, locate, render  # noqa: E402

PATH = ROOT / "data/pilot/pilot.jsonl"
P = {n: f"P{n}_" + s for n, s in ((1, "cohort"), (2, "geographic"), (3, "longitudinal"), (4, "academic"), (5, "text"), (6, "incident"))}


def S(text, label, action, role="other", **kw):
    return dict(text=text, gold_label=label, action=action, role=role, **kw)


CASES = [
    # --- mixed-fault (6)
    (1, "csv_row", "csv_row_mixed", "row,name,email,school,track\n17,Odalys Vermeer,odalys.vermeer@brindlemail.test,Wrenfield Collegiate,robotics",
     [S("Odalys Vermeer", "PERSON_PRIVATE", "redact", "student"), S("odalys.vermeer@brindlemail.test", "EMAIL", "keep"),
      S("Wrenfield Collegiate", "ORG", "redact")], None),
    (2, "attendance_note", "xp_mixed_over_under",
     "Attendance note: Ingrid Solheim missed first period on 27 February; her guardian said the bus never stopped at 48 Juniper Wharf and she walked to Oxbow Green instead.",
     [S("Ingrid Solheim", "PERSON_PRIVATE", "redact", "student"), S("27 February", "DATE_OTHER", "keep"),
      S("48 Juniper Wharf", "STREET_ADDRESS", "keep"), S("Oxbow Green", "LOCATION_OTHER", "redact")], None),
    (3, "email", "xp_mixed_over_under",
     "Subject: Follow-up for Priyanka Rautenbach\n\nPriyanka Rautenbach, born 9 February 2009, transferred from Mossbank Primary in the spring. Her file number is 55-2041.",
     [S("Priyanka Rautenbach", "PERSON_PRIVATE", "pseudonymize", "student", entity="priyanka rautenbach"),
      S("Priyanka Rautenbach", "PERSON_PRIVATE", "pseudonymize", "student", entity="priyanka rautenbach"),
      S("9 February 2009", "DOB", "keep"), S("Mossbank Primary", "ORG", "redact"), S("55-2041", "ID_NUM", "pseudonymize")], None),
    (4, "essay", "essay_mixed_figure_over_student_under",
     "Name: Calloway Brisbane\nPeriod 3 History\n\nSojourner Truth traveled the country giving speeches and argued that freedom meant nothing without a voice. I think her words still matter because they show how one person can move a whole country.",
     [S("Calloway Brisbane", "PERSON_PRIVATE", "keep", "student"), S("Sojourner Truth", "PERSON_PUBLIC", "redact", "public_figure")], None),
    (5, "essay", "essay_mixed_figure_over_student_under",
     "Student: Isidore Fenwick-Bray. I chose Marie Curie because she won two Nobel Prizes in different sciences, and I wrote this at home where my science teacher Mr. Venn first told me about her.",
     [S("Isidore Fenwick-Bray", "PERSON_PRIVATE", "keep", "student"), S("Marie Curie", "PERSON_PUBLIC", "redact", "public_figure"),
      S("Venn", "PERSON_PRIVATE", "redact", "staff")], None),
    (6, "counseling_note", "xp_mixed_over_under",
     "Incident note: Eamon Pettibone was named as a witness at Saltmarsh Junior High on 11 April, and the form lists his phone as 555-0193.",
     [S("Eamon Pettibone", "PERSON_PRIVATE", "redact", "student"), S("Saltmarsh Junior High", "ORG", "redact"),
      S("11 April", "DATE_OTHER", "keep"), S("555-0193", "PHONE", "keep")], None),
    # --- detected=false (4)
    (1, "email", "missed_phone",
     "Hello all, Tomasz Kierans (tomasz.k@fennelwood.test) asked to move the cohort check-in earlier. Please ring him on 555-0147 before Friday.",
     [S("Tomasz Kierans", "PERSON_PRIVATE", "redact", "staff"), S("tomasz.k@fennelwood.test", "EMAIL", "redact"),
      S("555-0147", "PHONE", "keep", detected=False)], None),
    (2, "counseling_note", "missed_address",
     "Counseling note: Wendell Aldous asked to switch homerooms. He lives at 912 Cormorant Lane and mentioned the Cobbleside Library as a quiet place to study.",
     [S("Wendell Aldous", "PERSON_PRIVATE", "redact", "student"), S("912 Cormorant Lane", "STREET_ADDRESS", "keep", detected=False),
      S("Cobbleside Library", "LOCATION_OTHER", "keep")], None),
    (3, "attendance_note", "missed_dob",
     "Attendance note: Bartholomew Quill, born 21 November 2010, was absent on 12 October and the front office logged a call from his mother in Redfern.",
     [S("Bartholomew Quill", "PERSON_PRIVATE", "pseudonymize", "student", entity="bartholomew quill"),
      S("21 November 2010", "DOB", "keep", detected=False), S("12 October", "DATE_OTHER", "keep"), S("Redfern", "LOCATION_OTHER", "keep")], None),
    (4, "email", "missed_url",
     "Dear Ms. Okonjo-Reyes, my portfolio is at www.junipercrow-art.test/dellamae if you want to see the draft. Thank you, Della Mae Hartsock.",
     [S("Okonjo-Reyes", "PERSON_PRIVATE", "redact", "staff"), S("www.junipercrow-art.test/dellamae", "URL_PERSONAL", "keep", detected=False),
      S("Della Mae Hartsock", "PERSON_PRIVATE", "redact", "student")], None),
    # --- quasi (4)
    (2, "counseling_note", "incident_quasi",
     "Counseling note: Mirela Dunmore's cousin was hurt at the Halyard Aquatic Centre on 14 July during the Ashgrove Swim Club gala.",
     [S("Mirela Dunmore", "PERSON_PRIVATE", "redact", "student"), S("Halyard Aquatic Centre", "LOCATION_OTHER", "keep"),
      S("14 July", "DATE_OTHER", "keep"), S("Ashgrove Swim Club", "ORG", "keep")], [[2, 3, 4]]),
    (6, "counseling_note", "incident_quasi",
     "Incident note: Zeynep Halvorsen reported that a car was stolen outside Thistledown Diner on 6 January during the Brackwater Rotary fundraiser.",
     [S("Zeynep Halvorsen", "PERSON_PRIVATE", "redact", "parent"), S("Thistledown Diner", "LOCATION_OTHER", "keep"),
      S("6 January", "DATE_OTHER", "keep"), S("Brackwater Rotary", "ORG", "keep")], [[2, 3, 4]]),
    (6, "email", "incident_quasi_mitigated",
     "Email: Reggie Stonebridge saw the fire at Windmere Pier on 19 August while volunteering for the Corvid Hollow Rescue Squad.",
     [S("Reggie Stonebridge", "PERSON_PRIVATE", "redact", "other"), S("Windmere Pier", "LOCATION_OTHER", "keep"),
      S("19 August", "DATE_OTHER", "redact"), S("Corvid Hollow Rescue Squad", "ORG", "keep")], [[2, 3, 4]]),
    (5, "attendance_note", "incident_quasi_mitigated",
     "Attendance note: Leopold Marchetti was removed from the bus after a fight at Dovecote Plaza on 2 December involving members of the Tamarack Rovers youth team.",
     [S("Leopold Marchetti", "PERSON_PRIVATE", "redact", "student"), S("Dovecote Plaza", "LOCATION_OTHER", "redact"),
      S("2 December", "DATE_OTHER", "keep"), S("Tamarack Rovers", "ORG", "keep")], [[2, 3, 4]]),
]


def build(n0):
    purposes, out = load_purposes(), []
    for k, (pn, fmt, hct, text, spans, quasi) in enumerate(CASES):
        c = {"case_id": f"pilot-{n0 + k:03d}", "purpose": P[pn], "format": fmt, "hard_case_type": hct, "original": text, "spans": []}
        for i, s in enumerate(spans, 1):
            c["spans"].append({"id": f"s{i}", "text": s["text"], "gold_label": s["gold_label"], "action": s["action"], "role": s["role"],
                               "detector_label": DETECTOR[s["gold_label"]], **{x: s[x] for x in ("entity", "detected") if x in s}})
        locate(c)
        if quasi:
            c["quasi_groups"] = [[f"s{i}" for i in g] for g in quasi]
        d = derive_verdict(c, purposes)
        c.update(gold_verdict=d["verdict"], offending_span_ids=d["offending_span_ids"], fault_type=d["fault_type"], redacted=render(c))
        out.append(c)
    return out


if __name__ == "__main__":
    have = [json.loads(l) for l in PATH.read_text().splitlines() if l.strip()]
    if any(c["case_id"] == "pilot-089" for c in have):
        sys.exit("already applied")
    new = build(1 + max(int(c["case_id"].split("-")[1]) for c in have))
    with open(PATH, "a") as f:
        f.writelines(json.dumps(c) + "\n" for c in new)
    print("added", len(new))
