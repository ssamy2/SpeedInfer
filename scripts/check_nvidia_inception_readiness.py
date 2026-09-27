#!/usr/bin/env python3
"""Audit public site copy for NVIDIA Inception eligibility and claim risks.

This is a deterministic pre-application review, not an NVIDIA certification tool.
The rules are based on NVIDIA's public Inception FAQ, logo/brand guidance, and
third-party announcement guidance. NVIDIA makes the final eligibility decision.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

OFFICIAL_SOURCES = {
    "inception_faq": "https://www.nvidia.com/en-us/startups/",
    "brand_guidelines": "https://www.nvidia.com/en-us/about-nvidia/legal-info/logo-brand-usage/",
    "announcement_guidelines": "https://www.nvidia.com/en-us/about-nvidia/pr-guidelines/",
}


@dataclass(frozen=True)
class Rule:
    rule_id: str
    severity: str
    pattern: str
    message: str
    source: str


@dataclass(frozen=True)
class Finding:
    rule_id: str
    severity: str
    file: str
    line: int
    excerpt: str
    message: str
    source: str


RULES = (
    Rule(
        "unverified-inception-status",
        "blocker",
        r"\b(?:NVIDIA\s+Inception\s+(?:member|startup|company)|member\s+of\s+NVIDIA\s+Inception|accepted\s+(?:into|by)\s+NVIDIA\s+Inception)\b",
        "Do not claim Inception membership or acceptance until NVIDIA has confirmed it.",
        OFFICIAL_SOURCES["inception_faq"],
    ),
    Rule(
        "unapproved-nvidia-relationship",
        "blocker",
        r"\b(?:NVIDIA.{0,45}(?:strategic\s+)?(?:partner(?:ship)?|alliance|endorsement|sponsor(?:ship)?|joint\s+development)|(?:partner(?:ship)?|alliance|endorsed|sponsored).{0,45}NVIDIA)\b",
        "Avoid language that implies an NVIDIA affiliation, endorsement, or sponsorship.",
        OFFICIAL_SOURCES["brand_guidelines"],
    ),
    Rule(
        "excluded-consulting-business",
        "blocker",
        r"\b(?:we\s+(?:are|provide|offer)|our\s+(?:company|business|team)\s+(?:is|provides))[^\n.]{0,60}(?:consulting\s+(?:company|firm|services?)|outsourced\s+development)\b",
        "The public Inception FAQ excludes consulting and outsourced-development firms.",
        OFFICIAL_SOURCES["inception_faq"],
    ),
    Rule(
        "excluded-cloud-provider-business",
        "blocker",
        r"\b(?:we\s+(?:are|operate\s+as)|our\s+(?:company|business)\s+is)\s+"
        r"(?:an?\s+)?(?:GPU\s+)?cloud\s+(?:service\s+)?provider\b|"
        r"\b(?:rent|rental|lease)\s+(?:NVIDIA\s+)?GPUs?\b",
        "The public Inception FAQ excludes cloud service providers; describe the software "
        "product accurately.",
        OFFICIAL_SOURCES["inception_faq"],
    ),
    Rule(
        "excluded-reseller-business",
        "blocker",
        r"\b(?:we\s+(?:are|operate\s+as)|our\s+(?:company|business)\s+is)\s+"
        r"(?:an?\s+)?(?:hardware\s+|GPU\s+)?(?:reseller|distributor)\b|"
        r"\bwe\s+(?:resell|distribute)\s+(?:raw\s+)?(?:GPU|hardware|cloud\s+capacity)\b",
        "The public Inception FAQ excludes resellers and distributors.",
        OFFICIAL_SOURCES["inception_faq"],
    ),
    Rule(
        "excluded-crypto-business",
        "blocker",
        r"\b(?:cryptocurrency|crypto)\s+(?:startup|company|platform|exchange|token|mining)\b",
        "The public Inception FAQ excludes companies associated with cryptocurrency.",
        OFFICIAL_SOURCES["inception_faq"],
    ),
    Rule(
        "excluded-public-company",
        "blocker",
        r"\b(?:we\s+are|is)\s+(?:a\s+)?publicly[ -]traded\s+company\b",
        "The public Inception FAQ excludes public companies.",
        OFFICIAL_SOURCES["inception_faq"],
    ),
    Rule(
        "unauthorized-nvidia-logo",
        "high",
        r"(?:src|href)=[\"'][^\"']*(?:nvidia[-_ ]?(?:logo|badge)|logo[-_ ]?nvidia)[^\"']*[\"']",
        "NVIDIA logo/badge use requires applicable authorization and must not imply endorsement.",
        OFFICIAL_SOURCES["brand_guidelines"],
    ),
    Rule(
        "absolute-retention-claim",
        "high",
        r"\b(?:strict\s+zero\s+data\s+retention|zero\s+data\s+retention\s+guarantee|prompts?\s+(?:are\s+)?never\s+(?:logged|stored|saved))\b",
        "Avoid blanket retention claims when connected-worker/provider behavior varies.",
        "internal://data-notice-consistency",
    ),
    Rule(
        "unverified-performance-claim",
        "medium",
        r"\b(?:sub[- ]?(?:millisecond|\d+\s*ms)|\d+(?:\.\d+)?x\s+"
        r"(?:faster|speedup)|guaranteed\s+latency)\b",
        "Quantified performance claims should link to reproducible measurements.",
        "internal://claim-evidence",
    ),
    Rule(
        "production-cluster-claim",
        "high",
        r"\b(?:our|the)\s+(?:production|secure)\s+(?:NVIDIA\s+)?GPU\s+clusters?\b|\bcurrently\s+running\s+on\s+NVIDIA\s+GPUs?\b",
        "Do not imply an active production GPU cluster unless it is currently verifiable.",
        "internal://claim-evidence",
    ),
)


EVIDENCE = (
    ("working_website", 10, r"<title>[^<]+</title>"),
    ("incorporated_company", 15, r"\b(?:INCORPORATED|L\.L\.C|Limited Liability Company)\b"),
    ("company_under_ten_years", 10, r"\bINCORPORATED</dt><dd>20(?:1[7-9]|2\d)\b"),
    ("developer_on_team", 15, r"\b(?:Lead Developer|Software Developer|Engineer)\b"),
    (
        "software_product_positioning",
        10,
        r"\b(?:AI inference software|software platform|multi-model gateway|inference gateway)\b",
    ),
)


def public_files(root: Path) -> list[Path]:
    frontend = root / "speedinfer" / "frontend"
    return sorted(
        path
        for path in frontend.rglob("*")
        if path.is_file() and path.suffix.lower() in {".html", ".js", ".md"}
    )


def scan(root: Path) -> dict[str, object]:
    files = public_files(root)
    findings: list[Finding] = []
    combined = ""
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        combined += "\n" + text
        relative = str(path.relative_to(root))
        for line_number, line in enumerate(text.splitlines(), start=1):
            if line.lstrip().startswith("<!--"):
                continue
            for rule in RULES:
                match = re.search(rule.pattern, line, flags=re.IGNORECASE)
                if match:
                    excerpt = re.sub(r"\s+", " ", line.strip())
                    findings.append(
                        Finding(
                            rule.rule_id,
                            rule.severity,
                            relative,
                            line_number,
                            excerpt[:240],
                            rule.message,
                            rule.source,
                        )
                    )

    evidence = []
    evidence_score = 0
    for evidence_id, points, pattern in EVIDENCE:
        present = bool(re.search(pattern, combined, flags=re.IGNORECASE))
        evidence.append({"id": evidence_id, "points": points, "present": present})
        if present:
            evidence_score += points

    penalties = {"blocker": 20, "high": 10, "medium": 4}
    claim_penalty = min(40, sum(penalties[item.severity] for item in findings))
    score = evidence_score + (40 - claim_penalty)
    blockers = sum(item.severity == "blocker" for item in findings)
    status = "ready_for_human_review" if score >= 90 and blockers == 0 else "needs_changes"
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "scope": [str(path.relative_to(root)) for path in files],
        "score": score,
        "status": status,
        "method": {
            "eligibility_evidence_points": evidence_score,
            "claim_safety_points": 40 - claim_penalty,
            "maximum": 100,
            "note": "Pre-application heuristic only; NVIDIA makes the final decision.",
        },
        "eligibility_evidence": evidence,
        "findings": [asdict(item) for item in findings],
        "official_sources": OFFICIAL_SOURCES,
    }


def print_text(report: dict[str, object]) -> None:
    print(f"NVIDIA Inception pre-application readiness: {report['score']}%")
    print(f"Status: {report['status']}")
    method = report["method"]
    print(
        f"Evidence: {method['eligibility_evidence_points']}/60 | "
        f"Claim safety: {method['claim_safety_points']}/40"
    )
    print("\nEligibility evidence:")
    for item in report["eligibility_evidence"]:
        mark = "PASS" if item["present"] else "MISSING"
        print(f"  [{mark}] {item['id']} ({item['points']} points)")
    print("\nFindings:")
    if not report["findings"]:
        print("  None")
    for item in report["findings"]:
        print(
            f"  [{item['severity'].upper()}] {item['rule_id']} "
            f"{item['file']}:{item['line']}"
        )
        print(f"    {item['message']}")
    print("\nThis score is not approval, certification, or a prediction of NVIDIA's decision.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero unless status is ready_for_human_review.",
    )
    args = parser.parse_args()
    report = scan(args.root.resolve())
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print_text(report)
    return int(args.strict and report["status"] != "ready_for_human_review")


if __name__ == "__main__":
    sys.exit(main())
