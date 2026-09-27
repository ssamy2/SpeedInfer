from pathlib import Path

from scripts.check_nvidia_inception_readiness import scan


def _write_site(root: Path, copy: str) -> None:
    frontend = root / "speedinfer" / "frontend"
    frontend.mkdir(parents=True)
    (frontend / "index.html").write_text(copy)


def test_readiness_scan_accepts_supported_software_positioning(tmp_path):
    _write_site(
        tmp_path,
        """<title>Example</title><p>INCORPORATED</dt><dd>2026</p>
        <p>Example L.L.C employs a Lead Developer building an AI inference
        software platform.</p>""",
    )
    report = scan(tmp_path)
    assert report["score"] == 100
    assert report["status"] == "ready_for_human_review"
    assert report["findings"] == []


def test_readiness_scan_blocks_excluded_positioning_and_false_affiliation(tmp_path):
    _write_site(
        tmp_path,
        """<title>Example</title><p>We are a cloud service provider.</p>
        <p>Our NVIDIA strategic partnership powers the platform.</p>""",
    )
    report = scan(tmp_path)
    rule_ids = {item["rule_id"] for item in report["findings"]}
    assert "excluded-cloud-provider-business" in rule_ids
    assert "unapproved-nvidia-relationship" in rule_ids
    assert report["status"] == "needs_changes"


def test_readiness_scan_does_not_flag_reseller_denial(tmp_path):
    _write_site(tmp_path, "<title>Example</title><p>Rather than reselling raw hardware.</p>")
    report = scan(tmp_path)
    rule_ids = {item["rule_id"] for item in report["findings"]}
    assert "excluded-reseller-business" not in rule_ids


def test_readiness_scan_flags_absolute_retention_and_performance_claims(tmp_path):
    _write_site(
        tmp_path,
        """<title>Example</title><p>Strict Zero Data Retention.</p>
        <p>Our gateway has sub-millisecond overhead.</p>""",
    )
    report = scan(tmp_path)
    rule_ids = {item["rule_id"] for item in report["findings"]}
    assert rule_ids == {"absolute-retention-claim", "unverified-performance-claim"}
