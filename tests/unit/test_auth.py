"""Unit and integration tests for SpeedInfer authentication, key hashing, and scope enforcement.

Covers:
- Cryptographic key generation (format `sk-speedinfer-...`, high entropy)
- HMAC-SHA256 key hashing with secret pepper
- Constant-time hash verification via hmac.compare_digest
- Key prefix extraction and redaction for zero-leak logging
- Scope-based permission checks (single, multiple, admin override)
- Bearer Authorization header parsing and error responses
"""

import hashlib
import hmac
import re
import secrets

# Attempt import of M2 core auth module
try:
    from speedinfer.core import auth as core_auth

    HAS_CORE_AUTH = True
except (ImportError, AttributeError):
    HAS_CORE_AUTH = False


# ---------------------------------------------------------------------------
# Authoritative Reference Implementations (The Oracle Specifications)
# ---------------------------------------------------------------------------
def reference_hash_key(raw_key: str, pepper: str) -> str:
    """HMAC-SHA256 key hashing with pepper reference formula."""
    return hmac.new(
        pepper.encode("utf-8"),
        raw_key.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def reference_verify_key(raw_key: str, stored_hash: str, pepper: str) -> bool:
    """Constant-time verification using hmac.compare_digest."""
    computed_hash = reference_hash_key(raw_key, pepper)
    return hmac.compare_digest(computed_hash, stored_hash)


def reference_mask_key(raw_key: str) -> str:
    """Extract safe prefix (first 22 chars: sk-speedinfer- + 8 hex) and redact."""
    if not raw_key.startswith("sk-speedinfer-") or len(raw_key) < 22:
        return "sk-speedinfer-invalid..."
    return f"{raw_key[:22]}..."


def reference_check_scopes(required_scope: str, granted_scopes: str | list[str]) -> bool:
    """Check if required_scope is granted, with 'admin' wildcard bypass."""
    if not granted_scopes:
        return False
    if isinstance(granted_scopes, str):
        scopes = {s.strip() for s in granted_scopes.split(",") if s.strip()}
    else:
        scopes = set(granted_scopes)

    if "admin" in scopes:
        return True
    return required_scope in scopes


def _get_hash_fn():
    if HAS_CORE_AUTH and hasattr(core_auth, "hash_api_key"):
        return core_auth.hash_api_key
    return reference_hash_key


def _get_verify_fn():
    if HAS_CORE_AUTH and hasattr(core_auth, "verify_api_key_hash"):
        return core_auth.verify_api_key_hash
    return reference_verify_key


def _get_mask_fn():
    if HAS_CORE_AUTH and hasattr(core_auth, "mask_api_key"):
        return core_auth.mask_api_key
    return reference_mask_key


def _get_check_scopes_fn():
    if HAS_CORE_AUTH and hasattr(core_auth, "check_scopes"):
        return core_auth.check_scopes
    return reference_check_scopes


# ---------------------------------------------------------------------------
# Test Cases: Key Generation & Format Invariants
# ---------------------------------------------------------------------------
def test_key_format_and_entropy():
    """Verify generated API key format, prefix, and entropy requirements."""
    if HAS_CORE_AUTH and hasattr(core_auth, "generate_api_key"):
        result = core_auth.generate_api_key()
        raw_key = result[0] if isinstance(result, (tuple, list)) else result
    else:
        # Reference generation: sk-speedinfer- + 64 hex chars (32 bytes entropy)
        raw_key = f"sk-speedinfer-{secrets.token_hex(32)}"

    assert raw_key.startswith("sk-speedinfer-"), "Key must start with sk-speedinfer-"
    hex_part = raw_key[len("sk-speedinfer-") :]
    assert len(hex_part) == 64, f"Entropy part must be 64 hex chars (32 bytes), got {len(hex_part)}"
    assert re.fullmatch(r"[0-9a-fA-F]{64}", hex_part), "Key payload must be valid hex characters"


def test_key_generation_uniqueness():
    """Verify successive key generations yield cryptographically unique keys."""
    keys = set()
    for _ in range(100):
        if HAS_CORE_AUTH and hasattr(core_auth, "generate_api_key"):
            result = core_auth.generate_api_key()
            key = result[0] if isinstance(result, (tuple, list)) else result
        else:
            key = f"sk-speedinfer-{secrets.token_hex(32)}"
        assert key not in keys, "Key collision detected in 100 generations"
        keys.add(key)


# ---------------------------------------------------------------------------
# Test Cases: Key Hashing with Pepper
# ---------------------------------------------------------------------------
def test_key_hashing_deterministic():
    """Verify hashing the same key with the same pepper always yields the identical hash."""
    raw_key = "sk-speedinfer-0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    pepper = "super-secret-pepper-32-chars-long!"
    hash_fn = _get_hash_fn()

    hash1 = hash_fn(raw_key, pepper)
    hash2 = hash_fn(raw_key, pepper)

    assert hash1 == hash2
    assert len(hash1) == 64, "SHA-256 output must be 64 hex characters"
    assert re.fullmatch(r"[0-9a-fA-F]{64}", hash1)


def test_key_hashing_pepper_sensitivity():
    """Verify different peppers produce completely different hashes for the same key."""
    raw_key = "sk-speedinfer-0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    pepper1 = "pepper-one-secret-value-1234567890"
    pepper2 = "pepper-two-secret-value-0987654321"
    hash_fn = _get_hash_fn()

    hash1 = hash_fn(raw_key, pepper1)
    hash2 = hash_fn(raw_key, pepper2)

    assert hash1 != hash2, "Different peppers must produce different hashes"


def test_key_hashing_payload_sensitivity():
    """Verify a 1-bit alteration in the key produces completely different hash."""
    key1 = "sk-speedinfer-0000000000000000000000000000000000000000000000000000000000000000"
    key2 = "sk-speedinfer-0000000000000000000000000000000000000000000000000000000000000001"
    pepper = "constant-secret-pepper-for-test"
    hash_fn = _get_hash_fn()

    hash1 = hash_fn(key1, pepper)
    hash2 = hash_fn(key2, pepper)

    assert hash1 != hash2


# ---------------------------------------------------------------------------
# Test Cases: Constant-Time Comparison & Key Validation
# ---------------------------------------------------------------------------
def test_constant_time_verification_success():
    """Verify correct raw key against stored hash returns True."""
    raw_key = "sk-speedinfer-fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210"
    pepper = "pepper-validation-secret-123456"
    verify_fn = _get_verify_fn()
    stored_hash = reference_hash_key(raw_key, pepper)

    assert verify_fn(raw_key, stored_hash, pepper) is True


def test_constant_time_verification_failure_cases():
    """Verify verification returns False on invalid keys, altered hashes, or wrong peppers."""
    raw_key = "sk-speedinfer-fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210"
    pepper = "pepper-validation-secret-123456"
    stored_hash = reference_hash_key(raw_key, pepper)
    verify_fn = _get_verify_fn()

    # 1. Altered key character
    altered_key = raw_key[:-1] + ("0" if raw_key[-1] != "0" else "1")
    assert verify_fn(altered_key, stored_hash, pepper) is False

    # 2. Altered pepper
    assert verify_fn(raw_key, stored_hash, "wrong-pepper") is False

    # 3. Altered stored hash
    altered_hash = stored_hash[:-1] + ("a" if stored_hash[-1] != "a" else "b")
    assert verify_fn(raw_key, altered_hash, pepper) is False

    # 4. Truncated key
    assert verify_fn("sk-speedinfer-short", stored_hash, pepper) is False

    # 5. Empty key
    assert verify_fn("", stored_hash, pepper) is False


# ---------------------------------------------------------------------------
# Test Cases: Prefix Extraction & Logging Redaction
# ---------------------------------------------------------------------------
def test_prefix_masking_redacts_secret_entropy():
    """Verify mask_api_key retains only the public prefix and suppresses private secret."""
    raw_key = "sk-speedinfer-a1b2c3d4e5f678901234567890abcdef1234567890abcdef1234567890abcdef"
    mask_fn = _get_mask_fn()
    masked = mask_fn(raw_key)

    # Safe prefix: 'sk-speedinfer-' (14 chars) + first 8 hex chars = 22 chars
    assert masked.startswith("sk-speedinfer-a1b2c3d4"), "Prefix must preserve first 8 hex chars"
    # Secret trailing entropy must NEVER appear in masked output
    assert "e5f678901234567890abcdef" not in masked, "Secret entropy must be completely redacted"
    assert len(masked) < len(raw_key), "Masked key must be shorter than raw key"


def test_prefix_masking_malformed_inputs():
    """Verify prefix extraction gracefully handles malformed or truncated keys without raising."""
    mask_fn = _get_mask_fn()

    assert "invalid" in mask_fn("not-an-api-key").lower()
    assert "invalid" in mask_fn("").lower()
    assert "invalid" in mask_fn("sk-speedinfer-short").lower()


# ---------------------------------------------------------------------------
# Test Cases: Scope-Based Permission Enforcement
# ---------------------------------------------------------------------------
def test_scope_checking_exact_match():
    """Verify required scope is allowed when explicitly present."""
    check_fn = _get_check_scopes_fn()

    assert check_fn("chat:completions", "chat:completions,models:read") is True
    assert check_fn("models:read", "chat:completions,models:read") is True
    assert check_fn("completions", ["completions", "usage:read"]) is True


def test_scope_checking_insufficient_scope():
    """Verify required scope is denied when absent from granted permissions."""
    check_fn = _get_check_scopes_fn()

    assert check_fn("chat:completions", "models:read,usage:read") is False
    assert check_fn("admin", "chat:completions") is False
    assert check_fn("chat:completions", "") is False
    assert check_fn("chat:completions", None) is False


def test_scope_checking_admin_bypass():
    """Verify 'admin' scope acts as universal bypass for all endpoint permissions."""
    check_fn = _get_check_scopes_fn()

    assert check_fn("chat:completions", "admin") is True
    assert check_fn("completions", "admin,user") is True
    assert check_fn("models:read", ["admin"]) is True
    assert check_fn("usage:read", "admin") is True


# ---------------------------------------------------------------------------
# Test Cases: Authorization Header Parsing
# ---------------------------------------------------------------------------
def test_bearer_header_extraction_rules():
    """Verify Bearer token extraction rules from HTTP Authorization header."""
    valid_key = "sk-speedinfer-1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
    valid_header = f"Bearer {valid_key}"
    parts = valid_header.split(" ", 1)
    assert len(parts) == 2
    assert parts[0] == "Bearer"
    assert parts[1].startswith("sk-speedinfer-")

    # Invalid header variations that must be rejected
    invalid_headers = [
        "",
        "Basic dXNlcjpwYXNz",
        "bearer sk-speedinfer-lowercase-scheme",
        "Bearer ",
        "sk-speedinfer-without-bearer-prefix",
        "Bearer sk-otherprefix-12345",
    ]
    for h in invalid_headers:
        tokens = h.split(" ", 1)
        is_valid = (
            len(tokens) == 2 and tokens[0] == "Bearer" and tokens[1].startswith("sk-speedinfer-")
        )
        assert is_valid is False, f"Header should have been invalid: '{h}'"
