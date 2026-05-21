"""
flask_app/preprocessing_utility.py
Lightweight email preprocessing helpers used by the Flask serving layer.
These are intentionally separate from the training-time transformers so the
Flask app has zero dependency on sklearn/scipy at import time.
"""

import re
import html
from email import policy as email_policy
from email.parser import BytesParser, Parser
from typing import Optional


# ═══════════════════════════════════════════════════════════════
# Text Cleaning
# ═══════════════════════════════════════════════════════════════

def strip_html(text: str) -> str:
    """Remove HTML tags and decode HTML entities."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return text


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces/newlines into single space."""
    return re.sub(r"\s+", " ", text).strip()


def clean_email_body(text: str) -> str:
    """Full cleaning pipeline for email body text."""
    text = strip_html(text)
    text = normalize_whitespace(text)
    return text.lower()


# ═══════════════════════════════════════════════════════════════
# Email Parsing
# ═══════════════════════════════════════════════════════════════

def parse_raw_email(raw: str) -> dict:
    """
    Parse a raw email string into a structured dict.

    Returns
    -------
    dict with keys: from, to, subject, date, body, raw
    """
    try:
        msg = Parser(policy=email_policy.default).parsestr(raw)
    except Exception:
        return {
            "from"   : "",
            "to"     : "",
            "subject": "",
            "date"   : "",
            "body"   : raw,
            "raw"    : raw,
        }

    return {
        "from"   : str(msg.get("From", "")),
        "to"     : str(msg.get("To", "")),
        "subject": str(msg.get("Subject", "")),
        "date"   : str(msg.get("Date", "")),
        "body"   : _extract_body(msg),
        "raw"    : raw,
    }


def _extract_body(msg) -> str:
    """Extract plain-text body from parsed email.message.Message."""
    parts = []
    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            cd = str(part.get("Content-Disposition", ""))
            if ct == "text/plain" and "attachment" not in cd:
                try:
                    parts.append(part.get_content())
                except Exception:
                    try:
                        payload = part.get_payload(decode=True)
                        charset = part.get_content_charset() or "utf-8"
                        parts.append(payload.decode(charset, errors="replace"))
                    except Exception:
                        pass
    else:
        try:
            parts.append(msg.get_content())
        except Exception:
            try:
                payload = msg.get_payload(decode=True)
                if payload:
                    charset = msg.get_content_charset() or "utf-8"
                    parts.append(payload.decode(charset, errors="replace"))
            except Exception:
                pass
    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════
# Quick Feature Extraction  (for display / UI only — NOT for model)
# ═══════════════════════════════════════════════════════════════

def quick_features(body: str) -> dict:
    """
    Compute display-level features for UI badges.
    Does NOT feed into the model — the PredictionPipeline handles that.
    """
    text  = body or ""
    words = text.split()
    urls  = re.findall(r"https?://\S+|www\.\S+", text)

    return {
        "word_count"      : len(words),
        "url_count"       : len(urls),
        "has_html"        : bool(re.search(r"<[a-z][\s\S]*>", text, re.I)),
        "dollar_count"    : text.count("$"),
        "exclamation_count": text.count("!"),
        "caps_ratio"      : (
            sum(1 for c in text if c.isupper()) / max(len(text), 1)
        ),
    }


# ═══════════════════════════════════════════════════════════════
# Truncation helper
# ═══════════════════════════════════════════════════════════════

def truncate(text: str, max_chars: int = 300) -> str:
    """Truncate text to max_chars with ellipsis."""
    return text[:max_chars] + "…" if len(text) > max_chars else text