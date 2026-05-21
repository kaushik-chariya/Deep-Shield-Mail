"""
Deep Shield Mail — Flask App
Supports: Gmail OAuth2 · Manual Paste
"""

import os, sys, traceback
from functools import wraps
from pathlib import Path

from flask import (
    Flask, render_template, redirect, url_for,
    session, request, jsonify, flash
)

# ── Path setup ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ── Prediction pipeline ─────────────────────────────────────────
from src.pipeline.prediction_pipeline import PredictionPipeline

# ── Google OAuth ────────────────────────────────────────────────
try:
    from google_auth_oauthlib.flow import Flow
    from googleapiclient.discovery import build
    import google.oauth2.credentials
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════
# App setup
# ═══════════════════════════════════════════════════════════════

app = Flask(
    __name__,
    template_folder=str(ROOT / "templates"),
    static_folder=str(ROOT / "static"),
)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-in-prod")

# Allow HTTP for local dev (remove in production)
os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "1")
os.environ.setdefault("OAUTHLIB_RELAX_TOKEN_SCOPE", "1")

# ── Lazy-load prediction pipeline ───────────────────────────────
_pipeline: PredictionPipeline | None = None

def get_pipeline() -> PredictionPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = PredictionPipeline()
    return _pipeline


# ═══════════════════════════════════════════════════════════════
# OAuth Config  (set via env vars or .env)
# ═══════════════════════════════════════════════════════════════

GOOGLE_CLIENT_ID     = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GOOGLE_SCOPES        = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "openid", "email", "profile",
]


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "provider" not in session:
            flash("Please connect your Gmail account first.", "warning")
            return redirect(url_for("index"))
        return f(*args, **kwargs)
    return decorated


# ═══════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════

# ── Health check (used by CI / load-balancers) ──────────────────
@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200


# ── Landing ─────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template(
        "index.html",
        google_enabled=bool(GOOGLE_CLIENT_ID) and GOOGLE_AVAILABLE,
    )


# ═══════════════════════════════════════════════════════════════
# Gmail OAuth2
# ═══════════════════════════════════════════════════════════════

@app.route("/auth/gmail")
def auth_gmail():
    if not GOOGLE_AVAILABLE:
        flash("google-auth-oauthlib not installed. Run: pip install google-auth-oauthlib google-api-python-client", "danger")
        return redirect(url_for("index"))
    if not GOOGLE_CLIENT_ID:
        flash("GOOGLE_CLIENT_ID not set in environment.", "danger")
        return redirect(url_for("index"))

    client_config = {
        "web": {
            "client_id"    : GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "auth_uri"     : "https://accounts.google.com/o/oauth2/auth",
            "token_uri"    : "https://oauth2.googleapis.com/token",
            "redirect_uris": [url_for("auth_gmail_callback", _external=True)],
        }
    }
    flow = Flow.from_client_config(client_config, scopes=GOOGLE_SCOPES)
    flow.redirect_uri = url_for("auth_gmail_callback", _external=True)

    auth_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        code_challenge_method="S256",
    )
    session["google_state"]         = state
    session["google_code_verifier"] = flow.code_verifier
    return redirect(auth_url)


@app.route("/auth/gmail/callback")
def auth_gmail_callback():
    state = session.get("google_state")
    client_config = {
        "web": {
            "client_id"    : GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "auth_uri"     : "https://accounts.google.com/o/oauth2/auth",
            "token_uri"    : "https://oauth2.googleapis.com/token",
            "redirect_uris": [url_for("auth_gmail_callback", _external=True)],
        }
    }
    flow = Flow.from_client_config(client_config, scopes=GOOGLE_SCOPES, state=state)
    flow.redirect_uri = url_for("auth_gmail_callback", _external=True)

    flow.fetch_token(
        authorization_response=request.url,
        code_verifier=session.get("google_code_verifier"),
    )

    creds = flow.credentials
    session["provider"]     = "gmail"
    session["google_token"] = {
        "token"        : creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri"    : creds.token_uri,
        "client_id"    : creds.client_id,
        "client_secret": creds.client_secret,
        "scopes"       : creds.scopes,
    }

    service = build("gmail", "v1", credentials=creds)
    profile = service.users().getProfile(userId="me").execute()
    session["user_email"] = profile.get("emailAddress", "")
    flash(f"Connected as {session['user_email']}", "success")
    return redirect(url_for("inbox"))


# ═══════════════════════════════════════════════════════════════
# Manual Email Paste
# ═══════════════════════════════════════════════════════════════

@app.route("/manual")
def manual_email():
    return render_template("manual_email.html")


@app.route("/api/predict/manual", methods=["POST"])
def api_predict_manual():
    """Predict spam on a raw email pasted by the user — no login required."""
    data     = request.get_json(force=True)
    raw_text = data.get("raw", "").strip()
    if not raw_text:
        return jsonify({"error": "No email text provided"}), 400
    try:
        result = get_pipeline().predict(raw_text)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════
# Inbox  (Gmail only)
# ═══════════════════════════════════════════════════════════════

@app.route("/inbox")
@login_required
def inbox():
    return render_template(
        "dashboard.html",
        user_email=session.get("user_email"),
        provider=session.get("provider"),
    )


@app.route("/api/emails")
@login_required
def api_emails():
    page  = int(request.args.get("page",  1))
    limit = int(request.args.get("limit", 20))
    try:
        emails = _fetch_gmail(page, limit)
        return jsonify({"emails": emails, "page": page})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
@login_required
def api_predict():
    data     = request.get_json(force=True)
    raw_text = data.get("raw", "")
    if not raw_text:
        return jsonify({"error": "No email text provided"}), 400
    try:
        result = get_pipeline().predict(raw_text)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/scan", methods=["POST"])
@login_required
def api_scan():
    """Predict all emails and return aggregated results."""
    data   = request.get_json(force=True)
    emails = data.get("emails", [])
    if not emails:
        return jsonify({"error": "No emails provided"}), 400

    results = []
    pipe    = get_pipeline()
    for em in emails:
        try:
            pred = pipe.predict(em.get("raw", ""))
            results.append({**em, **pred})
        except Exception as e:
            results.append({**em, "label": "Error", "prediction": -1,
                            "probability": 0.0, "error": str(e)})

    spam_count = sum(1 for r in results if r.get("prediction") == 1)
    return jsonify({
        "results"   : results,
        "total"     : len(results),
        "spam_count": spam_count,
        "ham_count" : len(results) - spam_count,
    })


# ═══════════════════════════════════════════════════════════════
# Gmail Fetcher
# ═══════════════════════════════════════════════════════════════

def _fetch_gmail(page: int, limit: int) -> list[dict]:
    token_data = session["google_token"]
    creds      = google.oauth2.credentials.Credentials(**token_data)
    service    = build("gmail", "v1", credentials=creds)

    results = service.users().messages().list(
        userId="me", labelIds=["INBOX"],
        maxResults=limit,
        pageToken=session.get("gmail_page_token") if page > 1 else None,
    ).execute()

    if page == 1:
        session.pop("gmail_page_token", None)
    next_token = results.get("nextPageToken")
    if next_token:
        session["gmail_page_token"] = next_token

    messages = results.get("messages", [])
    emails   = []
    for m in messages:
        msg_data = service.users().messages().get(
            userId="me", id=m["id"], format="full"
        ).execute()
        headers = {h["name"]: h["value"] for h in msg_data["payload"].get("headers", [])}
        snippet = msg_data.get("snippet", "")

        raw = (
            f"From: {headers.get('From','')}\n"
            f"To: {headers.get('To','')}\n"
            f"Subject: {headers.get('Subject','')}\n"
            f"Date: {headers.get('Date','')}\n\n"
            f"{snippet}"
        )
        emails.append({
            "id"     : m["id"],
            "from"   : headers.get("From", ""),
            "subject": headers.get("Subject", "(no subject)"),
            "date"   : headers.get("Date", ""),
            "snippet": snippet,
            "raw"    : raw,
        })
    return emails


# ═══════════════════════════════════════════════════════════════
# Logout
# ═══════════════════════════════════════════════════════════════

@app.route("/logout")
def logout():
    session.clear()
    flash("Disconnected successfully.", "info")
    return redirect(url_for("index"))


# ═════════════════════════════════════════════════════════════════
# Run
# ═════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from constants import APP_HOST, APP_PORT
    app.run(
        host=APP_HOST,
        port=APP_PORT,
        debug=os.getenv("FLASK_DEBUG", "0") == "1",  # CI में FLASK_DEBUG set नहीं होगा → False
    )