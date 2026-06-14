from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from urllib.parse import urlparse, parse_qs, urlencode
from pydantic import BaseModel, Field
import joblib
import json
import pandas as pd
import re
import math
import ipaddress
import socket
import threading
from collections import defaultdict, Counter
from datetime import datetime, timedelta

_rate_data = defaultdict(list)
_rate_lock = threading.Lock()

def is_rate_limited(ip: str, limit: int = 5, window_seconds: int = 60) -> bool:
    now = datetime.utcnow()
    cutoff = now - timedelta(seconds=window_seconds)
    with _rate_lock:
        _rate_data[ip] = [t for t in _rate_data[ip] if t > cutoff]
        if len(_rate_data[ip]) >= limit:
            return True
        _rate_data[ip].append(now)
        return False

app = FastAPI(title="OPay LinkCheck API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    model = joblib.load("model/opay_linkcheck.pkl")
    with open("model/feature_columns.json") as f:
        feature_columns = json.load(f)
    MODEL_LOADED = True
except Exception as e:
    print(f"Warning: Could not load model — {e}")
    model = None
    # Base features (Guardian AI) + OPay-specific features.
    # Update feature_columns.json after retraining with the new columns below.
    feature_columns = [
        "url_length", "num_dots", "num_hyphens", "num_slash", "num_special_chars",
        "has_ip", "has_at_symbol", "has_double_slash", "is_https", "subdomain_depth",
        "path_length", "num_subdomains", "has_suspicious_word", "entropy", "domain_length",
        # OPay-specific
        "opay_in_domain", "opay_homoglyph", "levenshtein_to_opay",
        "bait_keyword_count", "has_bad_tld", "has_subdomain_brand", "path_depth",
    ]
    MODEL_LOADED = False


OPAY_WHITELIST = {
    "opay.com",
    "opay.com.ng",
    "okash.com.ng",
}

COMPOUND_TLDS = {
    ".com.ng", ".gov.ng", ".edu.ng", ".org.ng", ".net.ng",
    ".co.uk", ".org.uk",
}

BAD_TLDS = {
    ".xyz", ".tk", ".click", ".live", ".ml", ".top",
    ".site", ".online", ".gq", ".cf", ".ga", ".pw", ".buzz",
}

OPAY_BAIT_KEYWORDS = {
    "loan", "apply", "promo", "bonus", "giveaway", "reward",
    "invest", "earn", "verify", "confirm", "register", "claim",
    "win", "free", "instant", "quick", "approve", "eligible",
    "double", "profit", "onboard", "signup", "access", "auth", "cash",
}

SCAM_TYPE_KEYWORDS = {
    "pos":    ["pos", "agent", "distributor", "terminal", "onboard"],
    "loan":   ["loan", "cash", "borrow", "okash", "lend", "credit", "quick"],
    "invest": ["invest", "investment", "wealth", "earn", "profit", "double", "plan"],
    "cred":   ["verify", "login", "signin", "auth", "confirm", "update",
               "secure", "account", "access"],
    "bonus":  ["bonus", "giveaway", "reward", "promo", "win", "free",
               "airtime", "claim"],
}

class URLRequest(BaseModel):
    url: str = Field(..., max_length=2048)


def normalize_url(url: str) -> str:
    
    url = url.strip()

    url = url.replace("hxxps://", "https://").replace("hxxp://", "http://")

    url = url.replace("[.]", ".")

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    return url


def get_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    probs = [c / len(text) for c in counts.values()]
    return -sum(p * math.log2(p) for p in probs)


def get_root_domain(url: str) -> str:
    
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if ":" in domain:
            domain = domain.split(":")[0]
        if domain.startswith("www."):
            domain = domain[4:]
        parts = domain.split(".")
        if len(parts) >= 3:
            suffix = "." + ".".join(parts[-2:])
            if suffix in COMPOUND_TLDS:
                return ".".join(parts[-3:])
        if len(parts) >= 2:
            return ".".join(parts[-2:])
        return domain
    except Exception:
        return ""


def is_safe_url(url: str) -> bool:
    """Block SSRF — rejects localhost, private IPs, non-HTTP schemes."""
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        hostname = parsed.hostname
        if not hostname:
            return False
        if hostname in ("localhost", "::1"):
            return False
        try:
            ip = ipaddress.ip_address(socket.gethostbyname(hostname))
            if ip.is_private or ip.is_loopback or ip.is_reserved or ip.is_link_local:
                return False
        except (socket.gaierror, ValueError):
            pass
        return True
    except Exception:
        return False


def strip_tracking_params(url: str) -> str:
    try:
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        clean = {
            k: v for k, v in params.items()
            if not k.startswith(("utm_", "gclid", "fbclid", "campaignid",
                                  "adgroupid", "gad_"))
        }
        return parsed._replace(query=urlencode(clean, doseq=True)).geturl()
    except Exception:
        return url

# ---------------------------------------------------------------------------
# Levenshtein distance — catches typosquats like opayy, opey (1–2 edits)
# ---------------------------------------------------------------------------
def levenshtein(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if not s2:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for c1 in s1:
        curr = [prev[0] + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


def extract_base_features(url: str) -> dict:
    """15 base features — identical to Guardian AI's extract_features()."""
    parsed = urlparse(url)
    domain = parsed.netloc
    path = parsed.path
    if not domain and path:
        parts = path.split("/")
        domain = parts[0]
        path = "/" + "/".join(parts[1:]) if len(parts) > 1 else ""
    return {
        "url_length":          len(url),
        "num_dots":            url.count("."),
        "num_hyphens":         url.count("-"),
        "num_slash":           url.count("/"),
        "num_special_chars":   len(re.findall(r"[@_!#$%^&*()<>?/|}{~:]", url)),
        "has_ip":              1 if re.match(r"https?://\d+\.\d+\.\d+\.\d+", url) else 0,
        "has_at_symbol":       1 if "@" in url else 0,
        "has_double_slash":    1 if "//" in path else 0,
        "is_https":            1 if parsed.scheme == "https" else 0,
        "subdomain_depth":     len(domain.split(".")) - 2 if domain else 0,
        "path_length":         len(path),
        "num_subdomains":      domain.count(".") if domain else 0,
        "has_suspicious_word": 1 if re.search(
            r"login|verify|secure|account|update|banking|confirm|password|signin|webscr",
            url, re.IGNORECASE,
        ) else 0,
        "entropy":             get_entropy(url),
        "domain_length":       len(domain),
    }


def extract_opay_features(url: str, domain: str, root_domain: str) -> dict:
    """7 OPay-specific features added for this project."""
    url_lower    = url.lower()
    domain_lower = domain.lower()
    if ":" in domain_lower:             # strip port
        domain_lower = domain_lower.split(":")[0]

    d = domain_lower[4:] if domain_lower.startswith("www.") else domain_lower
    sld = d.split(".")[0] if d else ""

    opay_in_domain    = int("opay" in domain_lower or "okash" in domain_lower)
    opay_homoglyph    = int("0pay" in url_lower)          # zero substitution for O
    lev_distance      = levenshtein(sld, "opay")
    bait_count        = sum(1 for kw in OPAY_BAIT_KEYWORDS if kw in url_lower)

    has_bad_tld = int(any(domain_lower.endswith(tld) for tld in BAD_TLDS))

    netloc_parts = domain_lower.split(".")
    has_subdomain_brand = int(
        len(netloc_parts) > 2 and
        ("opay" in netloc_parts[0] or "okash" in netloc_parts[0])
    )

    path_depth = len([p for p in urlparse(url).path.split("/") if p])

    return {
        "opay_in_domain":      opay_in_domain,
        "opay_homoglyph":      opay_homoglyph,
        "levenshtein_to_opay": lev_distance,
        "bait_keyword_count":  bait_count,
        "has_bad_tld":         has_bad_tld,
        "has_subdomain_brand": has_subdomain_brand,
        "path_depth":          path_depth,
    }


def run_layer_a(url: str, opay_features: dict) -> tuple:
    """
    Rule-based phishing check that runs before the ML model.
    Returns (is_phishing: bool, flags: list[str]).

    High-signal flags (opay_homoglyph, raw_ip_host, opay_in_subdomain) allow
    the caller to short-circuit Layer B entirely.
    """
    flags = []
    f = opay_features

    if f["opay_in_domain"]:
        flags.append("opay_in_domain")
    if f["opay_homoglyph"]:
        flags.append("opay_homoglyph")
    lev = f["levenshtein_to_opay"]
    if 0 < lev <= 2:
        flags.append(f"typosquat_lev:{lev}")
    if not url.startswith("https://"):
        flags.append("no_https")
    if re.match(r"https?://\d+\.\d+\.\d+\.\d+", url):
        flags.append("raw_ip_host")
    if f["has_bad_tld"]:
        flags.append("bad_tld")
    if f["bait_keyword_count"] > 0:
        flags.append(f"bait_keyword_count:{f['bait_keyword_count']}")
    if f["has_subdomain_brand"]:
        flags.append("opay_in_subdomain")

    HIGH_SIGNAL = {"opay_homoglyph", "raw_ip_host", "opay_in_subdomain"}
    triggered = (
        bool(HIGH_SIGNAL.intersection(flags)) or
        (f["opay_in_domain"] and len(flags) >= 2) or
        (0 < lev <= 2 and f["has_bad_tld"])
    )
    return triggered, flags


def classify_scam_type(url: str) -> str:
    """
    Secondary classifier — runs only after a phishing verdict.
    Returns one of: pos | loan | invest | cred | bonus | unknown
    """
    url_lower = url.lower()
    scores = {
        scam: sum(1 for kw in kws if kw in url_lower)
        for scam, kws in SCAM_TYPE_KEYWORDS.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "unknown"


def demo_score(base: dict, opay: dict, layer_a_phishing: bool) -> float:
    score = 0
    if base["has_ip"]:              score += 30
    if base["has_at_symbol"]:       score += 25
    if base["has_suspicious_word"]: score += 20
    if base["num_dots"] > 3:        score += 10
    if base["subdomain_depth"] > 1: score += 10
    if base["entropy"] > 4.5:       score += 5
    if opay["opay_in_domain"]:      score += 25
    if opay["has_bad_tld"]:         score += 20
    if opay["opay_homoglyph"]:      score += 35
    if opay["has_subdomain_brand"]: score += 30
    score += opay["bait_keyword_count"] * 10
    if layer_a_phishing:            score += 20
    return min(score / 150, 0.99)


@app.get("/")
def root():
    return {
        "message": "OPay LinkCheck API is running",
        "model_loaded": MODEL_LOADED,
        "version": "1.0.0",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": MODEL_LOADED,
        "features_count": len(feature_columns),
    }


@app.post("/check")
def check_url(request: Request, url_request: URLRequest):
    client_ip = request.client.host if request.client else "unknown"

    if is_rate_limited(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded: 5 requests per minute",
        )

    url = normalize_url(url_request.url)        # defang, prepend scheme
    if not is_safe_url(url):
        raise HTTPException(status_code=400, detail="URL not allowed")
    url = strip_tracking_params(url)            # strip UTM/fbclid params

    root_domain = get_root_domain(url)
    domain      = urlparse(url).netloc.lower()

    if root_domain in OPAY_WHITELIST:
        return {
            "url":               url,
            "is_phishing":       False,
            "confidence":        1.0,
            "scam_type":         None,
            "flags":             [],
            "whitelisted":       True,
            "layer_a_triggered": False,
        }

    base_features = extract_base_features(url)
    opay_features = extract_opay_features(url, domain, root_domain)
    all_features  = {**base_features, **opay_features}

    layer_a_phishing, flags = run_layer_a(url, opay_features)

    HIGH_SIGNAL_FLAGS = {"opay_homoglyph", "raw_ip_host", "opay_in_subdomain"}
    if HIGH_SIGNAL_FLAGS.intersection(flags):
        return {
            "url":               url,
            "is_phishing":       layer_a_phishing,
            "confidence":        0.97,
            "scam_type":         classify_scam_type(url) if layer_a_phishing else None,
            "flags":             flags,
            "whitelisted":       False,
            "layer_a_triggered": True,
        }

    if not MODEL_LOADED:
        prob        = demo_score(base_features, opay_features, layer_a_phishing)
        is_phishing = prob > 0.5
        return {
            "url":               url,
            "is_phishing":       is_phishing,
            "confidence":        round(max(prob, 1 - prob), 4),
            "scam_type":         classify_scam_type(url) if is_phishing else None,
            "flags":             flags,
            "whitelisted":       False,
            "layer_a_triggered": layer_a_phishing,
            "mode":              "demo",
        }

    try:
        feature_values = pd.DataFrame(
            [[all_features.get(col, 0) for col in feature_columns]],
            columns=feature_columns
        )
        prediction     = model.predict(feature_values)[0]
        probability    = model.predict_proba(feature_values)[0]
        is_phishing    = bool(prediction == 1)
        confidence     = round(float(probability[1]), 4)

        if layer_a_phishing and not is_phishing and opay_features["opay_in_domain"]:
            is_phishing = True
            confidence  = max(confidence, 0.75)
            flags.append("layer_a_override")

        return {
            "url":               url,
            "is_phishing":       is_phishing,
            "confidence":        confidence,
            "scam_type":         classify_scam_type(url) if is_phishing else None,
            "flags":             flags,
            "whitelisted":       False,
            "layer_a_triggered": layer_a_phishing,
            "mode":              "production",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
