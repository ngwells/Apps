import datetime, io, json, math, os, re, base64, random, secrets, hashlib
from flask import Flask, jsonify, request, render_template_string, session, redirect, url_for, send_file
from flask_caching import Cache
from mistralai.client import Mistral
import numpy as np
import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud, STOPWORDS
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
import plotly.graph_objects as go
import plotly.express as pxEvalbrin
import gc

# Google Gmail API imports
from email.mime.text import MIMEText
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "soccer-grade-secret-automation-key")

# Cache & DB Configuration
app.config["CACHE_TYPE"] = "FileSystemCache"
app.config["CACHE_DIR"] = os.path.join(app.instance_path, "flask_cache")
app.config["CACHE_DEFAULT_TIMEOUT"] = 3600
cache = Cache(app)
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE='Lax')

db_url = os.environ.get("DATABASE_URL")
if db_url:
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
    
    if "sslmode" not in db_url:
        separator = "&" if "?" in db_url else "?"
        db_url = f"{db_url}{separator}sslmode=require"

app.config['SQLALCHEMY_DATABASE_URI'] = db_url
db = SQLAlchemy(app)

API_KEY = os.environ.get("MISTRAL_API_KEY")
client = Mistral(api_key=API_KEY) if API_KEY else None

API_KEY2 = os.environ.get("GEMINI_API_KEY")
client2 = genai.Client(api_key=API_KEY2) if API_KEY2 else None


# --- User Model ---
class User(UserMixin, db.Model):
    __tablename__ = 'user'
    id = db.Column(db.String, primary_key=True, server_default=db.text("gen_random_uuid()"))
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(256), nullable=False)
    agreement = db.Column(db.Boolean, default=False, nullable=False)
    paid = db.Column(db.Boolean, default=False, nullable=False)
    verified = db.Column(db.Boolean, default=False, nullable=False)
    times_logged_in = db.Column(db.Integer, default=0, nullable=False)
    last_log_in = db.Column(db.DateTime(timezone=True))
    created_at = db.Column(db.DateTime(timezone=True), server_default=db.func.now())

# --- OTP Helper Store Model ---
class OTPCode(db.Model):
    __tablename__ = 'otp_codes'
    email = db.Column(db.String(120), primary_key=True)
    code = db.Column(db.String(6), nullable=False)
    expires_at = db.Column(db.DateTime(timezone=True), nullable=False)

# --- Remember Token Model ---
class RememberToken(db.Model):
    __tablename__ = 'remember_tokens'
    id = db.Column(db.String, primary_key=True, server_default=db.text("gen_random_uuid()"))
    user_id = db.Column(db.String, db.ForeignKey('user.id', ondelete='CASCADE'), nullable=False)
    token_hash = db.Column(db.String(256), unique=True, nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), server_default=db.func.now())
    expires_at = db.Column(db.DateTime(timezone=True), nullable=False)

with app.app_context():
    db.create_all()


# --- Device Token Helpers ---
def is_device_remembered(user_id):
    """Checks if incoming request has a valid, unexpired remember_device cookie."""
    cookie_token = request.cookies.get("remember_device")
    if not cookie_token:
        return False

    hashed_cookie = hashlib.sha256(cookie_token.encode('utf-8')).hexdigest()
    token_entry = RememberToken.query.filter_by(token_hash=hashed_cookie).first()

    if not token_entry or str(token_entry.user_id) != str(user_id):
        return False

    now = datetime.datetime.now(datetime.timezone.utc)
    expires_at = token_entry.expires_at if token_entry.expires_at.tzinfo else token_entry.expires_at.replace(tzinfo=datetime.timezone.utc)

    if expires_at.timestamp() < now.timestamp():
        db.session.delete(token_entry)
        db.session.commit()
        return False

    return True



def create_remember_cookie(response, user_id):
    """Generates a secure token, saves its hash to DB, and attaches HTTP-only cookie."""
    raw_token = secrets.token_hex(32)
    token_hash = hashlib.sha256(raw_token.encode('utf-8')).hexdigest()
    expires_at = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=30)

    new_token = RememberToken(
        user_id=str(user_id),
        token_hash=token_hash,
        expires_at=expires_at
    )
    db.session.add(new_token)
    db.session.commit()

    # Dynamically determine if the connection is HTTPS (Render) or HTTP (Localhost)
    is_secure = request.is_secure or request.headers.get("X-Forwarded-Proto") == "https"

    response.set_cookie(
        "remember_device",
        raw_token,
        max_age=30 * 24 * 60 * 60,
        httponly=True,
        samesite="Lax",
        secure=is_secure
    )
    return response


# --- Gmail API Email Dispatcher ---
GMAIL_SCOPES = ['https://www.googleapis.com/auth/gmail.send']

def get_gmail_service():
    """Retrieves or refreshes Gmail API credentials strictly from environment variables."""
    refresh_token = os.environ.get("GMAIL_REFRESH_TOKEN")
    client_id = os.environ.get("GMAIL_CLIENT_ID")
    client_secret = os.environ.get("GMAIL_CLIENT_SECRET")

    if not all([refresh_token, client_id, client_secret]):
        raise ValueError("Missing required Gmail OAuth environment variables.")

    creds = Credentials(
        token=None,
        refresh_token=refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=client_id,
        client_secret=client_secret,
        scopes=GMAIL_SCOPES
    )
    
    creds.refresh(Request())
    return build('gmail', 'v1', credentials=creds)

def send_gmail_otp(to_email, otp_code):
    """Sends 2FA OTP email using Gmail API from contact@evalbricks.com."""
    try:
        service = get_gmail_service()
        
        html_content = f"""
        <html>
            <body style="font-family: sans-serif; padding: 20px;">
                <h2>Security Verification</h2>
                <p>Your 2FA verification code is:</p>
                <h1 style="letter-spacing: 5px; color: #2563eb;">{otp_code}</h1>
                <p>This code will expire in 10 minutes.</p>
            </body>
        </html>
        """
        
        message = MIMEText(html_content, 'html')
        message['to'] = to_email
        message['from'] = 'contact@evalbricks.com'
        message['subject'] = 'Your 2FA Verification Code'
        
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
        service.users().messages().send(userId='me', body={'raw': raw_message}).execute()
    except Exception as e:
        print(f"Gmail API Error sending email: {e}")
        raise e

def send_2fa_otp(email):
    """Generates 6-digit OTP, updates database, and dispatches email via Gmail API."""
    otp_code = f"{random.randint(100000, 999999)}"
    expires_at = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=10)
    
    existing_otp = OTPCode.query.get(email.lower())
    if existing_otp:
        existing_otp.code = otp_code
        existing_otp.expires_at = expires_at
    else:
        new_otp = OTPCode(email=email.lower(), code=otp_code, expires_at=expires_at)
        db.session.add(new_otp)
    db.session.commit()

    send_gmail_otp(email, otp_code)


# --- Auth Routes ---
def is_valid_email(email):
    """Basic regex to check if the string looks like an email address."""
    return bool(re.match(r"[^@]+@[^@]+\.[^@]+", email))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password")
        
        if not is_valid_email(email):
            return "Please enter a valid email address. <a href='/login'>Try again</a>", 400
            
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            # Check if device token cookie is valid and skip 2FA
            if is_device_remembered(user.id):
                user.times_logged_in = (user.times_logged_in or 0) + 1
                user.last_log_in = datetime.datetime.now(datetime.timezone.utc)
                db.session.commit()

                session["user_id"] = str(user.id)
                return redirect(url_for("index"))

            # Otherwise proceed with 2FA dispatch
            session["pending_email"] = user.email
            try:
                send_2fa_otp(user.email)
            except Exception:
                return "Failed to send 2FA email. Please check server Gmail configuration.", 500
            return redirect(url_for("verify_2fa"))
            
        return "Invalid credentials. <a href='/login'>Try again</a>", 400
    return render_template_string(LOGIN_PAGE_HTML)

@app.route("/register", methods=["POST"])

def register():
    # Sanitize inputs by stripping out potential injection symbols: ^ { [ ( ) ] } < > ; ' " ` \
    sanitize_pattern = r"[^{\[\(\)\]\}<>;\'\"`\\]"
    
    raw_first = request.form.get("first_name", "").strip()
    raw_last = request.form.get("last_name", "").strip()
    raw_email = request.form.get("email", "").strip().lower()
    
    # Strip special characters
    first_name = "".join(re.findall(sanitize_pattern, raw_first))
    last_name = "".join(re.findall(sanitize_pattern, raw_last))
    email = "".join(re.findall(sanitize_pattern, raw_email))
    
    password = request.form.get("password", "")
    agreement = True if request.form.get("agreement") == "on" else False

    if not is_valid_email(email):
        return "Invalid email format. Please go back and try again.", 400

    if len(password) < 8:
        return "Password must be at least 8 characters long. <a href='/login'>Try again</a>", 400

    if not agreement:
        return "You must agree to the Terms of Service. <a href='/login'>Try again</a>", 400

    if User.query.filter_by(email=email).first():
        return "Email already registered. <a href='/login'>Go to login</a>", 400

    hashed_pw = generate_password_hash(password)
    now = datetime.datetime.now(datetime.timezone.utc)

    new_user = User(
        first_name=first_name,
        last_name=last_name,
        email=email,
        password=hashed_pw,
        agreement=True,
        paid=False,
        verified=False,
        times_logged_in=0,
        last_log_in=now
    )

    db.session.add(new_user)
    db.session.commit()

    session["pending_email"] = email

    try:
        send_2fa_otp(email)
    except Exception:
        return "Account created, but failed to dispatch 2FA email.", 500

    return redirect(url_for("verify_2fa"))

# --- KEEP ONLY THIS SINGLE INSTANCE IN YOUR FILE ---
@app.route("/verify-2fa", methods=["GET", "POST"])
def verify_2fa():
    pending_email = session.get("pending_email")
    if not pending_email:
        return redirect(url_for("login"))
        
    if request.method == "POST":
        entered_code = request.form.get("otp_code", "").strip()
        otp_record = OTPCode.query.get(pending_email)
        
        now = datetime.datetime.now(datetime.timezone.utc)
        if not otp_record or otp_record.code != entered_code:
            return "Invalid 2FA Code. <a href='/verify-2fa'>Try again</a>", 400
            
        if otp_record.expires_at.tzinfo is None:
            expires_at = otp_record.expires_at.replace(tzinfo=datetime.timezone.utc)
        else:
            expires_at = otp_record.expires_at
            
        if expires_at.timestamp() < now.timestamp():
            return "Expired 2FA Code. <a href='/login'>Request a new one</a>", 400
            
        user = User.query.filter_by(email=pending_email).first()
        if user:
            user.verified = True
            user.times_logged_in = (user.times_logged_in or 0) + 1
            user.last_log_in = now
            db.session.delete(otp_record)
            db.session.commit()
            
            session.pop("pending_email", None)
            session["user_id"] = str(user.id)
            
            # Create and attach remember_device cookie automatically
            response = redirect(url_for("index"))
            response = create_remember_cookie(response, user.id)

            return response
            
    return render_template_string(VERIFY_2FA_PAGE_HTML, email=pending_email)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/reset-password", methods=["POST"])
def reset_password():
    email = request.form.get("email", "").strip().lower()
    new_password = request.form.get("new_password")
    
    if not email or not new_password:
        return "Email and new password are required. <a href='/login'>Try again</a>", 400
        
    user = User.query.filter_by(email=email).first()
    
    if user:
        hashed_pw = generate_password_hash(new_password)
        user.password = hashed_pw
        db.session.commit()
        return "Password updated successfully! You can now <a href='/login'>log in</a>."
    else:
        return "Email not found. <a href='/login'>Try again</a>", 404


# --- Protected Routes ---
@app.route("/")
def index():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template_string(LANDING_PAGE_HTML, shared_nav=SHARED_NAV)

@app.route("/system/reset-session", methods=["POST"])
def reset_session():
    if "user_id" not in session: 
        return redirect(url_for("login"))
    session.clear(); cache.clear()
    return jsonify({"status": "cleared"})

# --- Mode & Local Session Management Routes ---

# --- Mode & Local Session Management Routes ---
@app.route("/start-mode/<mode_name>", methods=["GET"])
def start_mode(mode_name):
    if "user_id" not in session:
        return redirect(url_for("login"))

    valid_modes = {
        "tryout": "TryOut Mode",
        "practice": "Practice Mode",
        "postgame": "Post Game Mode",
    }
    if mode_name not in valid_modes:
        return redirect(url_for("index"))

    session["active_mode"] = valid_modes[mode_name]
    session["active_mode_slug"] = mode_name

    # Only clear cache when explicitly creating a brand-new session (?new=1)
    if request.args.get("new") == "1":
        cache.set("cached_raw", [])
        cache.set("cached_processed", [])

    if mode_name == "tryout":
        return redirect(url_for("tryout_mode"))
    elif mode_name == "practice":
        return redirect(url_for("practice_mode"))
    else:
        return redirect(url_for("postgame_mode"))


@app.route("/tryout-mode")
def tryout_mode():
    if "user_id" not in session:
        return redirect(url_for("login"))
    session["active_mode"] = "TryOut Mode"
    raw_session = cache.get("cached_raw") or []
    processed_session = cache.get("cached_processed") or []
    uploaded_session = cache.get("cached_uploaded") or []
    return render_template_string(
        TRYOUT_MODE_HTML,
        shared_nav=SHARED_NAV,
        session_top_bar=SESSION_TOP_BAR_HTML,
        active_mode=session["active_mode"],
        raw_count=len(raw_session),
        processed_count=len(processed_session),
        uploaded_count=len(uploaded_session),
    )


@app.route("/practice-mode", methods=["GET"])
def practice_mode():
    if "user_id" not in session:
        return redirect(url_for("login"))

    session["active_mode"] = "Practice Mode"
    session["active_mode_slug"] = "practice"
    return render_template_string(
        PRACTICE_MODE_HTML,
        shared_nav=SHARED_NAV,
        active_mode="Practice Mode",
    )


@app.route("/postgame-mode", methods=["GET"])
def postgame_mode():
    if "user_id" not in session:
        return redirect(url_for("login"))

    session["active_mode"] = "Post Game Mode"
    session["active_mode_slug"] = "postgame"
    return render_template_string(
        POSTGAME_MODE_HTML,
        shared_nav=SHARED_NAV,
        active_mode="Post Game Mode",
    )


@app.route("/export-session", methods=["GET"])
def export_session():
    if "user_id" not in session:
        return redirect(url_for("login"))

    active_mode = session.get("active_mode", "TryOut Mode")
    session_bundle = {
        "app": "evalbricks",
        "version": "1.0",
        "mode": active_mode,
        "saved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "cached_raw": cache.get("cached_raw") or [],
        "cached_processed": cache.get("cached_processed") or [],
    }

    buffer = io.BytesIO()
    buffer.write(json.dumps(session_bundle, indent=2).encode("utf-8"))
    buffer.seek(0)

    mode_slug = active_mode.lower().replace(" ", "_")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"evalbricks_{mode_slug}_{timestamp}.json"

    return send_file(
        buffer,
        as_attachment=True,
        download_name=filename,
        mimetype="application/json",
    )


@app.route("/import-session", methods=["POST"])
def import_session():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    if "session_file" not in request.files:
        return jsonify({"error": "No file selected"}), 400

    file = request.files["session_file"]
    expected_mode = request.form.get("expected_mode", "")

    try:
        bundle = json.load(file)
        if bundle.get("app") != "evalbricks":
            return jsonify({"error": "Invalid EvalBricks session file."}), 400

        loaded_mode = bundle.get("mode") or expected_mode or "TryOut Mode"
        session["active_mode"] = loaded_mode

        cache.set("cached_raw", bundle.get("cached_raw", []))
        cache.set("cached_processed", bundle.get("cached_processed", []))

        mode_route_map = {
            "TryOut Mode": url_for("tryout_mode"),
            "Practice Mode": url_for("practice_mode"),
            "Post Game Mode": url_for("postgame_mode"),
        }
        redirect_url = mode_route_map.get(loaded_mode, url_for("tryout_mode"))
        return jsonify({"status": "success", "redirect": redirect_url})
    except Exception as e:
        return jsonify({"error": f"Failed to parse session file: {str(e)}"}), 400

@app.route("/soccer-grade")
def soccer_grade():
    if "user_id" not in session:
        return redirect(url_for("login"))
    raw_records = cache.get("cached_raw") or []
    processed_records = cache.get("cached_processed") or []
    active_mode = session.get("active_mode", "TryOut Mode")
    return render_template_string(
        SOCCER_INTERFACE_HTML,
        shared_nav=SHARED_NAV,
        active_mode=active_mode,
        has_raw=len(raw_records) > 0,
        has_processed=len(processed_records) > 0,
        raw_records=raw_records,
        processed_records=processed_records,
        raw_records_json=json.dumps(raw_records),
        processed_records_json=json.dumps(processed_records),
    )


@app.route("/soccer-grade/sync-raw", methods=["POST"])
def sync_raw():
    if "user_id" not in session: return redirect(url_for("login"))
    req_data = request.get_json()
    if req_data and 'data' in req_data:
        cache.set('cached_raw', req_data['data'])
    return jsonify({"status": "synchronized"})

class TranscriptSegment(BaseModel):
    text: str = Field(description="The separate raw text segment from the transcript.")
    score: int = Field(description="Score from 100 to 0 rating how closely the text evaluates a single specific player's trait or characteristic. 100 = direct evaluation of one player. Lower scores (closer to 0) = general instructions, weird text, or event sequences involving multiple players.")

class TranscriptLog(BaseModel):
    segments: list[TranscriptSegment] = Field(description="List of segmented text chunks extracted from the transcript with their evaluation scores.")

@app.route("/soccer-grade/split-dataframe", methods=["POST"])
def split_dataframe():
    try:
        data = request.get_json()
        raw_rows = data.get("raw_rows", [])
        if not raw_rows:
            return jsonify({"status": "success", "processed_records": []})
        
        df_raw = pd.DataFrame(raw_rows)
        if len(df_raw.columns) >= 2:
            df_raw.columns = ['Timestamp', 'Transcript']
        else:
            return jsonify({"status": "error", "message": "Invalid data format received."}), 400
        
        # FIX: Join only the raw transcript text without prepending the timestamp string
        transcript_batch_str = "\n".join([str(row['Transcript']) for _, row in df_raw.iterrows()])
        
        prompt = f"""
        Analyze the following batch of transcript texts, break each down into natural segments/sentences, and for each segment:
        1. Extract the raw segment text cleanly without keeping any redundant timestamps or metadata.
           - the segment should have a subject (e.g., "Player 17", "Player 3", "Number 23")
        2. Assign a score from 100 to 0 based on how well it evaluates a specific individual player's trait/characteristic:
           - Score 51 to 100 by integers: Directly evaluates an individual player.
           - Lower scores / 0: General instructions, noise, or multi-player sequences.
           
        Return a JSON object with a single key "segments" containing a list of objects, each with "text" (string) and "score" (integer).
           
        Batch Transcripts:
        {transcript_batch_str}
        """
        
        model_options = ['gemini-3.5-flash-lite', 'gemini-3.1-flash-lite', 'gemini-2.5-flash']
        random.shuffle(model_options)
        
        response = None
        for chosen_model in model_options:
            try:
                print(f"Attempting model: {chosen_model}")
                response = client2.models.generate_content(
                    model=chosen_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.1,
                    ),
                )
                if response and getattr(response, 'text', None):
                    break
            except Exception as model_err:
                print(f"Model {chosen_model} failed/unavailable: {model_err}")
                continue
        
        rows = []
        parsed_segments = []
        
        if getattr(response, 'text', None):
            try:
                cleaned_text = response.text.strip()
                if cleaned_text.startswith("```json"):
                    cleaned_text = cleaned_text[7:-3].strip()
                elif cleaned_text.startswith("```"):
                    cleaned_text = cleaned_text[3:-3].strip()
                    
                data_dict = json.loads(cleaned_text)
                parsed_segments = data_dict.get("segments", [])
            except Exception as parse_err:
                print(f"JSON parsing error: {parse_err}")

        if parsed_segments:
            first_row = raw_rows[0] if raw_rows else {}
            if isinstance(first_row, dict):
                default_ts = first_row.get('Timestamp', '')
            elif isinstance(first_row, (list, tuple)) and len(first_row) > 0:
                default_ts = str(first_row[0])
            else:
                default_ts = str(first_row)
                
            for seg in parsed_segments:
                if isinstance(seg, dict):
                    seg_text = seg.get('text', '')
                    seg_score = int(seg.get('score', 0))
                else:
                    seg_text = getattr(seg, 'text', '')
                    seg_score = int(getattr(seg, 'score', 0))
                    
                rows.append({
                    "Timestamp": default_ts,
                    "Transcript": seg_text,
                    "Score": seg_score
                })
        
        if not rows:
            for _, row in df_raw.iterrows():
                rows.append({
                    "Timestamp": row['Timestamp'],
                    "Transcript": row['Transcript'],
                    "Score": 0
                })
                
        df_new = pd.DataFrame(rows)
        new_records = df_new.to_dict(orient="records")
        existing_processed = cache.get('cached_processed') or []
        updated_processed = existing_processed + new_records
        cache.set('cached_processed', updated_processed)
        
        gc.collect()
        return jsonify({"status": "success", "processed_records": new_records})
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error in split_dataframe batch route: {e}")
        gc.collect()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/soccer-grade/process-audio", methods=["POST"])
def process_audio():
    if "user_id" not in session: 
        return redirect(url_for("login"))
    if not client:
        return jsonify({"status": "error", "message": "Mistral API client context check failed."}), 500
    
    if 'audio_data' not in request.files:
        return jsonify({"status": "error", "message": "No audio data received"}), 400
    
    audio_file = request.files['audio_data']
    temp_filename = "temp_recording.webm"
    audio_file.save(temp_filename)
    
    try:
        with open(temp_filename, "rb") as f:
            transcription_response = client.audio.transcriptions.complete(
                model="voxtral-mini-latest",
                file={"content": f.read(), "file_name": temp_filename}
            )
        detected_text = transcription_response.text.strip()
    except Exception as e:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        return jsonify({"status": "error", "message": f"Transcription structural layer failure: {str(e)}"}), 500
    
    if os.path.exists(temp_filename):
        os.remove(temp_filename)
    
    if not detected_text:
        detected_text = "[Unintelligible audio recorded]"
    else:
        ones = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 
            'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 
            'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13, 
            'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 
            'eighteen': 18, 'nineteen': 19
        }
        tens = {
            'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50, 
            'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90
        }

        def replace_compound(match):
            t_word, o_word = match.groups()
            val = tens.get(t_word.lower(), 0) + ones.get(o_word.lower(), 0)
            return str(val)

        compound_pattern = r'\b(' + '|'.join(tens.keys()) + r')[\s-](' + '|'.join(ones.keys()) + r')\b'
        detected_text = re.sub(compound_pattern, replace_compound, detected_text, flags=re.IGNORECASE)

        for word, val in tens.items():
            detected_text = re.sub(r'\b' + word + r'\b', str(val), detected_text, flags=re.IGNORECASE)

        for word, val in ones.items():
            detected_text = re.sub(r'\b' + word + r'\b', str(val), detected_text, flags=re.IGNORECASE)
    
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return jsonify({"status": "success", "transcript": detected_text, "timestamp": current_time})
   

@app.route("/upload-manager")
def upload_manager():
    if "user_id" not in session: 
        return redirect(url_for("login"))
    raw_session = cache.get('cached_raw') or []
    processed_session = cache.get('cached_processed') or []
    uploaded_session = cache.get('cached_uploaded') or []
    was_overridden = cache.get('processed_was_overridden') or False
    
    raw_data_table = pd.DataFrame(raw_session).to_html(classes='table', index=False) if raw_session else None
    processed_data_table = pd.DataFrame(processed_session).to_html(classes='table', index=False) if processed_session else None
    uploaded_data_table = pd.DataFrame(uploaded_session).to_html(classes='table', index=False) if uploaded_session else None
    
    return render_template_string(
        UPLOAD_PAGE_HTML,
        raw_data_table=raw_data_table,
        processed_data_table=processed_data_table,
        uploaded_data_table=uploaded_data_table,
        was_overridden=was_overridden
    )

@app.route("/upload-manager/submit-file", methods=["POST"])
def submit_uploaded_file():
    if "user_id" not in session: return redirect(url_for("login"))
    if 'uploaded_csv' not in request.files:
        return "No file selected", 400
    file = request.files['uploaded_csv']
    if file.filename == '':
        return "Empty file selection", 400
    
    try:
        uploaded_df = pd.read_csv(file)
        cache.set('cached_uploaded', uploaded_df.to_dict(orient='records'))
        return render_template_string("<h3>Ingestion complete!</h3><p>File parsed successfully.</p><script>setTimeout(function(){window.location.href='/upload-manager';}, 1200);</script>")
    except Exception as e:
        return f"Error analyzing data structure: {str(e)}", 500

@app.route("/upload-manager/override-processed", methods=["POST"])
def override_processed_data():
    if "user_id" not in session: return redirect(url_for("login"))
    if 'mock_processed_csv' not in request.files:
        return "No file selected for testing override", 400
    file = request.files['mock_processed_csv']
    if file.filename == '':
        return "Empty file selection", 400
    
    try:
        mock_df = pd.read_csv(file)
        cache.set('cached_processed', mock_df.to_dict(orient='records'))
        cache.set('processed_was_overridden', True)
        return render_template_string("<h3>Testing Override Applied!</h3><p>Exploded dataset frame temporarily swapped.</p><script>setTimeout(function(){window.location.href='/upload-manager';}, 1200);</script>")
    except Exception as e:
        return f"Error applying template mock override: {str(e)}", 500

@app.route("/create-lineup")
def create_lineup():
    if "user_id" not in session: 
        return redirect(url_for("login"))
    raw_session = cache.get('cached_raw') or []
    processed_session = cache.get('cached_processed') or []
    uploaded_session = cache.get('cached_uploaded') or []
    selected_sport = cache.get('selected_sport') or ''
    selected_format = cache.get('selected_format') or ''
    blueprint_table = cache.get('cached_blueprint')
    
    return render_template_string(
        LINEUP_PAGE_HTML,
        raw_count=len(raw_session),
        processed_count=len(processed_session),
        uploaded_count=len(uploaded_session),
        selected_sport=selected_sport,
        selected_format=selected_format,
        blueprint_table=blueprint_table
    )

VALID_SPORT_FORMATS = {
    'soccer': {'7v7', '9v9', '11v11'},
    'basketball': {'5v5', '3v3'},
    'volleyball': {'6x6', '2v2'},
}

DEFAULT_SPORT_FORMATS = {
    'soccer': '7v7',
    'basketball': '5v5',
    'volleyball': '6x6',
}

@app.route("/create-lineup/select-sport", methods=["POST"])
def select_sport_sync():
    if "user_id" not in session:
        return redirect(url_for("login"))
    req_body = request.get_json(silent=True) or {}
    if 'sport_type' in req_body:
        sport_val = req_body['sport_type']
        cache.set('selected_sport', sport_val)
        sport_key = str(sport_val).strip().lower()
        current_fmt = str(cache.get('selected_format') or '').strip().lower()
        if sport_key in VALID_SPORT_FORMATS and current_fmt not in VALID_SPORT_FORMATS[sport_key]:
            cache.set('selected_format', DEFAULT_SPORT_FORMATS[sport_key])
        return jsonify({"status": "sport_cached"})
    return jsonify({"status": "no_sport_type"}), 400

@app.route("/create-lineup/select-format", methods=["POST"])
def select_format_sync():
    if "user_id" not in session:
        return redirect(url_for("login"))
    req_body = request.get_json()
    if req_body and 'format_type' in req_body:
        cache.set('selected_format', req_body['format_type'])
        return jsonify({"status": "format_cached"})

@app.route("/create-lineup/clear-blueprint", methods=["POST"])
def clear_blueprint():
    if "user_id" not in session: 
        return redirect(url_for("login"))
    cache.delete('cached_blueprint')
    return jsonify({"status": "success"})




import time

SOCCER_POSITION_NAMES = {
    'GK': 'Goalkeeper',
    'LDB': 'Left Defender Back',
    'CDB': 'Center Defender Back/Libero',
    'RDB': 'Right Defender Back',
    'LM': 'Left Midfielder',
    'CM': 'Center Midfielder/Attacking Midfielder/Sweeper',
    'RM': 'Right Midfielder',
    'LFW': 'Left Forward/Left Winger',
    'CF': 'Center Forward/Striker/false 9 ',
    'RFW': 'Right Forward/Right Winger'
}

VOLLEYBALL_POSITION_NAMES = {
    # 6v6 / 6x6 roles & aliases
    'S': 'Setter (Front & Back)',
    'SET': 'Setter (Front & Back)',
    'OH': 'Outside Hitter (Front & Back)',
    'MB': 'Middle Blocker (Front)',
    'OPP': 'Opposite / Right-Side Hitter (Front & Back)',
    'RS': 'Opposite / Right-Side Hitter (Front & Back)',
    'L': 'Libero (Back Only)',
    'LI': 'Libero (Back Only)',
    'DS': 'Defensive Specialist (Back)',
    # 2v2 side preferences
    'R': 'Right Side (Serve-Receive / Attacking / Setting)',
    'LS': 'Left Side (Serve-Receive / Attacking / Setting)'
}

BASKETBALL_POSITION_NAMES = {
    # 5v5 standard roles
    'PG': 'Point Guard (Primary Ball Handler / Playmaker / Perimeter)',
    'SG': 'Shooting Guard (Perimeter Scorer / Spacer / Secondary Handler)',
    'SF': 'Small Forward (Two-Way Wing / Slasher / Multi-Position Defender)',
    'PF': 'Power Forward (Stretch Forward / Weak-Side Rebounder / Forward-Center)',
    'C': 'Center (Rim Protector / Paint Anchor / Pick-and-Roll Finisher)',
    # 3v3 half-court roles
    'G': 'Guard (Perimeter Initiator / Spacer / Drive-and-Kick)',
    'W': 'Wing (Cut / Corner Shooter / Switch Defender)',
    'B': 'Big (Screen Setter / Paint Finisher / Rebounder)'
}

SPORT_POSITION_NAMES = {
    'soccer': SOCCER_POSITION_NAMES,
    'volleyball': VOLLEYBALL_POSITION_NAMES,
    'basketball': BASKETBALL_POSITION_NAMES
}


def _extract_positions_count(format_type: str) -> str:
    if not format_type:
        return "standard"
    fmt = str(format_type).strip().lower()
    for sep in ("v", "x"):
        if sep in fmt:
            prefix = fmt.split(sep, 1)[0].strip()
            if prefix.isdigit():
                return prefix
    return "standard"


def _build_board_position_context(sport_key: str, active_board_players: list) -> tuple[str, str]:
    pos_map = SPORT_POSITION_NAMES.get(sport_key)
    if not pos_map or not active_board_players:
        return "", ""

    raw_items = []
    for p in active_board_players:
        if isinstance(p, dict) and p.get('pos'):
            raw_items.append((str(p.get('pos')).strip(), p.get('zone')))

    if not raw_items:
        return "", ""

    raw_codes = [code for code, _ in raw_items]
    seen_counts = {}
    processed_full_names = []
    summary_items = []

    for code, zone in raw_items:
        base_name = pos_map.get(code, code)
        seen_counts[code] = seen_counts.get(code, 0) + 1

        if seen_counts[code] > 1 or raw_codes.count(code) > 1:
            labeled_name = f"{base_name} _{seen_counts[code]}"
        else:
            labeled_name = base_name

        processed_full_names.append(labeled_name)
        if sport_key in ('volleyball', 'basketball') and zone is not None:
            summary_items.append(f"{labeled_name} [Z{zone}]")
        else:
            summary_items.append(labeled_name)

    if sport_key == 'soccer':
        context_str = (
            f" Specifically, base the positions on this exact tactical board layout: "
            f"{', '.join(processed_full_names)}. For duplicate position roles (like _1 and _2), "
            f"detail their shared core traits as well as their distinct tactical responsibilities and "
            f"differing traits. Determine if 1 is more offensive and attacking while others are more defensive"
        )
    elif sport_key == 'volleyball':
        context_str = (
            f" Specifically, base the positions on this exact volleyball court layout: "
            f"{', '.join(summary_items)}. For duplicate position roles (like _1 and _2) or "
            f"multi-zone/side designations, detail their shared core traits as well as their "
            f"distinct rotational, front-row vs. back-row, attacking/blocking, serve-receive, "
            f"and defensive responsibilities."
        )
    elif sport_key == 'basketball':
        context_str = (
            f" Specifically, base the positions on this exact basketball court layout: "
            f"{', '.join(summary_items)}. For duplicate position roles (like _1 and _2) or "
            f"zone/spacing roles, detail their shared core traits as well as their distinct "
            f"perimeter vs. paint spacing, pick-and-roll/off-ball action, rebounding, transition, "
            f"and on/off-ball defensive responsibilities."
        )
    else:
        context_str = f" Specifically, base the positions on this layout: {', '.join(processed_full_names)}."

    return context_str, " → ".join(summary_items)


@app.route("/create-lineup/generate-tactics", methods=["POST"])
def generate_tactics():
    if "user_id" not in session:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401
    if not API_KEY:
        return jsonify({"status": "error", "message": "Mistral API key missing."}), 500
    
    req_body = request.get_json() or {}
    sport = req_body.get("sport") or cache.get('selected_sport') or "Soccer"
    format_type = req_body.get("format_type") or cache.get('selected_format') or "11v11"
    active_board_players = req_body.get("board_players", [])
    
    cache.set('selected_sport', sport)
    cache.set('selected_format', format_type)
    
    # --- Process Positions & Handle Duplicate Suffixes (_1, _2) ---
    sport_key = str(sport).strip().lower()
    position_context_str, board_summary_html = _build_board_position_context(
        sport_key, active_board_players
    )

    positions_count = _extract_positions_count(format_type)
    
    prompt_instruction = f"""Generate a JSON list of exactly {positions_count} positions for a {sport} team playing a {format_type} formation.{position_context_str} 
Each item in the list must be an object with two keys: "Position" and "Description". 
For the "Description" field, aggregate 5 different scout perspectives (Tactical Analyst, Elite Coach, Veteran Scout, Sports Scientist, and Data Modeler) into a single, cohesive block of text detailing sport and position characteristics, psychological traits, key technical skills, and where applicable, shared vs. differing traits for paired and multiple roles. Compliment the roles with offensive and defensive traits and skills.  
Output ONLY a valid JSON array. Do not use crazy and weird punctuation like @ and *"""
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "ministral-3b-2512",
        "messages": [
            {"role": "system", "content": "You are a lead sports data architect/general manager/coach combining multi-scout evaluations. Return ONLY a valid JSON array of objects with 'Position' and 'Description' keys. No markdown code blocks, no extra commentary. no weird characters like * or @"},
            {"role": "user", "content": prompt_instruction}
        ],
        "temperature": 0.2
    }
    
    api_response = None
    for attempt in range(3):
        try:
            api_response = requests.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=payload, timeout=60)
            if api_response.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            break
        except Exception as e:
            if attempt == 2:
                return jsonify({"status": "error", "message": f"Pipeline failure: {str(e)}"}), 500
            time.sleep(1 + attempt)
            
    if not api_response or api_response.status_code != 200:
        err_msg = api_response.text if api_response else "No response received"
        return jsonify({"status": "error", "message": f"Mistral API error: {err_msg}"}), 500
        
    try:
        data = api_response.json()
        raw_content = data['choices'][0]['message']['content'].strip()
        
        if "```" in raw_content:
            raw_content = re.sub(r"^```(?:json)?\s*", "", raw_content)
            raw_content = re.sub(r"\s*```$", "", raw_content)
            raw_content = raw_content.strip()
            
        # Clean common JSON syntax errors
        raw_content = re.sub(r',\s*([\]}])', r'\1', raw_content)
        
        try:
            parsed_json = json.loads(raw_content)
        except json.JSONDecodeError as jde:
            # Fallback: Extract JSON array/object structure and repair common missing colons/quotes
            match = re.search(r'(\[.*\]|\{.*\})', raw_content, re.DOTALL)
            if match:
                cleaned_match = match.group(1)
                cleaned_match = re.sub(r',\s*([\]}])', r'\1', cleaned_match)
                cleaned_match = re.sub(r'}\s*"', '},"', cleaned_match)
                cleaned_match = re.sub(r'}\s*{', '},{', cleaned_match)
                # Fix missing colons between property names and values if present
                cleaned_match = re.sub(r'"\s+"', '","', cleaned_match)
                try:
                    parsed_json = json.loads(cleaned_match)
                except Exception:
                    # Last resort fallback: Convert single quotes to double quotes
                    safe_match = re.sub(r'(?<![:,\s\[\{])"(?![,\]\}\s])', "'", cleaned_match)
                    parsed_json = json.loads(safe_match)
            else:
                raise jde
        
        if isinstance(parsed_json, dict):
            parsed_list = []
            for k, v in parsed_json.items():
                if isinstance(v, list):
                    parsed_list.extend(v)
                elif isinstance(v, dict):
                    parsed_list.append(v)
                else:
                    parsed_list.append({"Position": k, "Description": str(v)})
            if not parsed_list:
                parsed_list = [parsed_json]
        elif isinstance(parsed_json, list):
            parsed_list = parsed_json
        else:
            parsed_list = [{"Position": "Overview", "Description": str(parsed_json)}]
            
        df_output = pd.DataFrame(parsed_list)
        if len(df_output.columns) >= 2:
            df_output = df_output.iloc[:, :2]
            df_output.columns = ['Position', 'Narrative Description Summary']
            df_output['Narrative Description Summary'] = df_output['Narrative Description Summary'].apply(
                lambda x: " ".join(str(v) for v in x) if isinstance(x, (list, dict)) else str(x)
            )
            
        html_table = df_output.to_html(classes='table', index=False)
        cache.set('cached_blueprint', html_table)
        
        return jsonify({
            "status": "success", 
            "html_payload": html_table,
            "board_payload": board_summary_html
        })
    except Exception as e:
        print(f"generate_tactics parsing error: {str(e)}")
        return jsonify({"status": "error", "message": f"Pipeline failure: {str(e)} "}), 500



@app.route("/analytics")
def analytics():
    if "user_id" not in session: return redirect(url_for("login"))
    raw_session = cache.get('cached_raw') or []
    processed_session = cache.get('cached_processed') or []
    uploaded_session = cache.get('cached_uploaded') or []
    blueprint_table = cache.get('cached_blueprint')
    was_overridden = cache.get('processed_was_overridden') or False
    
    raw_table = pd.DataFrame(raw_session).to_html(classes='table', index=False) if raw_session else None
    processed_table = pd.DataFrame(processed_session).to_html(classes='table', index=False) if processed_session else None
    uploaded_table = pd.DataFrame(uploaded_session).to_html(classes='table', index=False) if uploaded_session else None
    similarity_results_table = cache.get('similarity_results_html')
    position_rankings_json = cache.get('position_rankings_json')
    
    barchart_data = cache.get('barchart_data')
    wordcloud_data = cache.get('wordcloud_data')
    barchart_data_json = json.dumps(barchart_data) if barchart_data else None
    sankey_json = cache.get('sankey_json')
    
    return render_template_string(
        ANALYTICS_PAGE_HTML,
        raw_table=raw_table,
        processed_table=processed_table,
        uploaded_table=uploaded_table,
        blueprint_table=blueprint_table,
        was_overridden=was_overridden,
        similarity_results_table=similarity_results_table,
        position_rankings_json=position_rankings_json,
        barchart_data_json=barchart_data_json,
        wordcloud_data=wordcloud_data,
        sankey_json=sankey_json
    )

@app.route("/analytics/clear-metrics", methods=["POST"])
def clear_metrics_dataframe():
    if "user_id" not in session: return redirect(url_for("login"))
    cache.delete('similarity_results_html')
    cache.delete('position_rankings_json')
    cache.delete('barchart_data')
    cache.delete('wordcloud_data')
    cache.delete('sankey_json')
    return render_template_string("<h3>Results Dataframe Flushed</h3><script>window.location.href='/analytics';</script>")

@app.route("/analytics/compute-metrics", methods=["POST"])
def compute_metrics():
    if "user_id" not in session: return redirect(url_for("login"))
    processed_session = cache.get('cached_processed') or []
    uploaded_session = cache.get('cached_uploaded') or []
    blueprint_html = cache.get('cached_blueprint')
    
    if not processed_session:
        return "Error: Processed Player Evaluations dataset frame missing.", 400
        
    if not uploaded_session and not blueprint_html:
        return "Error: Missing ideal target vectors. Load an external spreadsheet or generate a Line Up blueprint first.", 400
        
    player_evals_df = pd.DataFrame(processed_session)
    player_evals_df = player_evals_df[player_evals_df['Score'] >= 50]
    player_evals_df.columns = [str(c).strip() for c in player_evals_df.columns]
    
    player_col = next((c for c in player_evals_df.columns if c.lower() == 'player'), None)
    if player_col:
        player_evals_df.rename(columns={player_col: 'Player'}, inplace=True)
    else:
        target_col = 'Transcript' if 'Transcript' in player_evals_df.columns else (
            'Description' if 'Description' in player_evals_df.columns else player_evals_df.columns[-1]
        )
        player_evals_df['Player'] = player_evals_df[target_col].apply(
            lambda x: re.search(r'(player\s+\d+|\d+)', str(x), re.I).group(1) if re.search(r'(player\s+\d+|\d+)', str(x), re.I) else "Unknown"
        )
        
    desc_col = next((c for c in player_evals_df.columns if c.lower() == 'description'), None)
    if desc_col:
        player_evals_df.rename(columns={desc_col: 'Description'}, inplace=True)
    elif 'Transcript' in player_evals_df.columns:
        player_evals_df['Description'] = player_evals_df['Transcript']
    else:
        player_evals_df['Description'] = player_evals_df[player_evals_df.columns[-1]]

    player_evals_df['Description'] = player_evals_df.apply(
        lambda row: re.sub(
            rf"(?i)(?:^|\b)(?:\S+\s+)?{re.escape(str(row['Player']))}\b[,.\s:-]*", 
            "", 
            str(row['Description']), 
            count=1
        ).strip(),
        axis=1
    )

    # --- NORMALIZE PLAYER NAMES TO FIX CASE SENSITIVITY ---
    player_evals_df['Player'] = player_evals_df['Player'].apply(
        lambda x: f"Player {str(x).strip().title().replace('Player', '').strip()}" 
        if str(x).strip().lower().startswith('player') 
        else f"Player {str(x).strip().title()}" if str(x).strip().isdigit() 
        else str(x).strip().title()
    )

    player_evals_df = (
        player_evals_df[['Player', 'Description']]
        .groupby('Player', as_index=False)['Description']
        .agg(' '.join)
    )
        
    if uploaded_session:
        ideal_player_df = pd.DataFrame(uploaded_session)
    else:
        try:
            ideal_player_df = pd.read_html(blueprint_html)[0]
        except Exception:
            return "Error parsing system blueprint data frames.", 500
            
    ideal_player_df.columns = [str(c).strip() for c in ideal_player_df.columns]
    
    pos_col = next((c for c in ideal_player_df.columns if c.lower() == 'position'), None)
    if pos_col:
        ideal_player_df.rename(columns={pos_col: 'Position'}, inplace=True)
    else:
        ideal_player_df.rename(columns={ideal_player_df.columns[0]: 'Position'}, inplace=True)
        
    desc_target_col = next((c for c in ideal_player_df.columns if c.lower() == 'description'), None)
    if desc_target_col:
        ideal_player_df.rename(columns={desc_target_col: 'Description'}, inplace=True)
    elif len(ideal_player_df.columns) >= 2:
        ideal_player_df.rename(columns={ideal_player_df.columns[1]: 'Description'}, inplace=True)
            
    try:
        
        player_texts = player_evals_df["Description"].tolist()
        ideal_texts = ideal_player_df["Description"].tolist()  # (or whatever column name ideal_player_df uses)

        all_embeddings = get_mistral_embeddings(player_texts + ideal_texts)
        player_embeddings = all_embeddings[:len(player_texts)]
        ideal_embeddings = all_embeddings[len(player_texts):]
        
        
        #player_embeddings = [get_mistral_embeddings(desc) or [0]*1024 for desc in player_evals_df["Description"]]
        #ideal_embeddings = [get_mistral_embeddings(desc) or [0]*1024 for desc in ideal_player_df["Description"]]
        
        similarity_matrix = cosine_similarity(player_embeddings, ideal_embeddings)
        similarity_df = pd.DataFrame(similarity_matrix, index=player_evals_df["Player"], columns=ideal_player_df["Position"])
        
        top_players_per_position = {}
        for position in similarity_df.columns:
            top_players = similarity_df[position].sort_values(ascending=False)
            top_players_per_position[position] = top_players
            
        results = []
        for position, players in top_players_per_position.items():
            for player, score in players.items():
                results.append({
                    "Position": position,
                    "Confidence Score": round(float(score), 4),
                    "Player": player
                })
                
        results_df = pd.DataFrame(results)
        results_df.sort_values(by=["Position", "Confidence Score"], ascending=[True, False], inplace=True)
        results_df = results_df[["Position", "Confidence Score", "Player"]]
        cache.set('similarity_results_html', results_df.to_html(classes='table', index=False))
        
        position_rankings = {}
        for pos, group in results_df.groupby("Position", sort=False):
            position_rankings[pos] = [
                {"player": str(row["Player"]), "score": round(float(row["Confidence Score"]), 4)}
                for _, row in group.iterrows()
            ]
        cache.set('position_rankings_json', json.dumps(position_rankings))
        
        all_players = list(results_df['Player'].unique())
        all_positions = list(results_df['Position'].unique())
        
        labels = all_players + all_positions
        player_indices = {p: i for i, p in enumerate(all_players)}
        position_indices = {pos: i + len(all_players) for i, pos in enumerate(all_positions)}
        
        color_palette = px.colors.qualitative.Plotly * 3
        player_colors = {player: color_palette[i % len(color_palette)] for i, player in enumerate(all_players)}
        
        node_colors = []
        for label in labels:
            if label in player_colors:
                node_colors.append(player_colors[label])
            else:
                node_colors.append("#6c757d")
                
        sources = []
        targets = []
        values = []
        link_colors = []
        
        for _, row in results_df.sort_values('Confidence Score', ascending=False).groupby('Position', as_index=False).head(5).iterrows():
            player = row['Player']
            pos = row['Position']
            score = float(row['Confidence Score'])
            
            sources.append(player_indices[player])
            targets.append(position_indices[pos])
            values.append(score * 50)
            
            hex_color = player_colors.get(player, "#17a2b8").lstrip('#')
            rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            link_colors.append(f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.4)")
        
        dynamic_height = max(950, len(player_indices) * 32)

        sankey_fig = go.Figure(go.Sankey(
            arrangement="snap",
            node=dict(
                pad=24,
                thickness=24,
                line=dict(color="black", width=0.5),
                label=labels,
                color=node_colors
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=link_colors
            )
        ))
        
        sankey_fig.update_layout(
            title_text="Player-to-Position Flow Alignment",
            font=dict(size=13),
            height=dynamic_height,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        cache.set('sankey_json', sankey_fig.to_json())
        
        barchart_data = []
        for position, pos_data in top_players_per_position.items():
            barchart_data.append({
                "position": position,
                "players": list(pos_data.index),
                "scores": [round(float(v), 3) for v in pos_data.values]
            })
        cache.set('barchart_data', barchart_data)
        
        player_text = player_evals_df.groupby('Player')['Description'].apply(lambda x: ' '.join(x.astype(str))).to_dict()
        stopwords = set(STOPWORDS)
        wordcloud_data = []
        
        for player, raw_text in player_text.items():
            clean_tokens = " ".join(re.findall(r'\b\w{4,}\b', raw_text.lower()))
            if not clean_tokens.strip():
                continue
                
            wc = WordCloud(
                width=300,
                height=200,
                max_words=25,
                background_color='white',
                stopwords=stopwords,
                min_font_size=8,
                prefer_horizontal=0.8
            ).generate(clean_tokens)
            
            img = wc.to_image()
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            wordcloud_data.append({
                "player": player,
                "img": b64
            })
            
        cache.set('wordcloud_data', wordcloud_data)
        
        return render_template_string("<h3>Matrix Calculations Completed!</h3><script>window.location.href='/analytics';</script>")
        
    except Exception as e:
        return f"Execution matrix construction failure: {str(e)}", 500


# =====================================================================
# # SECTION 1: HTML INTERFACES (FRONTEND UI LAYOUTS)                    #
# =====================================================================
LOGIN_PAGE_HTML = """
<!DOCTYPE html>
<html>
<head><title>Sports Grader Login</title></head>
<body style="font-family: sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; background-color: #f4f6f9;">
    
    <div style="background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); width: 320px; text-align: center;">
        <h2>Sports Grader Login</h2>
        <form action="/login" method="POST">
            <input type="email" name="email" placeholder="Email" required style="width: 100%; padding: 10px; margin: 5px 0; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box;"><br>
            <input type="password" name="password" placeholder="Password" required style="width: 100%; padding: 10px; margin: 5px 0; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box;"><br>
            <button type="submit" style="width: 100%; padding: 10px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; margin-top: 10px;">Log In</button>
        </form>
        <p style="font-size: 13px; margin-top: 15px;">
            <a href="#" onclick="document.getElementById('reset-modal').style.display='block'" style="color: #dc3545; text-decoration: none;">Forgot Password? (Reset Account)</a>
        </p>
        <p style="font-size: 14px;">Don't have an account? <a href="#" onclick="document.getElementById('reg-modal').style.display='block'">Create Account</a></p>
    </div>

    <div id="reg-modal" style="display:none; position:fixed; top:10%; left:50%; transform: translateX(-50%); background:white; padding:20px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.2); width: 320px;">
        <h3>Register Account</h3>
        <form action="/register" method="POST">
            <input type="text" name="first_name" placeholder="First Name" required style="width: 100%; padding: 8px; margin: 5px 0; box-sizing: border-box;"><br>
            <input type="text" name="last_name" placeholder="Last Name" required style="width: 100%; padding: 8px; margin: 5px 0; box-sizing: border-box;"><br>
            <input type="email" name="email" placeholder="Email Address" required style="width: 100%; padding: 8px; margin: 5px 0; box-sizing: border-box;"><br>
            <input type="password" name="password" placeholder="Password (min 8 chars)" minlength="8" required style="width: 100%; padding: 8px; margin: 5px 0; box-sizing: border-box;"><br>
            
            <div style="text-align: left; margin: 10px 0; font-size: 12px;">
                <input type="checkbox" name="agreement" id="agreement" required>
                <label for="agreement">I agree to the <a href="terms.pdf" target="_blank">Terms of Service</a></label>
            </div>

            <button type="submit" style="width: 100%; padding: 10px; background: #28a745; color: white; border: none; border-radius: 4px; cursor: pointer;">Register & Send 2FA</button>
        </form>
        <button onclick="document.getElementById('reg-modal').style.display='none'" style="margin-top:10px; background:none; border:none; color:red; cursor:pointer;">Cancel</button>
    </div>

    <div id="reset-modal" style="display:none; position:fixed; top:10%; left:50%; transform: translateX(-50%); background:white; padding:20px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.2); width: 320px;">
        <h3 style="color: #007bff;">Reset Password</h3>
        <p style="font-size: 12px; color: #666;">Enter your email and a new password to update your account.</p>
        <form action="/reset-password" method="POST">
            <input type="email" name="email" placeholder="Enter your Email" required style="width: 100%; padding: 8px; margin: 5px 0; box-sizing: border-box;"><br>
            <input type="password" name="new_password" placeholder="New Password" required style="width: 100%; padding: 8px; margin: 5px 0; box-sizing: border-box;"><br>
            <button type="submit" style="width: 100%; padding: 10px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer;">Update Password</button>
        </form>
        <button onclick="document.getElementById('reset-modal').style.display='none'" style="margin-top:10px; background:none; border:none; color:black; cursor:pointer;">Cancel</button>
    </div>

</body>
</html>
"""

VERIFY_2FA_PAGE_HTML = """
<!DOCTYPE html>
<html>
<head><title>Two-Factor Authentication</title></head>
<body style="font-family: sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; background-color: #f4f6f9;">
    <div style="background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); width: 320px; text-align: center;">
        <h2 style="color: #007bff;">2-Factor Security</h2>
        <p style="font-size: 13px; color: #555;">We sent a 6-digit code from <strong>contact@evalbricks.com</strong> to {{ email }}.</p>
        <form action="/verify-2fa" method="POST">
            <input type="text" name="otp_code" placeholder="000000" maxlength="6" pattern="\\d{6}" required style="width: 100%; padding: 12px; margin: 10px 0; font-size: 20px; letter-spacing: 5px; text-align: center; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box;"><br>
            
            <div style="text-align: left; margin: 10px 0; font-size: 13px; color: #555;">
                <input type="checkbox" name="remember_device" id="remember_device" checked>
                <label for="remember_device">Remember this device for 30 days</label>
            </div>

            <button type="submit" style="width: 100%; padding: 10px; background: #28a745; color: white; border: none; border-radius: 4px; cursor: pointer; margin-top: 10px; font-weight: bold;">Verify & Continue</button>
        </form>
        <p style="font-size: 12px; margin-top: 15px;"><a href="/login" style="color: #666;">Cancel and return to login</a></p>
    </div>
</body>
</html>
"""

SHARED_CSS = """
<script async src="https://www.googletagmanager.com/gtag/js?id=G-W0VN6S115E"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-W0VN6S115E');
</script>
<script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-7701989446566369" crossorigin="anonymous"></script>
 :root {--primary-color: #007bff; --success-color: #28a745; --danger-color: #dc3545; --info-color: #17a2b8; --purple-color: #6f42c1; --dark-bg: #f4f6f9; --card-bg: #ffffff; --text-main: #333333; }
 * { box-sizing: border-box; }
 body {font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0; padding: 10px; background: var(--dark-bg); color: var(--text-main); line-height: 1.5; }
 .container {background: var(--card-bg); width: 100%; max-width: 1100px; margin: 10px auto 40px auto; padding: 20px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }
 h1, h2, h3, h4 { margin-top: 0; color: #111; }
 .btn-group {display: flex; flex-wrap: wrap; gap: 10px; margin: 15px 0; justify-content: center; }
 button, .btn-link {flex: 1 1 calc(50% - 10px); min-width: 140px; padding: 12px 18px; font-size: 14px; font-weight: bold; cursor: pointer; border: none; border-radius: 8px; transition: all 0.2s ease; text-align: center; display: inline-block; }
 @media (min-width: 768px) {body { padding: 30px; } .container { padding: 40px; } button, .btn-link { flex: 0 1 auto; } }
 button:disabled { background: #ccc !important; cursor: not-allowed; transform: none !important; }
 .table-wrap {width: 100%; overflow-x: auto; margin-top: 15px; border: 1px solid #dee2e6; border-radius: 6px; -webkit-overflow-scrolling: touch; }
 table {width: 100%; border-collapse: collapse; background: white; white-space: nowrap; }
 th, td {border: 1px solid #dee2e6; padding: 10px 14px; text-align: left; font-size: 13px; }
 th { background-color: #f8f9fa; position: sticky; top: 0; }
 .responsive-grid {display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; margin-top: 15px; }
 .plot-card {background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 15px; text-align: center; display: flex; flex-direction: column; }
 .plot-card h4 {margin: 0 0 10px 0; font-size: 14px; color: #333; }
 .chart-container {position: relative; height: 200px; width: 100%; }
 .plot-card img {width: 100%; height: auto; border-radius: 4px; }
 .top-nav { display: flex; flex-wrap: wrap; gap: 8px; background: var(--card-bg); padding: 12px 15px; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin: 10px auto 20px auto; width: 100%; max-width: 1100px; align-items: center; justify-content: center; }
 .top-nav a { text-decoration: none; color: var(--text-main); font-size: 13px; font-weight: 600; padding: 8px 14px; border-radius: 6px; transition: all 0.2s ease-in-out; background: var(--dark-bg); display: flex; align-items: center; gap: 6px; }
 .top-nav a:hover { background: var(--primary-color); color: white; transform: translateY(-2px); }
 .top-nav a.nav-home { background: #333; color: white; }
 .top-nav a.nav-home:hover { background: #111; }
 .top-nav a.nav-logout { background: #dc3545; color: white; margin-left: auto; }
 .top-nav a.nav-logout:hover { background: #c82333; }
 @media (min-width: 768px) { .top-nav { justify-content: flex-start; padding: 15px 25px; gap: 12px; } .top-nav a { font-size: 14px; padding: 10px 16px; } }
</style>
"""

SHARED_NAV = """
<nav class="top-nav">
    <a href="/" class="nav-home">🏠 Hub</a>
    <a href="/soccer-grade">🎤 Voice Logger</a>
    <a href="/upload-manager">📁 Data Manager</a>
    <a href="/create-lineup">⚽ Line Up</a>
    <a href="/analytics">📊 Analytics</a>
    <a href="/logout" class="nav-logout">🚪 Log Out</a>
</nav>
"""

SESSION_TOP_BAR_HTML = """
<div style="display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 12px; max-width: 1100px; margin: 12px auto 16px auto; padding: 12px 18px; background: #0f172a; color: #f8fafc; border-radius: 10px; border: 1px solid rgba(148, 163, 184, 0.22); box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);">
    <div style="display: flex; align-items: center; gap: 10px;">
        <span style="color: #94a3b8; font-size: 0.85rem; font-weight: 500;">Active Mode:</span>
        <span style="background: rgba(37, 99, 235, 0.2); color: #60a5fa; border: 1px solid rgba(96, 165, 250, 0.45); padding: 4px 12px; border-radius: 999px; font-size: 0.85rem; font-weight: 700;">
            {{ active_mode }}
        </span>
    </div>
    <div style="display: flex; flex-wrap: wrap; align-items: center; gap: 10px;">
        <a href="/" style="color: #cbd5e1; background: #1e293b; border: 1px solid #334155; text-decoration: none; font-size: 0.85rem; font-weight: 600; padding: 7px 13px; border-radius: 6px;">
            Mode Hub
        </a>
        <button type="button" onclick="triggerTopBarLoad()" style="color: #e2e8f0; background: #1e293b; border: 1px solid #475569; font-size: 0.85rem; font-weight: 600; padding: 7px 13px; border-radius: 6px; cursor: pointer;">
            📂 Load Session
        </button>
        <a href="/export-session" style="background: #2563eb; color: #ffffff; text-decoration: none; font-size: 0.85rem; font-weight: 700; padding: 7px 15px; border-radius: 6px;">
            💾 Save Session
        </a>
        <input type="file" id="topBarSessionFile" accept=".json" style="display: none;" onchange="handleTopBarSessionUpload(event)">
    </div>
</div>
<script>
function triggerTopBarLoad() {
    const input = document.getElementById('topBarSessionFile');
    if (input) input.click();
}
async function handleTopBarSessionUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    const formData = new FormData();
    formData.append('session_file', file);
    formData.append('expected_mode', '{{ active_mode }}');
    try {
        const res = await fetch('/import-session', { method: 'POST', body: formData });
        const data = await res.json();
        if (res.ok && data.redirect) {
            window.location.href = data.redirect;
        } else {
            alert(data.error || 'Could not load session file.');
        }
    } catch (err) {
        alert('Error uploading session file: ' + err.message);
    } finally {
        event.target.value = '';
    }
}
</script>
"""

LANDING_PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EvalBricks | Select Mode</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --dark-bg: #0b1120;
            --card-bg: #111827;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --primary-color: #2563eb;
            --primary-hover: #1d4ed8;
            --border-color: rgba(148, 163, 184, 0.2);
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            font-family: 'Inter', sans-serif;
            background: var(--dark-bg);
            color: var(--text-main);
            min-height: 100vh;
            padding: 16px;
        }
        .container {
            max-width: 1100px;
            margin: 24px auto;
        }
        .header {
            margin-bottom: 28px;
        }
        .header h1 {
            margin: 0 0 8px 0;
            font-size: 1.9rem;
            font-weight: 700;
        }
        .header h1 span { color: var(--primary-color); }
        .header p {
            margin: 0;
            color: var(--text-muted);
            font-size: 0.98rem;
        }
        .mode-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(290px, 1fr));
            gap: 20px;
        }
        .mode-card {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            padding: 22px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            gap: 18px;
        }
        .mode-badge {
            display: inline-block;
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #60a5fa;
            background: rgba(37, 99, 235, 0.16);
            border: 1px solid rgba(96, 165, 250, 0.35);
            padding: 4px 10px;
            border-radius: 999px;
            margin-bottom: 12px;
        }
        .mode-card h2 {
            margin: 0 0 8px 0;
            font-size: 1.25rem;
        }
        .mode-card p {
            margin: 0;
            color: var(--text-muted);
            font-size: 0.9rem;
            line-height: 1.5;
        }
        .mode-actions {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .btn-primary {
            display: block;
            text-align: center;
            text-decoration: none;
            background: var(--primary-color);
            color: #fff;
            font-weight: 600;
            font-size: 0.9rem;
            padding: 10px 14px;
            border-radius: 8px;
            border: none;
            cursor: pointer;
        }
        .btn-primary:hover { background: var(--primary-hover); }
        .btn-secondary {
            display: block;
            text-align: center;
            background: #1e293b;
            color: #e2e8f0;
            font-weight: 600;
            font-size: 0.9rem;
            padding: 10px 14px;
            border-radius: 8px;
            border: 1px solid #334155;
            cursor: pointer;
        }
        .btn-secondary:hover { background: #273549; }
            .modes-topbar {
      width: 100%;
      max-width: 1100px;
      margin: 0 auto 14px auto;
      display: flex;
      justify-content: flex-end;
    }
    .btn-logout-mode {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: #1e293b;
      color: #f8fafc;
      text-decoration: none;
      font-weight: 600;
      font-size: 0.9rem;
      padding: 10px 16px;
      border-radius: 8px;
      border: 1px solid #334155;
      cursor: pointer;
      transition: background 0.15s ease, border-color 0.15s ease;
    }
    .btn-logout-mode:hover {
      background: #273549;
      border-color: #475569;
    }
        </style>
    </head>
    <body>
    <div class="modes-topbar">
    <a href="/logout" class="btn-logout-mode">🚪 Log Out</a>
    </div>
    <div class="container">
        <div class="header">
            <h1>Eval<span>Bricks</span> Workspace Modes</h1>
            <p>Choose a mode below to start a brand-new session or load a previously saved session from your local drive.</p>
        </div>

        <div class="mode-grid">
            <div class="mode-card">
                <div>
                    <span class="mode-badge">Active Workspace</span>
                    <h2>TryOut Mode</h2>
                    <p>Full tryout evaluation workspace with all original landing/evaluation tools, logging, and data workflows intact.</p>
                </div>
                <div class="mode-actions">
                    <a href="/start-mode/tryout?new=1" class="btn-primary">+ Create New Session</a>
                    <button type="button" class="btn-secondary" onclick="openModeSessionPicker('TryOut Mode')">📂 Load Previous Session</button>
                </div>
            </div>

            <div class="mode-card">
                <div>
                    <span class="mode-badge">Placeholder</span>
                    <h2>Practice Mode</h2>
                    <p>Blank placeholder page ready for future Practice Mode workflows. Supports session creation and local session loading.</p>
                </div>
                <div class="mode-actions">
                    <a href="/start-mode/practice?new=1" class="btn-primary">+ Create New Session</a>
                    <button type="button" class="btn-secondary" onclick="openModeSessionPicker('Practice Mode')">📂 Load Previous Session</button>
                </div>
            </div>

            <div class="mode-card">
                <div>
                    <span class="mode-badge">Placeholder</span>
                    <h2>Post Game Mode</h2>
                    <p>Blank placeholder page ready for future Post Game Mode workflows. Supports session creation and local session loading.</p>
                </div>
                <div class="mode-actions">
                    <a href="/start-mode/postgame?new=1" class="btn-primary">+ Create New Session</a>
                    <button type="button" class="btn-secondary" onclick="openModeSessionPicker('Post Game Mode')">📂 Load Previous Session</button>
                </div>
            </div>
        </div>
    </div>

    <input type="file" id="modeSessionInput" accept=".json" style="display: none;" onchange="uploadModeSession(event)">
    <script>
        let selectedExpectedMode = 'TryOut Mode';
        function openModeSessionPicker(modeName) {
            selectedExpectedMode = modeName;
            document.getElementById('modeSessionInput').click();
        }
        async function uploadModeSession(event) {
            const file = event.target.files[0];
            if (!file) return;
            const formData = new FormData();
            formData.append('session_file', file);
            formData.append('expected_mode', selectedExpectedMode);
            try {
                const res = await fetch('/import-session', { method: 'POST', body: formData });
                const data = await res.json();
                if (res.ok && data.redirect) {
                    window.location.href = data.redirect;
                } else {
                    alert(data.error || 'Failed to load session.');
                }
            } catch (err) {
                alert('Error loading session: ' + err.message);
            } finally {
                event.target.value = '';
            }
        }
    </script>
</body>
</html>
"""

TRYOUT_MODE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EvalBricks | TryOut Mode</title>
<style>
 :root {
   --primary-color: #007bff;
   --success-color: #28a745;
   --danger-color: #dc3545;
   --info-color: #17a2b8;
   --purple-color: #6f42c1;
   --dark-bg: #f4f6f9;
   --card-bg: #ffffff;
   --text-main: #333333;
 }
 * { box-sizing: border-box; }
 body {
   font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
   margin: 0;
   padding: 10px;
   background: var(--dark-bg);
   color: var(--text-main);
   line-height: 1.5;
   text-align: center;
 }
 .container {
   background: var(--card-bg);
   width: 100%;
   max-width: 1100px;
   margin: 10px auto 40px auto;
   padding: 20px;
   border-radius: 12px;
   box-shadow: 0 4px 12px rgba(0,0,0,0.05);
 }
 h1, h2, h3, h4 { margin-top: 0; color: #111; }
 .btn-group {
   display: flex;
   flex-wrap: wrap;
   gap: 10px;
   margin: 15px 0;
   justify-content: center;
 }
 button, .btn-link {
   flex: 1 1 calc(50% - 10px);
   min-width: 140px;
   padding: 12px 18px;
   font-size: 14px;
   font-weight: bold;
   cursor: pointer;
   border: none;
   border-radius: 8px;
   transition: all 0.2s ease;
   text-align: center;
   display: inline-block;
 }
 @media (min-width: 768px) {
   body { padding: 30px; }
   .container { padding: 40px; }
   button, .btn-link { flex: 0 1 auto; }
 }
 button:disabled {
   background: #ccc !important;
   cursor: not-allowed;
   transform: none !important;
 }
 .top-nav {
   display: flex;
   flex-wrap: wrap;
   gap: 8px;
   background: var(--card-bg);
   padding: 12px 15px;
   border-radius: 10px;
   box-shadow: 0 4px 12px rgba(0,0,0,0.05);
   margin: 10px auto 20px auto;
   width: 100%;
   max-width: 1100px;
   align-items: center;
   justify-content: center;
 }
 .top-nav a {
   text-decoration: none;
   color: var(--text-main);
   font-size: 13px;
   font-weight: 600;
   padding: 8px 14px;
   border-radius: 6px;
   transition: all 0.2s ease-in-out;
   background: var(--dark-bg);
   display: flex;
   align-items: center;
   gap: 6px;
 }
 .top-nav a:hover {
   background: var(--primary-color);
   color: white;
   transform: translateY(-2px);
 }
 .top-nav a.nav-home { background: #333; color: white; }
 .top-nav a.nav-home:hover { background: #111; }
 .top-nav a.nav-logout { background: #dc3545; color: white; margin-left: auto; }
 .top-nav a.nav-logout:hover { background: #c82333; }
 @media (min-width: 768px) {
   .top-nav { justify-content: flex-start; padding: 15px 25px; gap: 12px; }
   .top-nav a { font-size: 14px; padding: 10px 16px; }
 }
 .mode-subtitle {
   text-align: center;
   color: #666;
   margin-bottom: 24px;
   font-size: 14px;
 }
 .grid {
   display: grid;
   grid-template-columns: 1fr;
   gap: 20px;
   margin: 25px 0 30px 0;
 }
 @media (min-width: 576px) { .grid { grid-template-columns: repeat(2, 1fr); } }
 @media (min-width: 992px) { .grid { grid-template-columns: repeat(4, 1fr); } }
 .card {
   background: white;
   padding: 25px;
   border-radius: 12px;
   box-shadow: 0 4px 12px rgba(0,0,0,0.05);
   text-align: left;
   transition: transform 0.2s, box-shadow 0.2s;
   text-decoration: none;
   color: inherit;
   display: flex;
   flex-direction: column;
   justify-content: space-between;
   border-top: 4px solid var(--primary-color);
 }
 .card:hover { transform: translateY(-4px); box-shadow: 0 8px 20px rgba(0,0,0,0.1); }
 .card h3 { margin-bottom: 10px; color: #222; font-size: 18px; }
 .card p { color: #555; font-size: 13px; line-height: 1.5; margin-bottom: 20px; }
 .badge {
   display: inline-block;
   background: #e1ecf4;
   color: #39739d;
   font-size: 11px;
   padding: 4px 8px;
   border-radius: 4px;
   font-weight: bold;
   align-self: flex-start;
 }
 .system-controls {
   margin-top: 35px;
   border-top: 1px solid #ddd;
   padding-top: 20px;
   display: flex;
   flex-wrap: wrap;
   gap: 12px;
   justify-content: center;
 }
 .btn-reset-all {
   background: #6c757d;
   color: white;
   width: 100%;
   max-width: 300px;
 }
 .debug-panel {
   background: #f1f3f5;
   border-radius: 8px;
   padding: 15px;
   font-size: 12px;
   margin-top: 25px;
   text-align: left;
 }
</style>
</head>
<body>
""" + SHARED_NAV + """
""" + SESSION_TOP_BAR_HTML + """
<div class="container">
    <h1>EvalBricks TryOut Mode</h1>
    <p class="mode-subtitle">
        Active evaluation hub. Raw records: <strong>{{ raw_count|default(0) }}</strong> |
        Processed records: <strong>{{ processed_count|default(0) }}</strong>
    </p>

    <div class="grid">
        <a href="/soccer-grade" class="card" style="border-top-color: var(--danger-color)">
            <div>
                <h3>1. Voice Logger</h3>
                <p>Voice-to-text logging assistant featuring automated player-by-player row splitting.</p>
            </div>
            <span class="badge">Voice Input</span>
        </a>

        <a href="/upload-manager" class="card" style="border-top-color: var(--primary-color)">
            <div>
                <h3>2. Data Manager</h3>
                <p>Upload external metrics data or spreadsheets and manage session assets.</p>
            </div>
            <span class="badge">File Processing</span>
        </a>

        <a href="/create-lineup" class="card" style="border-top-color: #ffc107">
            <div>
                <h3>3. Create Line Up</h3>
                <p>Build and arrange tactical team lineups utilizing active roster data templates.</p>
            </div>
            <span class="badge">Tactics</span>
        </a>

        <a href="/analytics" class="card" style="border-top-color: var(--success-color)">
            <div>
                <h3>4. Analytics</h3>
                <p>Review raw files, processed logs, and uploaded metrics dashboards side by side.</p>
            </div>
            <span class="badge">Insights</span>
        </a>
    </div>

    <div class="system-controls">
        <button type="button" class="btn-reset-all" onclick="clearFullSession()">Reset Environment Roster Cache</button>
    </div>

    <div class="debug-panel">
        <h4 style="margin:0;">Session Context:</h4>
        <ul style="margin: 5px 0 0 0; padding-left: 20px;">
            <li>Raw Rows Array: {{ raw_count|default(0) }}</li>
            <li>Exploded Rows Array: {{ processed_count|default(0) }}</li>
            <li>Uploaded Matrix Metrics: {{ uploaded_count|default(0) }}</li>
        </ul>
    </div>
</div>
<script>
function clearFullSession() {
    if (confirm("Are you sure you want to completely flush all loaded dataframes and generated files?")) {
        fetch('/system/reset-session', { method: 'POST' })
            .then(res => res.json())
            .then(() => {
                alert("Cache context cleared successfully.");
                window.location.reload();
            });
    }
}
</script>
</body>
</html>"""
PRACTICE_MODE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EvalBricks | Practice Mode</title>
<style>
 :root {
   --primary-color: #007bff;
   --success-color: #28a745;
   --danger-color: #dc3545;
   --info-color: #17a2b8;
   --purple-color: #6f42c1;
   --dark-bg: #f4f6f9;
   --card-bg: #ffffff;
   --text-main: #333333;
 }
 * { box-sizing: border-box; }
 body {
   font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
   margin: 0;
   padding: 10px;
   background: var(--dark-bg);
   color: var(--text-main);
   line-height: 1.5;
 }
 .container {
   background: var(--card-bg);
   width: 100%;
   max-width: 1100px;
   margin: 10px auto 40px auto;
   padding: 20px;
   border-radius: 12px;
   box-shadow: 0 4px 12px rgba(0,0,0,0.05);
 }
 @media (min-width: 768px) {
   body { padding: 30px; }
   .container { padding: 40px; }
 }
 .mode-subtitle {
   color: #666;
   font-size: 14px;
   margin-bottom: 20px;
 }
 .practice-workspace {
   border: 2px dashed #cbd5e1;
   background: #f8fafc;
   border-radius: 10px;
   padding: 28px;
   margin-top: 20px;
 }
</style>
</head>
<body>
""" + SHARED_NAV + """
""" + SESSION_TOP_BAR_HTML + """
<div class="container">
    <h1>EvalBricks Practice Mode</h1>
    <p class="mode-subtitle">
        Active Mode: <strong>{{ active_mode|default("Practice Mode") }}</strong> |
        Raw records: <strong>{{ raw_count|default(0) }}</strong> |
        Processed records: <strong>{{ processed_count|default(0) }}</strong>
    </p>
    <div class="practice-workspace">
        <h3>Practice Mode Workspace</h3>
        <p style="color:#64748b; margin-bottom:0;">
            Ready for a new practice session. Use the top session bar to save, load, or start a new session.
        </p>
    </div>
</div>
</body>
</html>"""
POSTGAME_MODE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EvalBricks | Post Game Mode</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --dark-bg: #0b1120;
            --card-bg: #111827;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --primary-color: #2563eb;
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            font-family: 'Inter', sans-serif;
            background: var(--dark-bg);
            color: var(--text-main);
            min-height: 100vh;
            padding: 16px;
        }
        .placeholder-box {
            max-width: 1100px;
            margin: 24px auto;
            min-height: 55vh;
            background: var(--card-bg);
            border: 1px dashed rgba(148, 163, 184, 0.3);
            border-radius: 14px;
            padding: 32px;
        }
    </style>
</head>
<body>
    {{ shared_nav | safe }}
""" + SESSION_TOP_BAR_HTML + """
    <div class="placeholder-box"></div>
</body>
</html>
"""

SOCCER_INTERFACE_HTML = """<!DOCTYPE html> 
<html lang="en"> 
<head> 
<meta charset="UTF-8"> 
<meta name="viewport" content="width=device-width, initial-scale=1.0"> 
<title>Audio Voice Logger</title> 
""" + SHARED_CSS + """ 
<style> 
 .btn-record { background: var(--danger-color); color: white; } 
 .btn-stop { background: var(--success-color); color: white; } 
 .btn-process { background: var(--info-color); color: white; } 
 .btn-save { background: var(--primary-color); color: white; } 
 .btn-save-processed { background: var(--purple-color); color: white; } 
 #status { margin: 20px 0; font-weight: bold; color: #555; text-align: center; } 
 .flex-container {display: grid; grid-template-columns: 1fr; gap: 20px; margin-top: 20px; } 
 @media(min-width: 768px) { .flex-container { grid-template-columns: 1fr 1fr; } } 
 .history-container { width: 100%; text-align: left; } 
 .history-list {background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 0; list-style: none; max-height: 300px; overflow-y: auto; } 
 .history-item { padding: 12px; border-bottom: 1px solid #eee; font-size: 13px; display: flex; justify-content: space-between; align-items: center; } 
 .timestamp { color: #888; font-weight: bold; margin-right: 5px; font-size: 11px; flex-shrink: 0; } 
 .transcript-text { flex-grow: 1; margin: 0 10px; }
 .score-badge { background: #e2e8f0; color: #2d3748; padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: bold; flex-shrink: 0; }
</style> 
</head> 
<body> 
""" + SHARED_NAV + """
""" + SESSION_TOP_BAR_HTML + """
<div class="container"> 
    <h1 style="text-align:center;">Voice Recorder Logger</h1> 
    <p style="text-align:center; color:#666;">Click "Start Recording" to open your mic, and "Stop & Process" to transcribe.</p> 
    <div class="btn-group"> 
        <button id="start-btn" class="btn-record">Start Recording</button> 
        <button id="stop-btn" class="btn-stop" disabled>Stop & Process</button> 
    </div> 
    <div class="btn-group" style="border-top: 1px solid #eee; padding-top: 20px;"> 
        <button id="process-btn" class="btn-process" {% if not has_raw %}disabled{% endif %}>Split & Process Rows</button> 
        <button id="export-btn" class="btn-save" {% if not has_raw %}disabled{% endif %}>Export Raw CSV</button> 
        <button id="export-processed-btn" class="btn-save-processed" {% if not has_processed %}disabled{% endif %}>Export Processed CSV</button> 
    </div> 
    <div id="status">Status: Idle</div> 
    <div class="flex-container"> 
        <div class="history-container"> 
            <h3>Raw Transcripts:</h3> 
            <ul id="history-list" class="history-list"> 
                {% if has_raw %} 
                    {% for row in raw_records %} 
                        <li class="history-item"><span class="timestamp">[{{ row.Timestamp }}]</span> <span class="transcript-text">{{ row.Transcript }}</span></li> 
                    {% endfor %} 
                {% else %} 
                    <li class="history-item" id="empty-state" style="color: #aaa; text-align:center; display:block;">No raw records yet.</li> 
                {% endif %} 
            </ul> 
        </div> 
        <div class="history-container"> 
            <h3>Processed Lines:</h3> 
            <ul id="processed-list" class="history-list"> 
                {% if has_processed %} 
                    {% for row in processed_records %} 
                        <li class="history-item">
                            <span class="timestamp">[{{ row.Timestamp }}]</span> 
                            <span class="transcript-text">{{ row.Transcript }}</span>
                            <span class="score-badge">Score: {{ row.Score }}</span>
                        </li> 
                    {% endfor %} 
                {% else %} 
                    <li class="history-item" id="empty-processed" style="color: #aaa; text-align:center; display:block;">No split data yet.</li> 
                {% endif %} 
            </ul> 
        </div> 
    </div> 
</div> 
<script> 
let mediaRecorder; 
let audioChunks = []; 
const startBtn = document.getElementById('start-btn'); 
const stopBtn = document.getElementById('stop-btn'); 
const processBtn = document.getElementById('process-btn'); 
const exportBtn = document.getElementById('export-btn'); 
const exportProcessedBtn = document.getElementById('export-processed-btn'); 
const statusDiv = document.getElementById('status'); 
const historyList = document.getElementById('history-list'); 
const processedList = document.getElementById('processed-list'); 
let sessionRecords = {{ raw_records_json|safe }}; 
let processedRecords = {{ processed_records_json|safe }}; 

function appendToSessionDOM(timestamp, transcript) {
    const emptyState = document.getElementById('empty-state'); 
    if (emptyState) emptyState.remove(); 
    sessionRecords.push({ Timestamp: timestamp, Transcript: transcript }); 
    exportBtn.disabled = false; 
    processBtn.disabled = false; 
    const li = document.createElement('li'); 
    li.className = 'history-item'; 
    li.innerHTML = `<span class="timestamp">[${timestamp}]</span> <span class="transcript-text">${transcript}</span>`; 
    historyList.insertBefore(li, historyList.firstChild); 
    fetch('/soccer-grade/sync-raw', {method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ data: sessionRecords }) }); 
} 

let lastProcessedIndex = 0;

processBtn.addEventListener('click', async () => {
    if (sessionRecords.length === 0) return; 
    
    const newRawRows = sessionRecords.slice(lastProcessedIndex);
    if (newRawRows.length === 0) {
        statusDiv.style.color = 'orange';
        statusDiv.innerText = "Status: No new raw rows to process!";
        return;
    }

    statusDiv.innerText = `Status: Processing ${newRawRows.length} new transcript(s)...`; 
    
    try {
        const response = await fetch('/soccer-grade/split-dataframe', {
            method: 'POST', 
            headers: { 'Content-Type': 'application/json' }, 
            body: JSON.stringify({ raw_rows: newRawRows.map(r => [r.Timestamp, r.Transcript]) }) 
        }); 
        const result = await response.json(); 
        
        if (result.status === 'success') {
            const newlyProcessed = result.processed_records || [];
            processedRecords = processedRecords.concat(newlyProcessed);
            
            newlyProcessed.forEach(row => {
                const li = document.createElement('li'); 
                li.className = 'history-item'; 
                li.innerHTML = `
                    <span class="timestamp">[${row.Timestamp}]</span> 
                    <span class="transcript-text">${row.Transcript}</span> 
                    <span class="score-badge">Score: ${row.Score}</span>
                `; 
                processedList.appendChild(li); 
            }); 
            
            lastProcessedIndex = sessionRecords.length;
            
            exportProcessedBtn.disabled = false; 
            statusDiv.style.color = 'green'; 
            statusDiv.innerText = "Status: New rows processed and appended!";         
        } else {
            statusDiv.style.color = 'red'; 
            statusDiv.innerText = "Error: " + (result.message || "Unknown error");
        }
    } catch (err) {
        statusDiv.style.color = 'red'; 
        statusDiv.innerText = "Server error during row processing."; 
    } 
});

async function downloadCSV(records, filename, isProcessed = false) {
    if (!records || records.length === 0) return;
    let csvContent = isProcessed ? "Timestamp,Transcript,Score\\n" : "Timestamp,Transcript\\n"; 
    records.forEach(row => {
        let text = row.Transcript || ""; 
        let time = row.Timestamp || ""; 
        let cleanTranscript = text.replace(/"/g, '""'); 
        if (isProcessed) {
            let score = row.Score !== undefined ? row.Score : 0;
            csvContent += `"${time}","${cleanTranscript}",${score}\\n`;
        } else {
            csvContent += `"${time}","${cleanTranscript}"\\n`; 
        }
    }); 
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' }); 
    const url = URL.createObjectURL(blob); 
    const link = document.createElement("a"); 
    link.setAttribute("href", url); 
    link.setAttribute("download", filename); 
    document.body.appendChild(link); 
    link.click(); 
    document.body.removeChild(link); 
} 

exportBtn.addEventListener('click', () => downloadCSV(sessionRecords, 'voice_history.csv', false)); 
exportProcessedBtn.addEventListener('click', () => downloadCSV(processedRecords, 'processed_voice_data.csv', true)); 

startBtn.addEventListener('click', async () => {
    audioChunks = []; 
    statusDiv.innerText = "Status: Requesting microphone access..."; 
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true }); 
        mediaRecorder = new MediaRecorder(stream); 
        mediaRecorder.ondataavailable = event => { audioChunks.push(event.data); }; 
        mediaRecorder.onstop = async () => {
            statusDiv.innerText = "Status: Transcribing audio file..."; 
            const audioBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType || 'audio/webm' }); 
            const formData = new FormData(); 
            formData.append('audio_data', audioBlob, 'recording.webm'); 
            fetch('/soccer-grade/process-audio', { method: 'POST', body: formData }) 
            .then(response => response.json()) 
            .then(data => {
                if (data.status === 'success') {
                    statusDiv.style.color = 'green'; 
                    statusDiv.innerText = "Transcribed successfully!"; 
                    appendToSessionDOM(data.timestamp, data.transcript); 
                } else {
                    statusDiv.style.color = 'red'; 
                    statusDiv.innerText = data.message; 
                } 
            }); 
        }; 
        mediaRecorder.start(); 
        statusDiv.innerText = "Status: Recording... speak now."; 
        startBtn.disabled = true; 
        stopBtn.disabled = false; 
    } catch (err) {
        statusDiv.style.color = 'red'; 
        statusDiv.innerText = "Status: Microphone access denied."; 
    } 
}); 

stopBtn.addEventListener('click', () => {
    mediaRecorder.stop(); 
    mediaRecorder.stream.getTracks().forEach(track => track.stop()); 
    startBtn.disabled = false; 
    stopBtn.disabled = true; 
}); 
</script> 
</body> 
</html>"""

UPLOAD_PAGE_HTML = """<!DOCTYPE html> 
<html lang="en"> 
<head> 
<meta charset="UTF-8"> 
<meta name="viewport" content="width=device-width, initial-scale=1.0"> 
<title>Data Upload Manager</title> 
""" + SHARED_CSS + """ 
<style> 
 .data-block, .upload-block {background: #f1f3f5; border-radius: 8px; padding: 15px; margin-bottom: 20px; max-height: 250px; overflow-y: auto; } 
 .upload-block { background: #e3f2fd; border: 1px solid #90caf9; } 
 .upload-flex-grid {display: grid; grid-template-columns: 1fr; gap: 20px; margin-top: 20px; } 
 @media(min-width: 768px) { .upload-flex-grid { grid-template-columns: 1fr 1fr; } } 
 .upload-zone {border: 2px dashed var(--primary-color); padding: 25px; text-align: center; background: #f8f9fa; border-radius: 8px; } 
 .upload-zone.override { border-color: var(--danger-color); background: #fff5f5; } 
 .upload-zone input[type="file"] { margin: 15px 0; width: 100%; } 
 .upload-zone button { width: 100%; } 
 .badge-alert { background: var(--danger-color); color: white; padding: 3px 6px; font-size: 11px; font-weight: bold; border-radius: 4px; display: inline-block; } 
</style> 
</head> 
<body> 
""" + SHARED_NAV + """
<div class="container"> 
    <h2>Data Upload Manager</h2> 
    <p style="color:#666;">Manage your imported spreadsheet files separately alongside your captured active voice logs.</p> 
    
    <h3 style="color: #0d47a1;">Active Uploaded Positions Spreadsheet Data</h3> 
    <div class="upload-block table-wrap"> 
        {% if uploaded_data_table %} 
            {{ uploaded_data_table|safe }} 
        {% else %} 
            <span style="color:#666; font-style: italic;">No uploaded spreadsheet records currently loaded.</span> 
        {% endif %} 
    </div> 
    
    <h3>Voice Stream Snapshot (Raw)</h3> 
    <div class="data-block table-wrap"> 
        {% if raw_data_table %} 
            {{ raw_data_table|safe }} 
        {% else %} 
            <span style="color:#999;">No raw recording logs currently loaded.</span> 
        {% endif %} 
    </div> 
    
    <h3>Voice Stream Snapshot (Processed & Exploded) {% if was_overridden %}<span class="badge-alert">Testing Override Active</span>{% endif %}</h3> 
    <div class="data-block table-wrap" {% if was_overridden %}style="border-left: 4px solid var(--danger-color);"{% endif %}> 
        {% if processed_data_table %} 
            {{ processed_data_table|safe }} 
        {% else %} 
            <span style="color:#999;">No processed/split comment structures currently loaded.</span> 
        {% endif %} 
    </div> 
    
    <div class="upload-flex-grid"> 
        <div class="upload-zone"> 
            <form action="/upload-manager/submit-file" method="POST" enctype="multipart/form-data"> 
                <label style="font-weight:bold; display:block; color:#0d47a1;">Ideal Skills and Traits</label>
                <span style="font-size: 11px; color: #7f8c8d; display:block;">Create positions and what type of player should fill them</span> 
                <span style="font-size: 11px; color: #7f8c8d; display:block;">csv file with 'Position' and 'Description' as columns</span> 
                <input type="file" name="uploaded_csv" accept=".csv" required> 
                <button type="submit" style="background: var(--success-color); color:white;">Upload Position Descriptions</button> 
            </form> 
        </div> 
        <div class="upload-zone override"> 
            <form action="/upload-manager/override-processed" method="POST" enctype="multipart/form-data"> 
                <label style="font-weight:bold; display:block; color:#c0392b;">Sandbox Testing Mock</label> 
                <span style="font-size: 11px; color: #7f8c8d; display:block;">Forces override of Processed & Exploded frame</span> 
                <span style="font-size: 11px; color: #7f8c8d; display:block;">Create csv file with 'Timestamp', 'Transcript', and 'Score'</span> 
                <input type="file" name="mock_processed_csv" accept=".csv" required> 
                <button type="submit" style="background: var(--danger-color); color:white;">Inject Test Override</button> 
            </form> 
        </div> 
    </div> 
</div> 
</body> 
</html>"""

LINEUP_PAGE_HTML = """<!DOCTYPE html> 
<html lang="en"> 
<head> 
<meta charset="UTF-8"> 
<meta name="viewport" content="width=device-width, initial-scale=1.0"> 
<title>Create Line Up</title> 
""" + SHARED_CSS + """ 
<style> 
 .placeholder-box { border: 2px dashed #ffc107; background: #fffde7; padding: 20px; border-radius: 8px; text-align: center; margin-bottom: 25px; } 
 .format-selector { display: flex; gap: 10px; margin: 20px 0; justify-content: center; } 
 .btn-format { background: white; color: var(--primary-color); border: 2px solid var(--primary-color); padding: 10px 20px; flex: 1; max-width: 150px; cursor: pointer; } 
 .btn-format.active { background: var(--primary-color); color: white; } 
 .btn-execute { background: var(--success-color); color: white; } 
 .btn-clear-frame { background: var(--danger-color); color: white; } 
 .llm-response-box { background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px; margin-top: 20px; min-height: 100px; } 
 .debug-panel { background: #f1f3f5; border-radius: 8px; padding: 15px; font-size: 12px; margin-top: 30px; } 
 .loader { display: none; text-align: center; font-weight: bold; color: #666; margin: 20px 0; } 
 .tactical-container { background-color: #2b2b2b; color: white; font-family: Arial, sans-serif; text-align: center; padding: 15px; border-radius: 8px; display: inline-block; max-width: 100%; }
 .tactical-hint { font-size: 13px; color: #ccc; }
 .board-wrapper { display: none; margin: 20px auto; text-align: center; }
</style> 
</head> 
<body> 
""" + SHARED_NAV + """
<div class="container"> 
    <h2>Line Up Builder Workspace</h2> 
    <p style="color:#666; text-align: center;">Select a configuration grid below to evaluate tactical alignment blueprints via Mistral AI.</p> 
    
    <h3>1. Select Sport</h3>
    <div class="format-selector sport-selector">
        <button type="button" id="btn-soccer" class="btn-format {% if selected_sport == 'soccer' or not selected_sport %}active{% endif %}" onclick="selectSport(this, 'soccer')">Soccer</button>
        <button type="button" id="btn-basketball" class="btn-format {% if selected_sport == 'basketball' %}active{% endif %}" onclick="selectSport(this, 'basketball')">Basketball</button>
        <button type="button" id="btn-volleyball" class="btn-format {% if selected_sport == 'volleyball' %}active{% endif %}" onclick="selectSport(this, 'volleyball')">Volleyball</button>
    </div>

    <div id="matrix-format-container"></div>
    
    <!-- Interactive Tactical Boards Container -->
    <div id="tactical-boards-stage">
        <!-- Interactive Soccer Formation Board -->
        <div id="soccer-pitch-wrapper" class="board-wrapper" style="display: block;">
            <div class="tactical-container">
                <h3 style="margin-top: 0;">Interactive Soccer Formation Board (<span id="active-soccer-format-label">7v7</span>)</h3>
                <p class="tactical-hint">Click and drag any player position. Labels update dynamically based on field zones!</p>
                <canvas id="soccerPitchCanvas" width="400" height="617" style="background-color: #2d6a30; border: 3px solid white; box-shadow: 0px 4px 15px rgba(0,0,0,0.5); cursor: pointer; touch-action: none; max-width: 100%;"></canvas>
            </div>
        </div>

        <!-- Interactive Volleyball Court Board -->
        <div id="volleyball-court-wrapper" class="board-wrapper">
            <div class="tactical-container">
                <h3 style="margin-top: 0;">Interactive Volleyball Court Board (<span id="active-volleyball-format-label">6x6</span>)</h3>
                <p class="tactical-hint">Click and drag any player across the 6 bottom-half court zones. Labels update dynamically by zone/side!</p>
                <canvas id="volleyballCourtCanvas" width="400" height="667" style="background-color: #c88a4b; border: 3px solid white; box-shadow: 0px 4px 15px rgba(0,0,0,0.5); cursor: pointer; touch-action: none; max-width: 100%;"></canvas>
            </div>
        </div>

        <!-- Interactive Full-Court Basketball Board -->
        <div id="basketball-court-wrapper" class="board-wrapper">
            <div class="tactical-container">
                <h3 style="margin-top: 0;">Interactive Basketball Full Court Board (<span id="active-basketball-format-label">5v5</span>)</h3>
                <p class="tactical-hint">Click and drag any player anywhere across the full court. Labels update dynamically by court zone & spacing!</p>
                <canvas id="basketballCourtCanvas" width="400" height="747" style="background-color: #c88a4b; border: 3px solid white; box-shadow: 0px 4px 15px rgba(0,0,0,0.5); cursor: pointer; touch-action: none; max-width: 100%;"></canvas>
            </div>
        </div>
    </div>
    
    <div class="btn-group"> 
        <button type="button" id="execute-btn" class="btn-execute" onclick="runTacticalPrompt()">Execute Blueprint Generation</button> 
        <button type="button" id="clear-btn" class="btn-clear-frame" onclick="clearBlueprintFrame()">Clear Frame</button> 
    </div> 
    <div class="loader" id="loading-spinner">Querying structural Mistral network matrix layers...</div> 
    
    <h3>2. Generated Dataframe Array</h3> 
    <div class="llm-response-box table-wrap" id="response-anchor"> 
        {% if blueprint_table %} 
            {{ blueprint_table|safe }} 
        {% else %} 
            <span style="color:#999; font-style: italic;">No configuration structure generated yet.</span> 
        {% endif %} 
    </div> 

    <!-- Separate Container for Extracted Board Positions Table -->
    <div id="board-positions-container" style="margin-top: 25px;">
        <div class="table-wrap" style="border: 2px solid #2e7d32; border-radius: 8px;">
            <table style="width: 100%; border-collapse: collapse;">
                <thead>
                    <tr style="background: #e8f5e9;">
                        <th style="padding: 12px; color: #2e7d32; text-align: left; font-size: 14px;">Active Tactical Board Layout (Extracted Positions)</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td id="board-positions-cell" style="padding: 15px; font-family: monospace; font-weight: bold; color: #1b5e20; font-size: 15px; background: #ffffff;">
                            <span style="color: #999; font-style: italic; font-weight: normal;">No board positions extracted yet.</span>
                        </td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>
    
    <div class="debug-panel"> 
        <h4 style="margin:0;">Session Context:</h4> 
        <ul style="margin: 5px 0 0 0; padding-left: 20px;"> 
            <li>Raw Rows Array: {{ raw_count }}</li> 
            <li>Exploded Rows Array: {{ processed_count }}</li> 
            <li>Uploaded Matrix Metrics: {{ uploaded_count }}</li> 
        </ul> 
    </div> 
</div> 
<script>
const SPORT_FORMATS = {
    soccer: {
        options: ['7v7', '9v9', '11v11'],
        defaultFormat: '7v7',
        wrapperId: 'soccer-pitch-wrapper',
        labelId: 'active-soccer-format-label',
        canvasId: 'soccerPitchCanvas',
        dimensions: { width: 68, length: 105 },
        formations: {
            11: [
                {pos: 'GK', x: 34, y: 6, locked: true},
                {pos: 'LDB', x: 12, y: 26},
                {pos: 'CDB', x: 26, y: 24},
                {pos: 'CDB', x: 42, y: 24},
                {pos: 'RDB', x: 56, y: 26},
                {pos: 'LM', x: 18, y: 55},
                {pos: 'CM', x: 34, y: 50},
                {pos: 'RM', x: 50, y: 55},
                {pos: 'LFW', x: 15, y: 78},
                {pos: 'CF', x: 34, y: 80},
                {pos: 'RFW', x: 53, y: 78}
            ],
            9: [
                {pos: 'GK', x: 34, y: 6, locked: true},
                {pos: 'LDB', x: 15, y: 26},
                {pos: 'CDB', x: 34, y: 24},
                {pos: 'RDB', x: 53, y: 26},
                {pos: 'CM', x: 24, y: 52},
                {pos: 'CM', x: 44, y: 52},
                {pos: 'LFW', x: 15, y: 78},
                {pos: 'CF', x: 34, y: 80},
                {pos: 'RFW', x: 53, y: 78}
            ],
            7: [
                {pos: 'GK', x: 34, y: 6, locked: true},
                {pos: 'LDB', x: 22, y: 26},
                {pos: 'RDB', x: 46, y: 26},
                {pos: 'LM', x: 15, y: 55},
                {pos: 'CM', x: 34, y: 52},
                {pos: 'RM', x: 53, y: 55},
                {pos: 'CF', x: 34, y: 80}
            ]
        }
    },
    volleyball: {
        options: ['6x6', '2v2'],
        defaultFormat: '6x6',
        wrapperId: 'volleyball-court-wrapper',
        labelId: 'active-volleyball-format-label',
        canvasId: 'volleyballCourtCanvas',
        dimensions: { width: 9.0, length: 18.0 },
        formations: {
            '6x6': [
                // Front row (near net at y=9, bottom half y=0..9): L->R in court coords is Zone 4, Zone 3, Zone 2
                {pos: 'OH', x: 2.25, y: 7.2, zone: 4},
                {pos: 'MB', x: 4.50, y: 7.2, zone: 3},
                {pos: 'S',  x: 6.75, y: 7.2, zone: 2},
                // Back row (bottom half y=0..4.5): L->R in court coords is Zone 5, Zone 6, Zone 1
                {pos: 'DS',  x: 2.25, y: 2.5, zone: 5},
                {pos: 'L',   x: 4.50, y: 2.5, zone: 6, locked: true},
                {pos: 'OPP', x: 6.75, y: 2.5, zone: 1}
            ],
            '2v2': [
                {pos: 'L', x: 2.5, y: 5.2, zone: 4},
                {pos: 'R', x: 6.5, y: 5.2, zone: 2}
            ]
        }
    },
    basketball: {
        options: ['5v5', '3v3'],
        defaultFormat: '5v5',
        wrapperId: 'basketball-court-wrapper',
        labelId: 'active-basketball-format-label',
        canvasId: 'basketballCourtCanvas',
        // Full court coordinate system: width 15.0m (x: 0..15), length 28.0m (y: 0 = bottom baseline, y: 28 = top baseline)
        dimensions: { width: 15.0, length: 28.0 },
        formations: {
            '5v5': [
                {pos: 'PG', x: 7.5,  y: 12.2, zone: 3},
                {pos: 'SG', x: 12.6, y: 9.0,  zone: 5},
                {pos: 'SF', x: 2.4,  y: 8.8,  zone: 4},
                {pos: 'PF', x: 10.2, y: 4.2,  zone: 2},
                {pos: 'C',  x: 6.2,  y: 2.8,  zone: 1}
            ],
            '3v3': [
                {pos: 'G', x: 7.5,  y: 11.5, zone: 3},
                {pos: 'W', x: 12.5, y: 6.5,  zone: 5},
                {pos: 'B', x: 6.0,  y: 3.0,  zone: 1}
            ]
        }
    }
};

let currentPlayers = [];
let selectedPlayer = null;
let activeBoardSport = null;
let activeBoardFormat = null;
let selectedSportName = "{{ selected_sport|safe }}" || "soccer";
let selectedFormatName = "{{ selected_format|safe }}" || "7v7";

function hideAllBoards() {
    document.querySelectorAll('.board-wrapper').forEach(el => {
        el.style.display = 'none';
    });
}

function renderFormatButtons(sportType, activeFormat) {
    const container = document.getElementById('matrix-format-container');
    const sportCfg = SPORT_FORMATS[sportType];
    if (!container || !sportCfg) return;

    const validFormat = sportCfg.options.includes(activeFormat) ? activeFormat : sportCfg.defaultFormat;
    selectedFormatName = validFormat;

    const buttonsHtml = sportCfg.options.map(fmt => {
        const safeId = `btn-${fmt.replace(/[^a-zA-Z0-9_-]/g, '')}`;
        const activeClass = fmt === validFormat ? 'active' : '';
        return `<button type="button" id="${safeId}" class="btn-format ${activeClass}" onclick="selectFormat(this, '${fmt}')">${fmt}</button>`;
    }).join('');

    container.innerHTML = `
        <h3>1. Select Roster Matrix Format</h3>
        <div class="format-selector">${buttonsHtml}</div>
    `;
}

function computeVolleyballZone(x, y) {
    // 6 zones on bottom half (0 <= y <= 9):
    // Front half (y > 4.5): 4 (left), 3 (center), 2 (right)
    // Back half (y <= 4.5): 5 (left), 6 (center), 1 (right)
    const isFront = y > 4.5;
    if (x < 3.0) return isFront ? 4 : 5;
    if (x < 6.0) return isFront ? 3 : 6;
    return isFront ? 2 : 1;
}

function computeBasketballZone(x, y) {
    // 5 unrendered tactical zones across full court (width 15m, length 28m):
    // Z1: Lower Paint / Rim Zone (y <= 6.5, 5.0 <= x <= 10.0)
    // Z2: Upper Paint / Rim Zone (y >= 21.5, 5.0 <= x <= 10.0)
    // Z3: Midcourt / Transition Hub (10.0 < y < 18.0, 4.5 <= x <= 10.5)
    // Z4: Left Wing / Perimeter Flank (x < 5.0)
    // Z5: Right Wing / Perimeter Flank (x >= 10.0)
    if (y <= 6.5 && x >= 5.0 && x <= 10.0) return 1;
    if (y >= 21.5 && x >= 5.0 && x <= 10.0) return 2;
    if (x < 5.0) return 4;
    if (x >= 10.0) return 5;
    return 3;
}

function updatePlayerPositionLabel(p, sportType, formatType) {
    if (!p || p.locked) return;

    if (sportType === 'soccer') {
        if (p.pos === 'GK') return;
        const col = p.x <= 22.66 ? 'left' : (p.x <= 45.33 ? 'center' : 'right');
        const row = p.y <= 35 ? 'defensive' : (p.y <= 70 ? 'middle' : 'attacking');

        if (row === 'attacking') {
            p.pos = col === 'left' ? 'LFW' : (col === 'center' ? 'CF' : 'RFW');
        } else if (row === 'middle') {
            p.pos = col === 'left' ? 'LM' : (col === 'center' ? 'CM' : 'RM');
        } else if (row === 'defensive') {
            p.pos = col === 'left' ? 'LDB' : (col === 'center' ? 'CDB' : 'RDB');
        }
        return;
    }

    if (sportType === 'volleyball') {
        const zone = computeVolleyballZone(p.x, p.y);
        p.zone = zone;

        if (formatType === '2v2') {
            p.pos = p.x < 4.5 ? 'L' : 'R';
            return;
        }

        // 6x6 dynamic role assignment based on front/back row and left/center/right zone
        const isFront = yIsFrontRowVolleyball(p.y);
        const col = p.x < 3.0 ? 'left' : (p.x < 6.0 ? 'center' : 'right');

        if (isFront) {
            if (col === 'left') p.pos = 'OH';
            else if (col === 'center') p.pos = 'MB';
            else p.pos = 'S';
        } else {
            if (col === 'left') p.pos = 'DS';
            else if (col === 'center') p.pos = 'L';
            else p.pos = 'OPP';
        }
        return;
    }

    if (sportType === 'basketball') {
        const zone = computeBasketballZone(p.x, p.y);
        p.zone = zone;

        // Distance from nearest baseline (0m or 28m) so both halves map symmetrically
        const distToBaseline = Math.min(p.y, 28.0 - p.y);

        if (formatType === '3v3') {
            if (distToBaseline <= 5.2 && p.x >= 4.8 && p.x <= 10.2) p.pos = 'B';
            else if (p.x < 4.8 || p.x > 10.2) p.pos = 'W';
            else p.pos = 'G';
            return;
        }

        // 5v5 dynamic role assignment across full court
        if (distToBaseline <= 4.6) {
            p.pos = (p.x >= 5.2 && p.x <= 9.8) ? 'C' : 'PF';
        } else if (distToBaseline <= 9.5) {
            p.pos = (p.x < 5.2) ? 'SF' : (p.x > 9.8 ? 'SG' : 'PF');
        } else {
            p.pos = (p.x >= 4.5 && p.x <= 10.5) ? 'PG' : (p.x < 4.5 ? 'SF' : 'SG');
        }
    }
}

function yIsFrontRowVolleyball(y) {
    return y > 4.5;
}

function drawSoccerPitch() {
    const cfg = SPORT_FORMATS.soccer;
    const canvas = document.getElementById(cfg.canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const { width: pitchWidth, length: pitchLength } = cfg.dimensions;
    const scaleX = canvas.width / pitchWidth;
    const scaleY = canvas.height / pitchLength;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = 'white';
    ctx.fillStyle = 'white';
    ctx.lineWidth = 2;

    ctx.strokeRect(0, 0, canvas.width, canvas.height);
    ctx.beginPath();
    ctx.moveTo(0, canvas.height / 2);
    ctx.lineTo(canvas.width, canvas.height / 2);
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(canvas.width / 2, canvas.height / 2, 9.15 * scaleX, 0, 2 * Math.PI);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(canvas.width / 2, canvas.height / 2, 3, 0, 2 * Math.PI);
    ctx.fill();

    let boxWidthPx = 40.32 * scaleX;
    let boxDepthPx = 16.5 * scaleY;
    let boxX = (canvas.width - boxWidthPx) / 2;
    let goalWidthPx = 18.32 * scaleX;
    let goalDepthPx = 5.5 * scaleY;
    let goalX = (canvas.width - goalWidthPx) / 2;

    ctx.strokeRect(boxX, 0, boxWidthPx, boxDepthPx);
    ctx.strokeRect(goalX, 0, goalWidthPx, goalDepthPx);
    ctx.beginPath();
    ctx.arc(canvas.width / 2, 11 * scaleY, 3, 0, 2 * Math.PI);
    ctx.fill();

    ctx.strokeRect(boxX, canvas.height - boxDepthPx, boxWidthPx, boxDepthPx);
    ctx.strokeRect(goalX, canvas.height - goalDepthPx, goalWidthPx, goalDepthPx);
    ctx.beginPath();
    ctx.arc(canvas.width / 2, canvas.height - (11 * scaleY), 3, 0, 2 * Math.PI);
    ctx.fill();

    drawPlayerBadges(ctx, canvas, scaleX, scaleY);
}

function drawHardwoodFloor(ctx, width, height) {
    // Reused warm maple/oak hardwood gradient & plank detailing
    const baseGrad = ctx.createLinearGradient(0, 0, width, height);
    baseGrad.addColorStop(0, '#d89b5c');
    baseGrad.addColorStop(0.5, '#c88a4b');
    baseGrad.addColorStop(1, '#b8783b');
    ctx.fillStyle = baseGrad;
    ctx.fillRect(0, 0, width, height);

    const plankCount = 18;
    const plankWidth = width / plankCount;
    for (let i = 0; i < plankCount; i++) {
        const px = i * plankWidth;
        ctx.fillStyle = i % 2 === 0
            ? 'rgba(255, 232, 196, 0.07)'
            : 'rgba(92, 50, 16, 0.06)';
        ctx.fillRect(px, 0, plankWidth, height);

        ctx.strokeStyle = 'rgba(110, 62, 22, 0.22)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(px, 0);
        ctx.lineTo(px, height);
        ctx.stroke();

        const jointY = ((i * 73) % Math.max(120, height - 60)) + 30;
        ctx.beginPath();
        ctx.moveTo(px, jointY);
        ctx.lineTo(px + plankWidth, jointY);
        ctx.stroke();
    }
}

function drawVolleyballCourt() {
    const cfg = SPORT_FORMATS.volleyball;
    const canvas = document.getElementById(cfg.canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const { width: courtWidth, length: courtLength } = cfg.dimensions;
    const scaleX = canvas.width / courtWidth;
    const scaleY = canvas.height / courtLength;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Hardwood court floor
    drawHardwoodFloor(ctx, canvas.width, canvas.height);

    // Highlight active bottom half (0..9m) and 6 zones
    const colWidthPx = 3.0 * scaleX;
    const rowHeightPx = 4.5 * scaleY;

    const zonesGrid = [
        [
            { z: 4, x0: 0, yTop: canvas.height - 9.0 * scaleY, w: colWidthPx, h: rowHeightPx },
            { z: 3, x0: colWidthPx, yTop: canvas.height - 9.0 * scaleY, w: colWidthPx, h: rowHeightPx },
            { z: 2, x0: colWidthPx * 2, yTop: canvas.height - 9.0 * scaleY, w: colWidthPx, h: rowHeightPx }
        ],
        [
            { z: 5, x0: 0, yTop: canvas.height - 4.5 * scaleY, w: colWidthPx, h: rowHeightPx },
            { z: 6, x0: colWidthPx, yTop: canvas.height - 4.5 * scaleY, w: colWidthPx, h: rowHeightPx },
            { z: 1, x0: colWidthPx * 2, yTop: canvas.height - 4.5 * scaleY, w: colWidthPx, h: rowHeightPx }
        ]
    ];

    zonesGrid.flat().forEach((zoneBox, idx) => {
        ctx.save();
        ctx.fillStyle = idx % 2 === 0 ? 'rgba(255, 255, 255, 0.08)' : 'rgba(0, 0, 0, 0.05)';
        ctx.fillRect(zoneBox.x0, zoneBox.yTop, zoneBox.w, zoneBox.h);

        ctx.strokeStyle = 'rgba(255, 255, 255, 0.45)';
        ctx.setLineDash([4, 4]);
        ctx.lineWidth = 1;
        ctx.strokeRect(zoneBox.x0, zoneBox.yTop, zoneBox.w, zoneBox.h);

        ctx.fillStyle = 'rgba(255, 255, 255, 0.65)';
        ctx.font = 'bold 11px Arial';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        ctx.fillText(`Z${zoneBox.z}`, zoneBox.x0 + 6, zoneBox.yTop + 5);
        ctx.restore();
    });

    // Outer boundary (9m x 18m)
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2;
    ctx.strokeRect(0, 0, canvas.width, canvas.height);

    // Attack lines at y = 6.0 and y = 12.0 (3m from center line)
    ctx.save();
    ctx.setLineDash([6, 4]);
    ctx.lineWidth = 1.5;
    [6.0, 12.0].forEach(yMeter => {
        const yPx = canvas.height - (yMeter * scaleY);
        ctx.beginPath();
        ctx.moveTo(0, yPx);
        ctx.lineTo(canvas.width, yPx);
        ctx.stroke();
    });
    ctx.restore();

    // Center line & Net at y = 9.0
    const centerYPx = canvas.height - (9.0 * scaleY);
    ctx.save();
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(0, centerYPx);
    ctx.lineTo(canvas.width, centerYPx);
    ctx.stroke();

    const netHeightPx = 10;
    const netTopPx = centerYPx - (netHeightPx / 2);

    ctx.fillStyle = 'rgba(25, 25, 25, 0.45)';
    ctx.fillRect(0, netTopPx, canvas.width, netHeightPx);

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.55)';
    ctx.lineWidth = 1;
    for (let x = 8; x < canvas.width; x += 8) {
        ctx.beginPath();
        ctx.moveTo(x, netTopPx);
        ctx.lineTo(x, netTopPx + netHeightPx);
        ctx.stroke();
    }

    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(0, netTopPx);
    ctx.lineTo(canvas.width, netTopPx);
    ctx.stroke();

    ctx.strokeStyle = '#111111';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(0, netTopPx + netHeightPx);
    ctx.lineTo(canvas.width, netTopPx + netHeightPx);
    ctx.stroke();

    ctx.fillStyle = '#222222';
    ctx.fillRect(0, centerYPx - 9, 6, 18);
    ctx.fillRect(canvas.width - 6, centerYPx - 9, 6, 18);
    ctx.restore();

    drawPlayerBadges(ctx, canvas, scaleX, scaleY);
}

function drawBasketballCourt() {
    const cfg = SPORT_FORMATS.basketball;
    const canvas = document.getElementById(cfg.canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const { width: courtWidth, length: courtLength } = cfg.dimensions;
    const scaleX = canvas.width / courtWidth;
    const scaleY = canvas.height / courtLength;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Reused volleyball hardwood floor colors (5 zones are tracked logically without rendering zone boundaries)
    drawHardwoodFloor(ctx, canvas.width, canvas.height);

    ctx.strokeStyle = 'white';
    ctx.fillStyle = 'white';
    ctx.lineWidth = 2;

    // Full-court outer boundary (15m x 28m)
    ctx.strokeRect(0, 0, canvas.width, canvas.height);

    // Half-court line & center circle (y = 14.0m)
    const halfYPx = canvas.height / 2;
    const centerXPx = canvas.width / 2;
    ctx.beginPath();
    ctx.moveTo(0, halfYPx);
    ctx.lineTo(canvas.width, halfYPx);
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(centerXPx, halfYPx, 1.8 * scaleX, 0, 2 * Math.PI);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(centerXPx, halfYPx, 3, 0, 2 * Math.PI);
    ctx.fill();

    // Key / Paint area, free-throw circle, 3-point arc, backboard & rim for both ends
    const keyWidthPx = 4.9 * scaleX;
    const keyDepthPx = 5.8 * scaleY;
    const keyXPx = (canvas.width - keyWidthPx) / 2;
    const arcRadiusPx = 6.75 * scaleX;
    const backboardWidthPx = 1.8 * scaleX;

    // Bottom end (baseline at y = 0 -> canvas.height)
    const bottomKeyYPx = canvas.height - keyDepthPx;
    ctx.strokeRect(keyXPx, bottomKeyYPx, keyWidthPx, keyDepthPx);
    ctx.beginPath();
    ctx.arc(centerXPx, bottomKeyYPx, 1.8 * scaleX, 0, 2 * Math.PI);
    ctx.stroke();

    const bottomHoopYPx = canvas.height - (1.575 * scaleY);
    ctx.beginPath();
    ctx.arc(centerXPx, bottomHoopYPx, arcRadiusPx, Math.PI + 0.36, 2 * Math.PI - 0.36);
    ctx.stroke();

    const bottomBoardYPx = canvas.height - (1.2 * scaleY);
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(centerXPx - backboardWidthPx / 2, bottomBoardYPx);
    ctx.lineTo(centerXPx + backboardWidthPx / 2, bottomBoardYPx);
    ctx.stroke();

    ctx.strokeStyle = '#ff5722';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.arc(centerXPx, bottomHoopYPx, 0.23 * scaleX, 0, 2 * Math.PI);
    ctx.stroke();

    // Top end (baseline at y = 28 -> 0)
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2;
    ctx.strokeRect(keyXPx, 0, keyWidthPx, keyDepthPx);
    ctx.beginPath();
    ctx.arc(centerXPx, keyDepthPx, 1.8 * scaleX, 0, 2 * Math.PI);
    ctx.stroke();

    const topHoopYPx = 1.575 * scaleY;
    ctx.beginPath();
    ctx.arc(centerXPx, topHoopYPx, arcRadiusPx, 0.36, Math.PI - 0.36);
    ctx.stroke();

    const topBoardYPx = 1.2 * scaleY;
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(centerXPx - backboardWidthPx / 2, topBoardYPx);
    ctx.lineTo(centerXPx + backboardWidthPx / 2, topBoardYPx);
    ctx.stroke();

    ctx.strokeStyle = '#ff5722';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.arc(centerXPx, topHoopYPx, 0.23 * scaleX, 0, 2 * Math.PI);
    ctx.stroke();

    drawPlayerBadges(ctx, canvas, scaleX, scaleY);
}

function drawPlayerBadges(ctx, canvas, scaleX, scaleY) {
    currentPlayers.forEach(p => {
        const px = p.x * scaleX;
        const py = canvas.height - (p.y * scaleY);

        ctx.beginPath();
        ctx.arc(px, py, 16, 0, 2 * Math.PI);
        ctx.fillStyle = p.pos === 'L' ? '#fff3cd' : 'white';
        ctx.fill();
        ctx.lineWidth = 2;
        ctx.strokeStyle = 'black';
        ctx.stroke();

        ctx.fillStyle = 'black';
        ctx.font = 'bold 10px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(p.pos, px, py);
    });
}

function renderBoard(sportType, formatValue) {
    hideAllBoards();
    activeBoardSport = null;
    activeBoardFormat = null;

    if (sportType === 'soccer') {
        const cfg = SPORT_FORMATS.soccer;
        const match = String(formatValue).match(/7|9|11/);
        const formatNum = match ? parseInt(match[0], 10) : 7;
        const wrapper = document.getElementById(cfg.wrapperId);
        const label = document.getElementById(cfg.labelId);
        if (!wrapper) return;

        wrapper.style.display = 'block';
        if (label) label.textContent = `${formatNum}v${formatNum}`;
        currentPlayers = JSON.parse(JSON.stringify(cfg.formations[formatNum] || cfg.formations[7]));
        activeBoardSport = 'soccer';
        activeBoardFormat = String(formatNum);
        setTimeout(drawSoccerPitch, 30);
        return;
    }

    if (sportType === 'volleyball') {
        const cfg = SPORT_FORMATS.volleyball;
        const normalizedFormat = cfg.options.includes(formatValue) ? formatValue : cfg.defaultFormat;
        const wrapper = document.getElementById(cfg.wrapperId);
        const label = document.getElementById(cfg.labelId);
        if (!wrapper) return;

        wrapper.style.display = 'block';
        if (label) label.textContent = normalizedFormat;
        currentPlayers = JSON.parse(JSON.stringify(cfg.formations[normalizedFormat] || cfg.formations[cfg.defaultFormat]));
        currentPlayers.forEach(p => {
            p.zone = computeVolleyballZone(p.x, p.y);
        });
        activeBoardSport = 'volleyball';
        activeBoardFormat = normalizedFormat;
        setTimeout(drawVolleyballCourt, 30);
        return;
    }

    if (sportType === 'basketball') {
        const cfg = SPORT_FORMATS.basketball;
        const normalizedFormat = cfg.options.includes(formatValue) ? formatValue : cfg.defaultFormat;
        const wrapper = document.getElementById(cfg.wrapperId);
        const label = document.getElementById(cfg.labelId);
        if (!wrapper) return;

        wrapper.style.display = 'block';
        if (label) label.textContent = normalizedFormat;
        currentPlayers = JSON.parse(JSON.stringify(cfg.formations[normalizedFormat] || cfg.formations[cfg.defaultFormat]));
        currentPlayers.forEach(p => {
            p.zone = computeBasketballZone(p.x, p.y);
        });
        activeBoardSport = 'basketball';
        activeBoardFormat = normalizedFormat;
        setTimeout(drawBasketballCourt, 30);
    }
}

function redrawActiveBoard() {
    if (activeBoardSport === 'soccer') {
        drawSoccerPitch();
    } else if (activeBoardSport === 'volleyball') {
        drawVolleyballCourt();
    } else if (activeBoardSport === 'basketball') {
        drawBasketballCourt();
    }
}

function bindCanvasDrag(canvasId, sportType) {
    const canvas = document.getElementById(canvasId);
    const cfg = SPORT_FORMATS[sportType];
    if (!canvas || !cfg) return;

    const { width: worldWidth, length: worldLength } = cfg.dimensions;

    function getWorldCoords(clientX, clientY) {
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / worldWidth;
        const scaleY = canvas.height / worldLength;
        const x = (clientX - rect.left) * (canvas.width / rect.width);
        const y = (clientY - rect.top) * (canvas.height / rect.height);
        return { x, y, scaleX, scaleY };
    }

    function startDrag(clientX, clientY) {
        if (activeBoardSport !== sportType) return;
        const { x, y, scaleX, scaleY } = getWorldCoords(clientX, clientY);

        currentPlayers.forEach(p => {
            const px = p.x * scaleX;
            const py = canvas.height - (p.y * scaleY);
            if (Math.hypot(x - px, y - py) < 22) {
                selectedPlayer = p;
            }
        });
    }

    function moveDrag(clientX, clientY) {
        if (!selectedPlayer || activeBoardSport !== sportType) return;
        const { x, y, scaleX, scaleY } = getWorldCoords(clientX, clientY);

        const minY = sportType === 'basketball' ? 0.5 : 0.4;
        const maxY = sportType === 'volleyball'
            ? 8.6
            : (sportType === 'basketball' ? (worldLength - 0.5) : (worldLength - 3));
        const padX = sportType === 'basketball' ? 0.5 : 0.4;

        selectedPlayer.x = Math.max(padX, Math.min(worldWidth - padX, x / scaleX));
        selectedPlayer.y = Math.max(minY, Math.min(maxY, (canvas.height - y) / scaleY));

        updatePlayerPositionLabel(selectedPlayer, sportType, activeBoardFormat);
        redrawActiveBoard();
    }

    canvas.addEventListener('mousedown', (e) => startDrag(e.clientX, e.clientY));
    canvas.addEventListener('mousemove', (e) => moveDrag(e.clientX, e.clientY));
    window.addEventListener('mouseup', () => { selectedPlayer = null; });

    canvas.addEventListener('touchstart', (e) => {
        if (e.touches.length === 1) {
            startDrag(e.touches[0].clientX, e.touches[0].clientY);
            e.preventDefault();
        }
    }, { passive: false });

    canvas.addEventListener('touchmove', (e) => {
        if (e.touches.length === 1) {
            moveDrag(e.touches[0].clientX, e.touches[0].clientY);
            e.preventDefault();
        }
    }, { passive: false });

    window.addEventListener('touchend', () => { selectedPlayer = null; });
}

function autoSyncBoards() {
    const activeSport = document.querySelector('.sport-selector .btn-format.active');
    const sportName = activeSport ? activeSport.textContent.trim().toLowerCase() : (selectedSportName || 'soccer');
    selectedSportName = sportName;

    const activeFormatBtn = document.querySelector('#matrix-format-container .btn-format.active');
    const formatText = activeFormatBtn ? activeFormatBtn.textContent.trim() : selectedFormatName;

    renderFormatButtons(sportName, formatText);
    renderBoard(sportName, selectedFormatName);
}

function selectSport(button, sportType) {
    document.querySelectorAll('.sport-selector .btn-format').forEach(btn => btn.classList.remove('active'));
    button.classList.add('active');
    selectedSportName = sportType;

    const sportCfg = SPORT_FORMATS[sportType];
    if (sportCfg && !sportCfg.options.includes(selectedFormatName)) {
        selectedFormatName = sportCfg.defaultFormat;
    }

    renderFormatButtons(sportType, selectedFormatName);
    renderBoard(sportType, selectedFormatName);

    fetch('/create-lineup/select-sport', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sport_type: sportType })
    });
}

function selectFormat(clickedButton, formatValue) {
    const parentSelector = clickedButton.closest('.format-selector');
    if (parentSelector) {
        parentSelector.querySelectorAll('.btn-format').forEach(btn => btn.classList.remove('active'));
    }
    clickedButton.classList.add('active');
    selectedFormatName = formatValue;

    renderBoard(selectedSportName, formatValue);

    fetch('/create-lineup/select-format', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ format_type: formatValue })
    });
}

function getActiveBoardPlayers() {
    const cfg = SPORT_FORMATS[selectedSportName];
    if (!cfg || !cfg.wrapperId) return [];
    const wrapper = document.getElementById(cfg.wrapperId);
    if (wrapper && wrapper.style.display !== 'none' && Array.isArray(currentPlayers)) {
        return currentPlayers;
    }
    return [];
}

async function runTacticalPrompt() {
    if (!selectedFormatName) return alert("Please choose a lineup layout format.");
    const runButton = document.getElementById('execute-btn');
    const spinner = document.getElementById('loading-spinner');
    const responseAnchor = document.getElementById('response-anchor');
    const boardPositionsCell = document.getElementById('board-positions-cell');

    runButton.disabled = true;
    spinner.style.display = "block";

    const boardPlayersPayload = getActiveBoardPlayers();

    try {
        const response = await fetch('/create-lineup/generate-tactics', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                sport: selectedSportName,
                format_type: selectedFormatName,
                board_players: boardPlayersPayload
            })
        });
        const data = await response.json();
        spinner.style.display = "none";
        runButton.disabled = false;

        if (data.status === 'success') {
            responseAnchor.innerHTML = data.html_payload;
            if (boardPositionsCell) {
                boardPositionsCell.innerHTML = data.board_payload || '<span style="color: #999; font-style: italic; font-weight: normal;">No active board positions found.</span>';
            }
        } else {
            responseAnchor.innerHTML = `<span style="color:var(--danger-color); font-weight:bold;">Error: ${data.message}</span>`;
        }
    } catch (err) {
        spinner.style.display = "none";
        runButton.disabled = false;
        responseAnchor.innerHTML = '<span style="color:var(--danger-color); font-weight:bold;">Network pipeline failure.</span>';
    }
}

function clearBlueprintFrame() {
    fetch('/create-lineup/clear-blueprint', { method: 'POST' })
    .then(res => res.json())
    .then(data => {
        if (data.status === 'success') {
            document.getElementById('response-anchor').innerHTML = '<span style="color:#999; font-style: italic;">No configuration structure generated yet.</span>';
            const cell = document.getElementById('board-positions-cell');
            if (cell) cell.innerHTML = '<span style="color: #999; font-style: italic; font-weight: normal;">No board positions extracted yet.</span>';
        }
    });
}

document.addEventListener('DOMContentLoaded', () => {
    bindCanvasDrag('soccerPitchCanvas', 'soccer');
    bindCanvasDrag('volleyballCourtCanvas', 'volleyball');
    bindCanvasDrag('basketballCourtCanvas', 'basketball');
    autoSyncBoards();
});

window.addEventListener('load', autoSyncBoards);
setTimeout(autoSyncBoards, 50);
</script>
</body> 
</html>"""
ANALYTICS_PAGE_HTML = """<!DOCTYPE html> 
<html lang="en"> 
<head> 
<meta charset="UTF-8"> 
<meta name="viewport" content="width=device-width, initial-scale=1.0"> 
<title>Analytics & Reports</title> 
""" + SHARED_CSS + """ 
<style> 
 .section-box { background: #fff; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px; margin-bottom: 25px; } 
 .metric-banner { background: #f1f3f5; border: 1px solid #ccc; padding: 20px; border-radius: 8px; margin-bottom: 25px; } 
 .metric-banner button { width: 100%; margin-top: 10px; } 
 @media(min-width: 768px) {
    .metric-banner { display: flex; align-items: center; justify-content: space-between; } 
    .metric-banner button { width: auto; margin-top: 0; } 
 } 
 .badge-alert { background: var(--danger-color); color: white; padding: 2px 5px; font-size: 10px; font-weight: bold; border-radius: 3px; } 
</style> 
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.27.0/plotly.min.js"></script>
</head> 
<body> 
""" + SHARED_NAV + """
<div class="container"> 
    <h2>Analytics & Unified Data Reporting</h2> 
    
    <div class="metric-banner"> 
        <div> 
            <strong>Semantic Player-to-Position Evaluation System</strong> 
            <p style="margin:5px 0 0 0; font-size:12px; color:#666;">Calculates Cosine Similarity matrices and builds fast player word clouds.</p> 
        </div> 
        <div class="btn-group" style="margin: 0;"> 
            <form action="/analytics/compute-metrics" method="POST" style="display:inline-block; margin:0;"> 
                <button type="submit" style="background: #2e7d32; color:white;">Initiate Analytics Processing</button> 
            </form> 
            <form action="/analytics/clear-metrics" method="POST" style="display:inline-block; margin:0;"> 
                <button type="submit" style="background: var(--danger-color); color:white;">Clear Frame</button> 
            </form> 
        </div> 
    </div> 
    
    {% if position_rankings_json %}
    <script src="https://cdn.jsdelivr.net/npm/html2canvas@1.4.1/dist/html2canvas.min.js"></script>
    
    <div class="section-box" style="border-left: 4px solid #10b981; background: #ffffff; border-radius: 12px; padding: 24px; box-shadow: 0 4px 16px rgba(0,0,0,0.05);">
        <!-- Header & Slide Controls -->
        <div style="display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 18px;">
            <div>
                <h3 style="color: #0f172a; margin: 0; font-size: 1.35rem;">🎯 Position Fit Intelligence Deck</h3>
                <p style="color: #64748b; margin: 4px 0 0 0; font-size: 0.9rem;">Sorted by Confidence Score per position • Filter candidates or export position slides</p>
            </div>
            <div style="display: flex; gap: 8px; align-items: center;">
                <button onclick="prevSlide()" style="padding: 8px 14px; border-radius: 8px; border: 1px solid #cbd5e1; background: #f8fafc; cursor: pointer; font-weight: 600;">◀ Prev</button>
                <span id="slide-counter" style="font-size: 0.9rem; font-weight: 600; color: #334155; min-width: 70px; text-align: center;">1 / 1</span>
                <button onclick="nextSlide()" style="padding: 8px 14px; border-radius: 8px; border: 1px solid #cbd5e1; background: #f8fafc; cursor: pointer; font-weight: 600;">Next ▶</button>
                <button onclick="exportCurrentSlide()" style="padding: 8px 16px; border-radius: 8px; border: none; background: #10b981; color: white; cursor: pointer; font-weight: 600;">📸 Export Position Slide</button>
            </div>
        </div>
    
        <!-- Position Pill Selector -->
        <div id="position-pills" style="display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 18px;"></div>
    
        <!-- Filter Bar -->
        <div style="display: flex; flex-wrap: wrap; gap: 14px; align-items: center; background: #f8fafc; padding: 12px 16px; border-radius: 10px; margin-bottom: 20px; border: 1px solid #e2e8f0;">
            <input type="text" id="player-search" placeholder="🔍 Filter by Player (e.g., Player 20)..." oninput="renderSlide()"
                   style="flex: 1; min-width: 200px; padding: 8px 12px; border-radius: 8px; border: 1px solid #cbd5e1; font-size: 0.92rem;">
            
            <div style="display: flex; align-items: center; gap: 10px;">
                <label for="min-score-slider" style="font-size: 0.9rem; font-weight: 600; color: #334155;">Min Confidence:</label>
                <input type="range" id="min-score-slider" min="0.50" max="0.90" step="0.01" value="0.65" oninput="updateScoreLabel(); renderSlide();">
                <span id="min-score-val" style="font-weight: 700; color: #0f766e; min-width: 45px;">0.65</span>
            </div>
    
            <label style="font-size: 0.9rem; color: #334155; display: flex; align-items: center; gap: 6px; cursor: pointer;">
                <input type="checkbox" id="top-n-toggle" onchange="renderSlide()"> Top 10 Only
            </label>
        </div>
    
        <!-- Exportable Position Slide Canvas -->
        <div id="position-slide-card" style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: #f8fafc; border-radius: 14px; padding: 28px; box-shadow: 0 10px 25px rgba(15, 23, 42, 0.15);">
            <div style="display: flex; justify-content: space-between; align-items: flex-end; border-bottom: 1px solid rgba(255,255,255,0.12); padding-bottom: 16px; margin-bottom: 20px;">
                <div>
                    <span style="text-transform: uppercase; letter-spacing: 1.5px; font-size: 0.75rem; color: #38bdf8; font-weight: 700;">Position Depth Chart</span>
                    <h2 id="slide-position-title" style="margin: 4px 0 0 0; font-size: 1.65rem; color: #ffffff;">Position Name</h2>
                </div>
                <div style="text-align: right;">
                    <span id="slide-candidate-count" style="background: rgba(56, 189, 248, 0.15); color: #38bdf8; padding: 6px 12px; border-radius: 20px; font-size: 0.85rem; font-weight: 600;">0 Candidates</span>
                </div>
            </div>
    
            <!-- Ranked Candidate Bars -->
            <div id="slide-rankings-list" style="display: flex; flex-direction: column; gap: 10px; max-height: 540px; overflow-y: auto; padding-right: 6px;"></div>
        </div>
    </div>
    
    <script>
        const positionRankings = {{ position_rankings_json|safe }};
        const positions = Object.keys(positionRankings);
        let currentSlideIdx = 0;
    
        function updateScoreLabel() {
            document.getElementById('min-score-val').textContent = parseFloat(document.getElementById('min-score-slider').value).toFixed(2);
        }
    
        function renderPills() {
            const pillContainer = document.getElementById('position-pills');
            pillContainer.innerHTML = '';
            positions.forEach((pos, idx) => {
                const btn = document.createElement('button');
                btn.textContent = pos;
                const isActive = idx === currentSlideIdx;
                btn.style.cssText = `
                    padding: 6px 14px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; cursor: pointer; transition: all 0.15s;
                    border: 1px solid ${isActive ? '#0f172a' : '#cbd5e1'};
                    background: ${isActive ? '#0f172a' : '#ffffff'};
                    color: ${isActive ? '#ffffff' : '#334155'};
                `;
                btn.onclick = () => { currentSlideIdx = idx; renderPills(); renderSlide(); };
                pillContainer.appendChild(btn);
            });
        }
    
        function renderSlide() {
            if (!positions.length) return;
            const pos = positions[currentSlideIdx];
            const query = document.getElementById('player-search').value.trim().toLowerCase();
            const minScore = parseFloat(document.getElementById('min-score-slider').value);
            const topOnly = document.getElementById('top-n-toggle').checked;
    
            document.getElementById('slide-position-title').textContent = pos;
            document.getElementById('slide-counter').textContent = `${currentSlideIdx + 1} / ${positions.length}`;
    
            // Ensure sorted descending by confidence score
            let candidates = [...positionRankings[pos]].sort((a, b) => b.score - a.score);
    
            // Apply filters
            candidates = candidates.filter(c =>
                c.score >= minScore &&
                (!query || c.player.toLowerCase().includes(query))
            );
            if (topOnly) candidates = candidates.slice(0, 10);
    
            document.getElementById('slide-candidate-count').textContent = `${candidates.length} Matching Players`;
    
            const listEl = document.getElementById('slide-rankings-list');
            listEl.innerHTML = '';
    
            if (!candidates.length) {
                listEl.innerHTML = `<div style="text-align:center; padding: 32px; color: #94a3b8;">No players match the current filter threshold.</div>`;
                return;
            }
    
            candidates.forEach((item, rankIdx) => {
                const pct = Math.min(100, Math.max(0, (item.score * 100))).toFixed(1);
                const tierColor = item.score >= 0.80 ? '#10b981' : (item.score >= 0.75 ? '#38bdf8' : '#94a3b8');
                const row = document.createElement('div');
                row.style.cssText = `
                    display: grid; grid-template-columns: 42px 160px 1fr 75px; align-items: center; gap: 12px;
                    background: rgba(255, 255, 255, 0.05); padding: 10px 14px; border-radius: 10px;
                    border-left: 4px solid ${tierColor};
                `;
                row.innerHTML = `
                    <span style="font-weight: 800; color: ${tierColor}; font-size: 0.95rem;">#${rankIdx + 1}</span>
                    <span style="font-weight: 600; color: #f1f5f9; font-size: 0.95rem;">${item.player}</span>
                    <div style="background: rgba(255,255,255,0.1); height: 10px; border-radius: 999px; overflow: hidden;">
                        <div style="width: ${pct}%; height: 100%; background: ${tierColor}; border-radius: 999px;"></div>
                    </div>
                    <span style="font-family: monospace; font-weight: 700; font-size: 0.95rem; text-align: right; color: #ffffff;">${item.score.toFixed(4)}</span>
                `;
                listEl.appendChild(row);
            });
        }
    
        function prevSlide() {
            currentSlideIdx = (currentSlideIdx - 1 + positions.length) % positions.length;
            renderPills();
            renderSlide();
        }
    
        function nextSlide() {
            currentSlideIdx = (currentSlideIdx + 1) % positions.length;
            renderPills();
            renderSlide();
        }
    
        function exportCurrentSlide() {
            const card = document.getElementById('position-slide-card');
            const posName = positions[currentSlideIdx].replace(/[^a-z0-9]/gi, '_').toLowerCase();
            html2canvas(card, { scale: 2, backgroundColor: '#0f172a' }).then(canvas => {
                const link = document.createElement('a');
                link.download = `position_slide_${posName}.png`;
                link.href = canvas.toDataURL('image/png');
                link.click();
            });
        }
    
        renderPills();
        renderSlide();
    </script>
    {% endif %}
        
    {% if barchart_data_json %} 
    <div class="section-box" style="border-left: 4px solid #17a2b8;"> 
        <h3 style="color:#17a2b8;">📈 Top 3 Candidate Comparisons</h3> 
        <div class="responsive-grid" id="bar-charts-container"></div> 
        <script> 
            const barchartData = {{ barchart_data_json|safe }}; 
            const container = document.getElementById('bar-charts-container'); 
            barchartData.forEach((data, index) => {
                const card = document.createElement('div'); 
                card.className = 'plot-card'; 
                card.innerHTML = ` 
                    <h4>Top Fits: ${data.position}</h4> 
                    <div class="chart-container"> 
                        <canvas id="chart-${index}"></canvas> 
                    </div> 
                `; 
                container.appendChild(card); 
                
                const ctx = document.getElementById(`chart-${index}`).getContext('2d'); 
                new Chart(ctx, {
                    type: 'bar', 
                    data: {
                        labels: data.players, 
                        datasets: [{
                            label: 'Confidence Score', 
                            data: data.scores, 
                            backgroundColor: '#17a2b8', 
                            borderColor: '#117a8b', 
                            borderWidth: 1 
                        }] 
                    }, 
                    options: {
                        responsive: true, 
                        maintainAspectRatio: false, 
                        scales: { y: { beginAtZero: true, max: 1.0 } }, 
                        plugins: { legend: { display: false } } 
                    } 
                }); 
            }); 
        </script> 
    </div> 
    {% endif %} 
     

    {% if sankey_json %}
    <div class="section-box" style="border-left: 4px solid #17a2b8; height: auto; overflow: visible; margin-bottom: 24px;">
        <h3 style="color:#17a2b8;">🌊 Player-to-Position Flow Sankey</h3>
        <div id="sankey-chart-container" style="width: 100%; height: auto; min-height: 950px;"></div>
        <script>
            const sankeyData = {{ sankey_json|safe }};
            Plotly.newPlot('sankey-chart-container', sankeyData.data, sankeyData.layout, {responsive: true});
        </script>
    </div>
    {% endif %}

    
    {% if wordcloud_data %} 
    <div class="section-box" style="border-left: 4px solid #9467bd;"> 
        <h3 style="color:#9467bd;">📊 High-Speed Data Visualizations</h3> 
        <div class="responsive-grid"> 
            {% for wc in wordcloud_data %} 
            <div class="plot-card"> 
                <h4>Word Cloud for {{ wc.player }}</h4> 
                <img src="data:image/png;base64,{{ wc.img }}" alt="Word Cloud for {{ wc.player }}"> 
            </div> 
            {% endfor %} 
        </div> 
    </div> 
    {% endif %} 
    
    <div class="section-box" style="border-left: 4px solid var(--primary-color);"> 
        <h3>1. Live Voice Stream Ingestion (Raw Logs)</h3> 
        <div class="table-wrap"> 
            {% if raw_table %} 
                {{ raw_table|safe }} 
            {% else %} 
                <span style="color:#999; font-style:italic;">No active voice capture rows cached.</span> 
            {% endif %} 
        </div> 
    </div> 
    
    <div class="section-box" style="border-left: 4px solid var(--purple-color);"> 
        <h3>2. Exploded Roster Logs {% if was_overridden %}<span class="badge-alert">Testing Override Active</span>{% endif %}</h3> 
        <div class="table-wrap"> 
            {% if processed_table %} 
                {{ processed_table|safe }} 
            {% else %} 
                <span style="color:#999; font-style:italic;">No entity layers parsed out yet.</span> 
            {% endif %} 
        </div> 
    </div> 
    
    <div class="section-box" style="border-left: 4px solid var(--success-color);"> 
        <h3>3. Ingested External Metrics Spreadsheet</h3> 
        <div class="table-wrap"> 
            {% if uploaded_table %} 
                {{ uploaded_table|safe }} 
            {% else %} 
                <span style="color:#999; font-style:italic;">No metrics spreadsheets ingested yet.</span> 
            {% endif %} 
        </div> 
    </div> 
    
    <div class="section-box" style="border-left: 4px solid #ffc107;"> 
        <h3>4. Generated Tactical Blueprint Frame</h3> 
        <div class="table-wrap"> 
            {% if blueprint_table %} 
                {{ blueprint_table|safe }} 
            {% else %} 
                <span style="color:#999; font-style:italic;">No tactical lineup configurations compiled yet.</span> 
            {% endif %} 
        </div> 
    </div> 
</div> 
</body> 
</html>"""


# =====================================================================
# # SECTION 2: UTILITIES & VECTOR EMBEDDINGS ENGINES                        #
# =====================================================================

def get_mistral_embeddings(texts):
    if not texts:
        return []
    url = "https://api.mistral.ai/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    # Ensure every item is a non-empty string
    cleaned_texts = [str(t).strip() if pd.notna(t) and str(t).strip() else " " for t in texts]
    payload = {
        "model": "mistral-embed",
        "input": cleaned_texts
    }
    for attempt in range(3):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json().get("data", [])
            # Sort by index to guarantee exact row alignment
            data_sorted = sorted(data, key=lambda x: x["index"])
            return [item["embedding"] for item in data_sorted]
        except Exception as e:
            print(f"Mistral batch embedding attempt {attempt + 1} failed: {e}")
            time.sleep(1 + attempt)
    return [[0.0] * 1024 for _ in cleaned_texts]




def fix_misspelled_position_header(df):
    target = "position"
    best_col = None
    max_matches = 0
    for col in df.columns:
        col_str = str(col).lower().strip()
        if col_str == target:
            return df
        matches = sum(1 for char in target if char in col_str)
        if matches > max_matches and len(col_str) >= 4:
            max_matches = matches
            best_col = col
    if best_col is not None and max_matches >= 5:
        df.rename(columns={best_col: "Position"}, inplace=True)
    return df


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
