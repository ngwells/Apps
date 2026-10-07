import datetime, io, json, math, os, re, base64
from flask import Flask, jsonify, request, render_template_string, session, redirect, url_for
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
import plotly.express as px
import gc


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
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    first_name = db.Column(db.String(50))
    last_name = db.Column(db.String(50))
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(256), nullable=False)

with app.app_context():
    db.create_all()

# --- Auth Routes ---
def is_valid_email(email):
    """Basic regex to check if the string looks like an email address."""
    return bool(re.match(r"[^@]+@[^@]+\.[^@]+", email))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        if not is_valid_email(email):
            return "Please enter a valid email address. <a href='/login'>Try again</a>", 400
            
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, request.form.get("password")):
            session["user_id"] = user.id
            return redirect(url_for("index"))
            
        return "Invalid credentials. <a href='/login'>Try again</a>"
    return render_template_string(LOGIN_PAGE_HTML)

@app.route("/register", methods=["POST"])
def register():
    email = request.form.get("email")
    if not is_valid_email(email):
         return "Invalid email format. Please go back and try again.", 400
         
    if User.query.filter_by(email=email).first():
         return "Email already registered. <a href='/login'>Go to login</a>", 400

    hashed_pw = generate_password_hash(request.form.get("password"))
    new_user = User(
        first_name=request.form.get("first_name"),
        last_name=request.form.get("last_name"),
        email=email,
        password=hashed_pw
    )
    db.session.add(new_user)
    db.session.commit()
    return redirect(url_for("login"))

@app.route("/reset-password", methods=["POST"])
def reset_password():
    email = request.form.get("email")
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
    return render_template_string(LANDING_PAGE_HTML)

@app.route("/system/reset-session", methods=["POST"])
def reset_session():
    if "user_id" not in session: 
        return redirect(url_for("login"))
    session.clear(); cache.clear()
    return jsonify({"status": "cleared"})

@app.route("/soccer-grade")
def soccer_grade():
    if "user_id" not in session: 
        return redirect(url_for("login"))
    raw_records = cache.get('cached_raw') or []
    processed_records = cache.get('cached_processed') or []
    return render_template_string(
        SOCCER_INTERFACE_HTML,
        has_raw=len(raw_records) > 0,
        has_processed=len(processed_records) > 0,
        raw_records=raw_records,
        processed_records=processed_records,
        raw_records_json=json.dumps(raw_records),
        processed_records_json=json.dumps(processed_records)
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

import random

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
        
        # Format all transcripts into a single structured payload for 1 single API call
        transcript_batch_str = "\n".join([f"[{row['Timestamp']}] {row['Transcript']}" for _, row in df_raw.iterrows()])
        
        prompt = f"""
        Analyze the following batch of timestamped transcript texts, break each down into natural segments/sentences, and for each segment:
        1. Extract the raw segment text, preserving its timestamp context.
           - the segment should have a subject (e.g., "Player 17", "Player 3", "Number 23", "45", "D12", "Player C86", "Number P456")
        2. Assign a score from 100 to 0 based on how well it evaluates a specific individual player's trait/characteristic:
           - Score 51 to 100 by integers: Directly evaluates an individual player.
           - Lower scores / 0: General instructions, noise, or multi-player sequences.
           
        Return a JSON object with a single key "segments" containing a list of objects, each with "text" (string) and "score" (integer).
           
        Batch Transcripts:
        {transcript_batch_str}
        """
        
        chosen_model = random.choice(['gemini-3.5-flash-lite', 'gemini-3.1-flash-lite'])
        
        response = client2.models.generate_content(
            model=chosen_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )
        
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
            default_ts = raw_rows[0].get('Timestamp', '') if raw_rows else ''
            for seg in parsed_segments:
                # Handle both dict items and object attributes safely
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
        
        # Fallback if model parsing returns empty
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

@app.route("/create-lineup/select-sport", methods=["POST"])
def select_sport_sync():
    if "user_id" not in session:
        return redirect(url_for("login"))
    req_body = request.get_json()
    if req_body and 'sport_type' in req_body:
        cache.set('selected_sport', req_body['sport_type'])
        return jsonify({"status": "sport_cached"})

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

@app.route("/create-lineup/generate-tactics", methods=["POST"])
def generate_tactics():
    if "user_id" not in session:
        return redirect(url_for("login"))
    if not API_KEY:
        return jsonify({"status": "error", "message": "Mistral API key missing."}), 500
    
    req_body = request.get_json() or {}
    sport = req_body.get("sport") or cache.get('selected_sport') or "Soccer"
    format_type = req_body.get("format_type") or cache.get('selected_format') or "11v11"
    
    cache.set('selected_sport', sport)
    cache.set('selected_format', format_type)
    
    positions_count = format_type[0] if format_type else "standard"
    
    prompt_instruction = f"""Generate a JSON list of exactly {positions_count} positions for a {sport} team playing a {format_type} formation. 
Each item in the list must be an object with two keys: "Position" and "Description". 
For the "Description" field, aggregate 5 different scout perspectives (Tactical Analyst, Elite Coach, Veteran Scout, Sports Scientist, and Data Modeler) into a single, cohesive block of text detailing characteristics, psychological traits, and key technical skills based on all-time great players for that position. 
Output ONLY a valid JSON array."""
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "ministral-3b-2512",
        "messages": [
            {"role": "system", "content": "You are a lead sports data architect combining multi-scout evaluations. Return ONLY a valid JSON array of objects with 'Position' and 'Description' keys. No markdown code blocks, no extra commentary."},
            {"role": "user", "content": prompt_instruction}
        ],
        "temperature": 0.2
    }
    
    import time
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
        
        # Clean markdown code blocks
        if "```" in raw_content:
            raw_content = re.sub(r"^```(?:json)?\s*", "", raw_content)
            raw_content = re.sub(r"\s*```$", "", raw_content)
            raw_content = raw_content.strip()
            
        # Sanitize trailing commas before closing brackets or braces
        raw_content = re.sub(r',\s*([\]}])', r'\1', raw_content)
        
        try:
            parsed_json = json.loads(raw_content)
        except json.JSONDecodeError as jde:
            # Fallback regex extraction and comma repair for multi-line JSON blocks
            match = re.search(r'(\[.*\]|\{.*\})', raw_content, re.DOTALL)
            if match:
                cleaned_match = match.group(1)
                cleaned_match = re.sub(r',\s*([\]}])', r'\1', cleaned_match)
                # Fix missing commas between adjacent closing brace and opening brace/quote
                cleaned_match = re.sub(r'}\s*"', '},"', cleaned_match)
                cleaned_match = re.sub(r'}\s*{', '},{', cleaned_match)
                try:
                    parsed_json = json.loads(cleaned_match)
                except Exception:
                    # Final safety fallback: replace unescaped internal double quotes
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
        
        return jsonify({"status": "success", "html_payload": html_table})
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
        barchart_data_json=barchart_data_json,
        wordcloud_data=wordcloud_data,
        sankey_json=sankey_json
    )

@app.route("/analytics/clear-metrics", methods=["POST"])
def clear_metrics_dataframe():
    if "user_id" not in session: return redirect(url_for("login"))
    cache.delete('similarity_results_html')
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
        player_embeddings = [get_mistral_embeddings(desc) or [0]*1024 for desc in player_evals_df["Description"]]
        ideal_embeddings = [get_mistral_embeddings(desc) or [0]*1024 for desc in ideal_player_df["Description"]]
        
        similarity_matrix = cosine_similarity(player_embeddings, ideal_embeddings)
        similarity_df = pd.DataFrame(similarity_matrix, index=player_evals_df["Player"], columns=ideal_player_df["Position"])
        
        top_players_per_position = {}
        for position in similarity_df.columns:
            top_players = similarity_df[position].sort_values(ascending=False).head(3)
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
        
        for _, row in results_df.iterrows():
            player = row['Player']
            pos = row['Position']
            score = float(row['Confidence Score'])
            
            sources.append(player_indices[player])
            targets.append(position_indices[pos])
            values.append(score * 50)
            
            hex_color = player_colors.get(player, "#17a2b8").lstrip('#')
            rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            link_colors.append(f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.4)")
        
        sankey_fig = go.Figure(go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
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
        
        sankey_fig.update_layout(title_text="Player-to-Position Flow Alignment", font_size=11, height=350)
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
    
    <div style="background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); width: 300px; text-align: center;">
        <h2>Sports Grader Login</h2>
        <form action="/login" method="POST">
            <input type="email" name="email" placeholder="Email" required style="width: 100%; padding: 10px; margin: 5px 0; border: 1px solid #ccc; border-radius: 4px;"><br>
            <input type="password" name="password" placeholder="Password" required style="width: 100%; padding: 10px; margin: 5px 0; border: 1px solid #ccc; border-radius: 4px;"><br>
            <button type="submit" style="width: 100%; padding: 10px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; margin-top: 10px;">Log In</button>
        </form>
        <p style="font-size: 13px; margin-top: 15px;">
            <a href="#" onclick="document.getElementById('reset-modal').style.display='block'" style="color: #dc3545; text-decoration: none;">Forgot Password? (Reset Account)</a>
        </p>
        <p style="font-size: 14px;">Don't have an account? <a href="#" onclick="document.getElementById('reg-modal').style.display='block'">Create Account</a></p>
    </div>

    <div id="reg-modal" style="display:none; position:fixed; top:10%; left:35%; background:white; padding:20px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.2); width: 300px;">
        <h3>Register</h3>
        <form action="/register" method="POST">
            <input type="text" name="first_name" placeholder="First Name" required style="width: 100%; padding: 8px; margin: 5px 0;"><br>
            <input type="text" name="last_name" placeholder="Last Name" required style="width: 100%; padding: 8px; margin: 5px 0;"><br>
            <input type="email" name="email" placeholder="Email Address" required style="width: 100%; padding: 8px; margin: 5px 0;"><br>
            <input type="password" name="password" placeholder="Password" required style="width: 100%; padding: 8px; margin: 5px 0;"><br>
            <button type="submit" style="width: 100%; padding: 10px; background: #28a745; color: white; border: none; border-radius: 4px; cursor: pointer;">Register</button>
        </form>
        <button onclick="document.getElementById('reg-modal').style.display='none'" style="margin-top:10px; background:none; border:none; color:red; cursor:pointer;">Cancel</button>
    </div>

    <div id="reset-modal" style="display:none; position:fixed; top:10%; left:35%; background:white; padding:20px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.2); width: 300px;">
        <h3 style="color: #007bff;">Reset Password</h3>
        <p style="font-size: 12px; color: #666;">Enter your email and a new password to update your account.</p>
        <form action="/reset-password" method="POST">
            <input type="email" name="email" placeholder="Enter your Email" required style="width: 100%; padding: 8px; margin: 5px 0;"><br>
            <input type="password" name="new_password" placeholder="New Password" required style="width: 100%; padding: 8px; margin: 5px 0;"><br>
            <button type="submit" style="width: 100%; padding: 10px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer;">Update Password</button>
        </form>
        <button onclick="document.getElementById('reset-modal').style.display='none'" style="margin-top:10px; background:none; border:none; color:black; cursor:pointer;">Cancel</button>
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
<style>
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
</nav>
"""

LANDING_PAGE_HTML = """<!DOCTYPE html> 
<html lang="en"> 
<head> 
<meta charset="UTF-8"> 
<meta name="viewport" content="width=device-width, initial-scale=1.0"> 
<title>Application Hub</title> 
""" + SHARED_CSS + """ 
<style> 
 body { text-align: center; } 
 .grid {display: grid; grid-template-columns: 1fr; gap: 20px; margin: 30px 0; } 
 @media (min-width: 576px) { .grid { grid-template-columns: repeat(2, 1fr); } } 
 @media (min-width: 992px) { .grid { grid-template-columns: repeat(4, 1fr); } } 
 .card {background: white; padding: 25px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); text-align: left; transition: transform 0.2s, box-shadow 0.2s; text-decoration: none; color: inherit; display: flex; flex-direction: column; justify-content: space-between; border-top: 4px solid var(--primary-color); } 
 .card:hover { transform: translateY(-4px); box-shadow: 0 8px 20px rgba(0,0,0,0.1); } 
 .card h3 { margin-bottom: 10px; color: #222; font-size: 18px; } 
 .card p { color: #555; font-size: 13px; line-height: 1.5; margin-bottom: 20px; } 
 .badge {display: inline-block; background: #e1ecf4; color: #39739d; font-size: 11px; padding: 4px 8px; border-radius: 4px; font-weight: bold; align-self: flex-start; } 
 .system-controls { margin-top: 40px; border-top: 1px solid #ddd; padding-top: 20px; } 
 .btn-reset-all { background: #6c757d; color: white; width: 100%; max-width: 300px; } 
</style> 
</head> 
<body> 
<div class="container"> 
    <h1>Data & Voice Automation Suite</h1> 
    <p style="color:#666;">Select an engine interface workflow layout environment below:</p> 
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
        <button class="btn-reset-all" onclick="clearFullSession()">Reset Environment Roster Cache</button> 
    </div> 
</div> 
<script> 
function clearFullSession() {
    if (confirm("Are you sure you want to completely flush all loaded dataframes and generated files?")) {
        fetch('/system/reset-session', { method: 'POST' }) 
        .then(() => { alert("Cache context cleared successfully."); window.location.reload(); }); 
    } 
} 
</script> 
</body> 
</html>"""

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
 .btn-format { background: white; color: var(--primary-color); border: 2px solid var(--primary-color); padding: 10px 20px; flex: 1; max-width: 150px; } 
 .btn-format.active { background: var(--primary-color); color: white; } 
 .btn-execute { background: var(--success-color); color: white; } 
 .btn-clear-frame { background: var(--danger-color); color: white; } 
 .llm-response-box { background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px; margin-top: 20px; min-height: 100px; } 
 .debug-panel { background: #f1f3f5; border-radius: 8px; padding: 15px; font-size: 12px; margin-top: 30px; } 
 .loader { display: none; text-align: center; font-weight: bold; color: #666; margin: 20px 0; } 
</style> 
</head> 
<body> 
""" + SHARED_NAV + """
<div class="container"> 
    <h2>Line Up Builder Workspace</h2> 
    <p style="color:#666; text-align: center;">Select a configuration grid below to evaluate tactical alignment blueprints via Mistral AI.</p> 
    
    <h3>1. Select Sport</h3>
    <div class="format-selector sport-selector">
        <button type="button" id="btn-soccer" class="btn-format {% if selected_sport == 'soccer' %}active{% endif %}" onclick="selectSport(this, 'soccer')">Soccer</button>
        <button type="button" id="btn-basketball" class="btn-format {% if selected_sport == 'basketball' %}active{% endif %}" onclick="selectSport(this, 'basketball')">Basketball</button>
        <button type="button" id="btn-volleyball" class="btn-format {% if selected_sport == 'volleyball' %}active{% endif %}" onclick="selectSport(this, 'volleyball')">Volleyball</button>
    </div>

    <div id="matrix-format-container">
        {% if selected_sport == 'soccer' %}
            <h3>1. Select Roster Matrix Format</h3>
            <div class="format-selector">
                <button type="button" id="btn-7v7" class="btn-format {% if selected_format == '7v7' %}active{% endif %}" onclick="selectFormat(this, '7v7')">7v7</button>
                <button type="button" id="btn-9v9" class="btn-format {% if selected_format == '9v9' %}active{% endif %}" onclick="selectFormat(this, '9v9')">9v9</button>
                <button type="button" id="btn-11v11" class="btn-format {% if selected_format == '11v11' %}active{% endif %}" onclick="selectFormat(this, '11v11')">11v11</button>
            </div>
        {% elif selected_sport == 'basketball' %}
            <h3>1. Select Roster Matrix Format</h3>
            <div class="format-selector">
                <button type="button" id="btn-5x5" class="btn-format {% if selected_format == '5x5' %}active{% endif %}" onclick="selectFormat(this, '5x5')">5x5</button>
            </div>
        {% elif selected_sport == 'volleyball' %}
            <h3>1. Select Roster Matrix Format</h3>
            <div class="format-selector">
                <button type="button" id="btn-6x6" class="btn-format {% if selected_format == '6x6' %}active{% endif %}" onclick="selectFormat(this, '6x6')">6x6</button>
            </div>
        {% endif %}
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
let selectedSportName = "{{ selected_sport|safe }}" || "soccer";
let selectedFormatName = "{{ selected_format|safe }}"; 

function selectSport(button, sportType) {
    document.querySelectorAll('.sport-selector .btn-format').forEach(btn => {
        btn.classList.remove('active');
    });
    button.classList.add('active');
    selectedSportName = sportType;

    const container = document.getElementById('matrix-format-container');
    container.innerHTML = '';

    if (sportType === 'soccer') {
        container.innerHTML = `
            <h3>1. Select Roster Matrix Format</h3>
            <div class="format-selector">
                <button type="button" id="btn-7v7" class="btn-format" onclick="selectFormat(this, '7v7')">7v7</button>
                <button type="button" id="btn-9v9" class="btn-format" onclick="selectFormat(this, '9v9')">9v9</button>
                <button type="button" id="btn-11v11" class="btn-format" onclick="selectFormat(this, '11v11')">11v11</button>
            </div>
        `;
    } else if (sportType === 'basketball') {
        container.innerHTML = `
            <h3>1. Select Roster Matrix Format</h3>
            <div class="format-selector">
                <button type="button" id="btn-5v5" class="btn-format" onclick="selectFormat(this, '5v5')">5v5</button>
                <button type="button" id="btn-3v3" class="btn-format" onclick="selectFormat(this, '3v3')">3v3</button>
            </div>
        `;
    } else if (sportType === 'volleyball') {
        container.innerHTML = `
            <h3>1. Select Roster Matrix Format</h3>
            <div class="format-selector">
                <button type="button" id="btn-6v6" class="btn-format" onclick="selectFormat(this, '6v6')">6v6</button>
                <button type="button" id="btn-2v2" class="btn-format" onclick="selectFormat(this, '2v2')">2v2</button>
            </div>
        `;
    }

    fetch('/create-lineup/select-sport', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sport_type: sportType })
    });
}

function selectFormat(clickedButton, formatValue) {
    const parentSelector = clickedButton.closest('.format-selector');
    parentSelector.querySelectorAll('.btn-format').forEach(btn => btn.classList.remove('active')); 
    clickedButton.classList.add('active'); 
    selectedFormatName = formatValue; 
    fetch('/create-lineup/select-format', {method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ format_type: formatValue }) }); 
}

async function runTacticalPrompt() {
    if (!selectedFormatName) return alert("Please choose a lineup layout metric variant layout format."); 
    const runButton = document.getElementById('execute-btn'); 
    const spinner = document.getElementById('loading-spinner'); 
    const responseAnchor = document.getElementById('response-anchor'); 
    runButton.disabled = true; 
    spinner.style.display = "block"; 
    
    try {
        const response = await fetch('/create-lineup/generate-tactics', {
            method: 'POST', 
            headers: { 'Content-Type': 'application/json' }, 
            body: JSON.stringify({ sport: selectedSportName, format_type: selectedFormatName }) 
        }); 
        const data = await response.json(); 
        spinner.style.display = "none"; 
        runButton.disabled = false; 
        
        if (data.status === 'success') {
            responseAnchor.innerHTML = data.html_payload; 
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
        if(data.status === 'success') {
            document.getElementById('response-anchor').innerHTML = '<span style="color:#999; font-style: italic;">No configuration structure generated yet.</span>'; 
        } 
    }); 
} 
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
    
    {% if similarity_results_table %} 
    <div class="section-box" style="border-left: 4px solid #2e7d32; background: #fbfdfb;"> 
        <h3 style="color:#2e7d32;">🎯 Top Position Fit Recommendations</h3> 
        <div class="table-wrap"> 
            {{ similarity_results_table|safe }} 
        </div> 
    </div> 
    {% endif %} 

    {% if sankey_json %} 
    <div class="section-box" style="border-left: 4px solid #17a2b8;"> 
        <h3 style="color:#17a2b8;">🌊 Player-to-Position Flow Sankey</h3> 
        <div id="sankey-chart-container" style="width:100%; height:350px;"></div> 
        <script> 
            const sankeyData = {{ sankey_json|safe }}; 
            Plotly.newPlot('sankey-chart-container', sankeyData.data, sankeyData.layout, {responsive: true}); 
        </script> 
    </div> 
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
def get_mistral_embeddings(text):
    url = "https://api.mistral.ai/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    clean_text = str(text)[:1000] if text else "Empty text node frame"
    payload = {
        "model": "mistral-embed",
        "input": [clean_text]
    }
    import time
    for attempt in range(3):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=15)
            if response.status_code == 200:
                return response.json()["data"][0]["embedding"]
            elif response.status_code in [520, 502, 503, 504]:
                time.sleep(1 + attempt)
                continue
            else:
                return None
        except Exception:
            time.sleep(1 + attempt)
            continue
    return None

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
