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

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "soccer-grade-secret-automation-key")

# Cache & DB Configuration
app.config["CACHE_TYPE"] = "FileSystemCache"
app.config["CACHE_DIR"] = os.path.join(app.instance_path, "flask_cache")
app.config["CACHE_DEFAULT_TIMEOUT"] = 3600
cache = Cache(app)
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE='Lax')
db_url = os.environ.get("DATABASE_URL")
if db_url and db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = db_url
db = SQLAlchemy(app)

API_KEY = os.environ.get("MISTRAL_API_KEY")
client = Mistral(api_key=API_KEY) if API_KEY else None

# --- User Model ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(50))
    last_name = db.Column(db.String(50))
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(256), nullable=False)

with app.app_context():
    db.create_all()

# --- Auth Routes ---
import re # Make sure this is at the top of your file with the other imports if not already there

def is_valid_email(email):
    """Basic regex to check if the string looks like an email address."""
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)

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
         
    # Optional: Check if email already exists to prevent duplicate errors
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
    """Updates the user's password without deleting their account."""
    email = request.form.get("email")
    new_password = request.form.get("new_password")
    
    # Optional: Basic validation
    if not email or not new_password:
        return "Email and new password are required. <a href='/login'>Try again</a>", 400
        
    user = User.query.filter_by(email=email).first()
    
    if user:
        # Generate a new hash for the new password and update the user record
        hashed_pw = generate_password_hash(new_password)
        user.password = hashed_pw
        
        # Commit the changes to the database
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

@app.route("/soccer-grade/split-dataframe", methods=["POST"])
def split_dataframe():
    if "user_id" not in session: 
        return redirect(url_for("login"))
    try:
        req_data = request.get_json()
        if not req_data or 'data' not in req_data:
            return jsonify({"status": "error", "message": "No data structure provided"}), 400
        
        raw_rows = req_data['data']
        if not raw_rows:
            return jsonify({"status": "success", "processed_data": []})
        
        df_raw = pd.DataFrame(raw_rows)
        df_raw.columns = ['Timestamp', 'Transcript']
        cache.set('cached_raw', df_raw.to_dict(orient='records'))
        
        df_processed = df_raw.copy()
        df_processed['Transcript'] = df_processed['Transcript'].apply(split_comments_smart)
        df_exploded = df_processed.explode('Transcript').reset_index(drop=True)
        
        cache.set('cached_processed', df_exploded.to_dict(orient='records'))
        cache.delete('processed_was_overridden')
        
        return jsonify({"status": "success", "processed_data": df_exploded.to_dict(orient='records')})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/soccer-grade/process-audio", methods=["POST"])
def process_audio():
    if "user_id" not in session: 
        return redirect(url_for("login"))
    if not client:
        return jsonify({"status": "error", "message": "Mistral API client context check failed. Missing key variable configuration setup parameters."}), 500
    
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
        return jsonify({"status": "error", "message": f"Transcription structural system layer loop failure exception framework block trace: {str(e)}"}), 500
    
    if os.path.exists(temp_filename):
        os.remove(temp_filename)
    
    if not detected_text:
        detected_text = "[Unintelligible audio recorded]"
    
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
    selected_format = cache.get('selected_format') or ''
    blueprint_table = cache.get('cached_blueprint')
    
    return render_template_string(
        LINEUP_PAGE_HTML,
        raw_count=len(raw_session),
        processed_count=len(processed_session),
        uploaded_count=len(uploaded_session),
        selected_format=selected_format,
        blueprint_table=blueprint_table
    )
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
    if not client:
        return jsonify({"status": "error", "message": "Mistral API client missing orchestration credentials."}), 500
    
    req_body = request.get_json()
    format_type = req_body.get("format_type", "11v11")
    cache.set('selected_format', format_type)
    
    prompt_instruction = f"""You are 5 different soccer scouts with various opinions that need to select players for positions on a team. Based on your knowledge, Give me the characteristics and skills required for a player for each position in {format_type} line up. sperate each position and use the characteristics and skills from all the all time great players for that position. Create a data frame with one column being position and the other column being a narrative description of the characteristics and skills for that position. CRITICAL OUTPUT RULE: Return ONLY a valid JSON format list of objects representing this dataframe array. No extra commentary prose text. make sure the columns are labeled 'Position' and 'Description'. Format Example: [{{"Position": "Goalkeeper (GK)", "Description": "Exceptional shot-stopping reflexes..."}} ]"""
    
    
    try:
        response_stream = client.chat.complete(
            model="mistral-large-latest",
            messages=[
                {"role": "system", "content": "You are an advanced soccer tactics architect. Output requested data exclusively as clean JSON arrays."},
                {"role": "user", "content": prompt_instruction}
            ],
            response_format={"type": "json_object"}
        )
        
        raw_content = response_stream.choices[0].message.content.strip()
        parsed_json = json.loads(raw_content)
        
        if isinstance(parsed_json, dict) and len(parsed_json.keys()) == 1:
            key = list(parsed_json.keys())[0]
            parsed_list = parsed_json[key]
        else:
            parsed_list = parsed_json
            
        df_output = pd.DataFrame(parsed_list)
        if len(df_output.columns) >= 2:
            df_output.columns = ['Position', 'Narrative Description Summary']
            
        html_table = df_output.to_html(classes='table', index=False)
        cache.set('cached_blueprint', html_table)
        
        return jsonify({"status": "success", "html_payload": html_table})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Pipeline failure: {str(e)}"}), 500




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
    
    # Retrieve pre-processed chart data from cache
    barchart_data = cache.get('barchart_data')
    wordcloud_data = cache.get('wordcloud_data')
    barchart_data_json = json.dumps(barchart_data) if barchart_data else None
    
    return render_template_string(
        ANALYTICS_PAGE_HTML,
        raw_table=raw_table,
        processed_table=processed_table,
        uploaded_table=uploaded_table,
        blueprint_table=blueprint_table,
        was_overridden=was_overridden,
        similarity_results_table=similarity_results_table,
        barchart_data_json=barchart_data_json,
        wordcloud_data=wordcloud_data
    )
@app.route("/analytics/clear-metrics", methods=["POST"])
def clear_metrics_dataframe():
    if "user_id" not in session: return redirect(url_for("login"))
    cache.delete('similarity_results_html')
    cache.delete('barchart_data')
    cache.delete('wordcloud_data')
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
    
    # ==========================================
    # NEW FIX: Resilient Column Mapping
    # ==========================================
    player_evals_df.columns = [str(c).strip() for c in player_evals_df.columns]
    
    # 1. Safely handle the 'Player' column
    player_col = next((c for c in player_evals_df.columns if c.lower() == 'player'), None)
    if player_col:
        player_evals_df.rename(columns={player_col: 'Player'}, inplace=True)
    else:
        # Fallback: Extract from the best available text column
        target_col = 'Transcript' if 'Transcript' in player_evals_df.columns else (
            'Description' if 'Description' in player_evals_df.columns else player_evals_df.columns[-1]
        )
        player_evals_df['Player'] = player_evals_df[target_col].apply(
            lambda x: re.search(r'(player\s+\d+|\d+)', str(x), re.I).group(1) if re.search(r'(player\s+\d+|\d+)', str(x), re.I) else "Unknown"
        )
        
    # 2. Safely handle the 'Description' column for embeddings
    desc_col = next((c for c in player_evals_df.columns if c.lower() == 'description'), None)
    if desc_col:
        player_evals_df.rename(columns={desc_col: 'Description'}, inplace=True)
    elif 'Transcript' in player_evals_df.columns:
        player_evals_df['Description'] = player_evals_df['Transcript']
    else:
        player_evals_df['Description'] = player_evals_df[player_evals_df.columns[-1]]
    # ==========================================
        
    if uploaded_session:
        ideal_player_df = pd.DataFrame(uploaded_session)
    else:
        try:
            ideal_player_df = pd.read_html(blueprint_html)[0]
        except Exception:
            return "Error parsing system blueprint data frames.", 500
            
    ideal_player_df = fix_misspelled_position_header(ideal_player_df)
    
    if 'Position' not in ideal_player_df.columns:
        # Strip whitespace from all column names first
        ideal_player_df.columns = [str(c).strip() for c in ideal_player_df.columns]
        if 'Position' not in ideal_player_df.columns:
            print(f"Warning: Forcing rename of column '{ideal_player_df.columns[0]}' to 'Position'")
            ideal_player_df.rename(columns={ideal_player_df.columns[0]: "Position"}, inplace=True)
    
    if 'Description' not in ideal_player_df.columns:
        if len(ideal_player_df.columns) >= 2:
            ideal_player_df.rename(columns={ideal_player_df.columns[1]: "Description"}, inplace=True)
            
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
        
        # =======================================================
        # Generate JSON Data for Frontend Chart.js Bar Charts
        # =======================================================
        barchart_data = []
        for position, pos_data in top_players_per_position.items():
            barchart_data.append({
                "position": position,
                "players": list(pos_data.index),
                "scores": [round(float(v), 3) for v in pos_data.values]
            })
        cache.set('barchart_data', barchart_data)
        
        # =======================================================
        # Generate Native WordCloud Images (No Matplotlib Grid)
        # =======================================================
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
            
            # Use native WordCloud to_image() method to bypass matplotlib
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
# # SECTION 1: HTML INTERFACES (FRONTEND UI LAYOUTS)                      #
# =====================================================================
LOGIN_PAGE_HTML = """
<!DOCTYPE html>
<html>
<head><title>Soccer Grader Login</title></head>
<body style="font-family: sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; background-color: #f4f6f9;">
    
    <div style="background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); width: 300px; text-align: center;">
        <h2>Soccer Grader Login</h2>
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
# --- COMMON STYLES AND RESPONSIVE GRID CONFIGURATION ---
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
 /* Responsive Buttons & Containers */
 .btn-group {display: flex; flex-wrap: wrap; gap: 10px; margin: 15px 0; justify-content: center; }
 button, .btn-link {flex: 1 1 calc(50% - 10px); min-width: 140px; padding: 12px 18px; font-size: 14px; font-weight: bold; cursor: pointer; border: none; border-radius: 8px; transition: all 0.2s ease; text-align: center; display: inline-block; }
 @media (min-width: 768px) {body { padding: 30px; } .container { padding: 40px; } button, .btn-link { flex: 0 1 auto; } }
 button:disabled { background: #ccc !important; cursor: not-allowed; transform: none !important; }
 /* Responsive Tables */
 .table-wrap {width: 100%; overflow-x: auto; margin-top: 15px; border: 1px solid #dee2e6; border-radius: 6px; -webkit-overflow-scrolling: touch; }
 table {width: 100%; border-collapse: collapse; background: white; white-space: nowrap; }
 th, td {border: 1px solid #dee2e6; padding: 10px 14px; text-align: left; font-size: 13px; }
 th { background-color: #f8f9fa; position: sticky; top: 0; }
 /* Fully Responsive Grid for Plots */
 .responsive-grid {display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; margin-top: 15px; }
 .plot-card {background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 15px; text-align: center; display: flex; flex-direction: column; }
 .plot-card h4 {margin: 0 0 10px 0; font-size: 14px; color: #333; }
 /* Forces chart.js canvas to resize correctly */
 .chart-container {position: relative; height: 200px; width: 100%; }
 .plot-card img {width: 100%; height: auto; border-radius: 4px; }

 /* Modern Adaptive Navigation */
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
 .top-nav a.nav-home {
     background: #333;
     color: white;
 }
 .top-nav a.nav-home:hover {
     background: #111;
 }
 @media (min-width: 768px) {
     .top-nav {
         justify-content: flex-start;
         padding: 15px 25px;
         gap: 12px;
     }
     .top-nav a {
         font-size: 14px;
         padding: 10px 16px;
     }
 }
</style>
"""

# --- SHARED NAVIGATION HTML ---
SHARED_NAV = """
<nav class="top-nav">
    <a href="/" class="nav-home">🏠 Hub</a>
    <a href="/soccer-grade">🎤 Voice Logger</a>
    <a href="/upload-manager">📁 Data Manager</a>
    <a href="/create-lineup">⚽ Line Up</a>
    <a href="/analytics">📊 Analytics</a>
</nav>
"""

# --- PAGE A: CENTRAL HUB LANDING PAGE ---
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

# --- PAGE B: VOICE RECORDER & ROWS SPLITTER INTERFACE ---
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
 .history-item { padding: 12px; border-bottom: 1px solid #eee; font-size: 13px; } 
 .timestamp { color: #888; font-weight: bold; margin-right: 5px; font-size: 11px; } 
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
                        <li class="history-item"><span class="timestamp">[{{ row.Timestamp }}]</span> {{ row.Transcript }}</li> 
                    {% endfor %} 
                {% else %} 
                    <li class="history-item" id="empty-state" style="color: #aaa; text-align:center;">No raw records yet.</li> 
                {% endif %} 
            </ul> 
        </div> 
        <div class="history-container"> 
            <h3>Processed Lines:</h3> 
            <ul id="processed-list" class="history-list"> 
                {% if has_processed %} 
                    {% for row in processed_records %} 
                        <li class="history-item"><span class="timestamp">[{{ row.Timestamp }}]</span> {{ row.Transcript }}</li> 
                    {% endfor %} 
                {% else %} 
                    <li class="history-item" id="empty-processed" style="color: #aaa; text-align:center;">No split data yet.</li> 
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
    li.innerHTML = `<span class="timestamp">[${timestamp}]</span> ${transcript}`; 
    historyList.insertBefore(li, historyList.firstChild); 
    fetch('/soccer-grade/sync-raw', {method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ data: sessionRecords }) }); 
} 

processBtn.addEventListener('click', async () => {
    if (sessionRecords.length === 0) return; 
    statusDiv.innerText = "Status: Split-processing transcripts..."; 
    try {
        const response = await fetch('/soccer-grade/split-dataframe', {method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ data: sessionRecords.map(r => ({ timestamp: r.Timestamp, transcript: r.Transcript })) }) }); 
        const result = await response.json(); 
        if (result.status === 'success') {
            processedRecords = result.processed_data; 
            processedList.innerHTML = ''; 
            processedRecords.forEach(row => {
                const li = document.createElement('li'); 
                li.className = 'history-item'; 
                li.innerHTML = `<span class="timestamp">[${row.Timestamp}]</span> ${row.Transcript}`; 
                processedList.appendChild(li); 
            }); 
            exportProcessedBtn.disabled = false; 
            statusDiv.style.color = 'green'; 
            statusDiv.innerText = "Status: Split processing finished and cached!"; 
        } 
    } catch (err) {
        statusDiv.style.color = 'red'; 
        statusDiv.innerText = "Server error during row processing."; 
    } 
}); 

async function downloadCSV(records, filename) {
    let csvContent = "Timestamp,Transcript\\n"; 
    records.forEach(row => {
        let text = row.Transcript || ""; 
        let time = row.Timestamp || ""; 
        let cleanTranscript = text.replace(/"/g, '""'); 
        csvContent += `"${time}","${cleanTranscript}"\\n`; 
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

exportBtn.addEventListener('click', () => downloadCSV(sessionRecords, 'voice_history.csv')); 
exportProcessedBtn.addEventListener('click', () => downloadCSV(processedRecords, 'processed_voice_data.csv')); 

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

# --- PAGE C: DATA UPLOAD MANAGER ---
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
    
    <h3 style="color: #0d47a1;">Active Uploaded Spreadsheet Data</h3> 
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
                <input type="file" name="uploaded_csv" accept=".csv" required> 
                <button type="submit" style="background: var(--success-color); color:white;">Upload Position Descriptions</button> 
            </form> 
        </div> 
        <div class="upload-zone override"> 
            <form action="/upload-manager/override-processed" method="POST" enctype="multipart/form-data"> 
                <label style="font-weight:bold; display:block; color:#c0392b;">Sandbox Testing Mock</label> 
                <span style="font-size: 11px; color: #7f8c8d; display:block;">Forces override of Processed & Exploded frame</span> 
                <input type="file" name="mock_processed_csv" accept=".csv" required> 
                <button type="submit" style="background: var(--danger-color); color:white;">Inject Test Override</button> 
            </form> 
        </div> 
    </div> 
</div> 
</body> 
</html>"""

# --- PAGE D: CREATE LINE UP INTERFACE ---
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
    
    <div class="placeholder-box"> 
        <h3 style="color: #856404; margin: 0; font-size: 15px;">Canvas System Status</h3> 
        <p style="color: #666; font-size: 13px; margin: 5px 0 0 0;">Visual drag-and-drop arranger components are actively staging.</p> 
    </div> 
    
    <h3>1. Select Roster Matrix Format</h3> 
    <div class="format-selector"> 
        <button type="button" id="btn-7v7" class="btn-format {% if selected_format == '7v7' %}active{% endif %}" onclick="selectFormat(this, '7v7')">7v7</button> 
        <button type="button" id="btn-9v9" class="btn-format {% if selected_format == '9v9' %}active{% endif %}" onclick="selectFormat(this, '9v9')">9v9</button> 
        <button type="button" id="btn-11v11" class="btn-format {% if selected_format == '11v11' %}active{% endif %}" onclick="selectFormat(this, '11v11')">11v11</button> 
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
let selectedFormatName = "{{ selected_format|safe }}"; 

function selectFormat(clickedButton, formatValue) {
    const buttons = document.querySelectorAll('.btn-format'); 
    buttons.forEach(btn => btn.classList.remove('active')); 
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
        const response = await fetch('/create-lineup/generate-tactics', {method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ format_type: selectedFormatName }) }); 
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

# --- PAGE E: ANALYTICS & REPORTING INTERFACE ---
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
    
    {% if barchart_data_json %} 
    <div class="section-box" style="border-left: 4px solid #17a2b8;"> 
        <h3 style="color:#17a2b8;">📈 Top 3 Candidate Comparisons</h3> 
        <div class="responsive-grid" id="bar-charts-container"></div> 
        <script> 
            const barchartData = {{ barchart_data_json|safe }}; 
            const container = document.getElementById('bar-charts-container'); 
            barchartData.forEach((data, index) => {
                // Create Card structure 
                const card = document.createElement('div'); 
                card.className = 'plot-card'; 
                card.innerHTML = ` 
                    <h4>Top Fits: ${data.position}</h4> 
                    <div class="chart-container"> 
                        <canvas id="chart-${index}"></canvas> 
                    </div> 
                `; 
                container.appendChild(card); 
                
                // Render interactive chart 
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
# --- PASTE ALL YOUR UTILITY FUNCTIONS (split_comments_smart, etc.) HERE ---

# =====================================================================
# # SECTION 2: UTILITIES & VECTOR EMBEDDINGS ENGINES                       #
# =====================================================================
def split_comments_smart(text):
    pattern = (
        r"[.,;\s]+"
        r"(?<!\bto\s)(?<!\bfrom\s)(?<!\bwith\s)(?<!\bfor\s)(?<!\bby\s)"
        r"(?<!\bon\s)(?<!\bof\s)(?<!\band\s)(?<!\bor\s)(?<!\bat\s)"
        r"(?<!\bnumber\s)(?<!\bplayer\s)"
        r"(?=\b(?:number\s+\d+|player\s+\d+|\d+)\b)"
    )
    segments = re.split(pattern, text, flags=re.IGNORECASE)
    return [seg.strip() for seg in segments if seg.strip()]

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
