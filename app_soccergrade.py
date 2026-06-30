import datetime
import json
import os
import re
from flask import Flask, jsonify, request, render_template_string, session
from mistralai.client import Mistral
import pandas as pd

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "soccer-grade-secret-automation-key")

# Initialize Mistral Client from environment variable
API_KEY = os.environ.get("MISTRAL_API_KEY")
client = Mistral(api_key=API_KEY) if API_KEY else None

# ===================================================================== #
# SECTION 1: HTML INTERFACES (FRONTEND JAVASCRIPT & UI LAYOUTS)        #
# ===================================================================== #

# --- PAGE A: CENTRAL HUB LANDING PAGE ---
LANDING_PAGE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Application Hub</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 1000px; margin: 60px auto; background: #f4f6f9; text-align: center; color: #333; }
        h1 { margin-bottom: 10px; color: #222; }
        .subtitle { color: #666; margin-bottom: 40px; font-size: 16px; font-weight: bold; }
        .grid { display: flex; justify-content: center; gap: 25px; flex-wrap: wrap; padding: 0 20px; }
        .card { background: white; width: 200px; padding: 25px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); text-align: left; transition: transform 0.2s, box-shadow 0.2s; text-decoration: none; color: inherit; display: flex; flex-direction: column; justify-content: space-between; }
        .card:hover { transform: translateY(-5px); box-shadow: 0 8px 20px rgba(0,0,0,0.1); }
        .card h3 { margin-top: 0; color: #007bff; font-size: 18px; }
        .card p { color: #555; font-size: 13px; line-height: 1.5; min-height: 60px; margin-bottom: 15px; }
        .badge { display: inline-block; background: #e1ecf4; color: #39739d; font-size: 11px; padding: 4px 8px; border-radius: 4px; font-weight: bold; align-self: flex-start; }
    </style>
</head>
<body>
    <h1>Data & Voice Automation Suite</h1>
    <p class="subtitle">Click one of the links below to launch an environment:</p>
    <div class="grid">
        <a href="/soccer-grade" class="card">
            <div>
                <h3>1. Voice Logger</h3>
                <p>Voice-to-text logging assistant featuring automated player-by-player row splitting.</p>
            </div>
            <span class="badge">Voice Input</span>
        </a>
        <a href="/upload-manager" class="card">
            <div>
                <h3>2. Data Manager</h3>
                <p>Upload external metrics data or spreadsheets and manage session assets.</p>
            </div>
            <span class="badge">File Processing</span>
        </a>
        <a href="/create-lineup" class="card">
            <div>
                <h3>3. Create Line Up</h3>
                <p>Build and arrange tactical team lineups utilizing active roster data templates.</p>
            </div>
            <span class="badge">Tactics</span>
        </a>
        <a href="/analytics" class="card">
            <div>
                <h3>4. Analytics</h3>
                <p>Review raw files, processed logs, and uploaded metrics dashboards side by side.</p>
            </div>
            <span class="badge">Insights</span>
        </a>
    </div>
</body>
</html>
"""

# --- PAGE B: VOICE RECORDER & ROWS SPLITTER INTERFACE ---
SOCCER_INTERFACE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Audio Voice Logger</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 700px; margin: 40px auto; text-align: center; background: #f9f9f9; }
        .back-nav { text-align: left; margin-bottom: 20px; }
        .back-link { text-decoration: none; color: #007bff; font-weight: bold; font-size: 14px; }
        .container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        button { padding: 12px 24px; font-size: 14px; font-weight: bold; cursor: pointer; border: none; border-radius: 25px; margin: 6px; transition: background 0.3s; }
        .btn-record { background: #dc3545; color: white; }
        .btn-record:hover { background: #bd2130; }
        .btn-stop { background: #28a745; color: white; }
        .btn-stop:hover { background: #218838; }
        .btn-process { background: #17a2b8; color: white; }
        .btn-process:hover { background: #138496; }
        .btn-save { background: #007bff; color: white; }
        .btn-save:hover { background: #0056b3; }
        .btn-save-processed { background: #6f42c1; color: white; }
        .btn-save-processed:hover { background: #5a32a3; }
        button:disabled { background: #ccc; cursor: not-allowed; }
        #status { margin: 20px 0; font-weight: bold; color: #555; }
        .flex-container { display: flex; justify-content: space-between; margin-top: 30px; }
        .history-container { width: 48%; text-align: left; }
        .history-list { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 0; list-style: none; max-height: 300px; overflow-y: auto; }
        .history-item { padding: 10px 12px; border-bottom: 1px solid #eee; font-size: 13px; }
        .history-item:last-child { border-bottom: none; }
        .timestamp { color: #888; font-weight: bold; margin-right: 5px; font-size: 11px; }
    </style>
</head>
<body>
    <div class="back-nav">
        <a href="/" class="back-link">← Back to Hub</a>
    </div>
    <div class="container">
        <h1>Voice Recorder Logger</h1>
        <p>Click "Start Recording" to open your mic, and "Stop & Process" to transcribe.</p>
        <div>
            <button id="start-btn" class="btn-record">Start Recording</button>
            <button id="stop-btn" class="btn-stop" disabled>Stop & Process</button>
        </div>
        <div style="margin-top: 15px; border-top: 1px solid #eee; padding-top: 15px;">
            <button id="process-btn" class="btn-process" disabled>Split & Process Rows</button>
            <button id="export-btn" class="btn-save" disabled>Export Raw CSV</button>
            <button id="export-processed-btn" class="btn-save-processed" disabled>Export Processed CSV</button>
        </div>
        <div id="status">Status: Idle</div>
        <div class="flex-container">
            <div class="history-container">
                <h3>Raw Transcripts:</h3>
                <ul id="history-list" class="history-list">
                    <li class="history-item" id="empty-state" style="color: #aaa; text-align:center;">No raw records yet.</li>
                </ul>
            </div>
            <div class="history-container">
                <h3>Processed Lines:</h3>
                <ul id="processed-list" class="history-list">
                    <li class="history-item" id="empty-processed" style="color: #aaa; text-align:center;">No split data yet.</li>
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
        let sessionRecords = [];
        let processedRecords = [];

        function appendToSessionDOM(timestamp, transcript) {
            const emptyState = document.getElementById('empty-state');
            if (emptyState) emptyState.remove();
            sessionRecords.push({ timestamp, transcript });
            exportBtn.disabled = false;
            processBtn.disabled = false;
            const li = document.createElement('li');
            li.className = 'history-item';
            li.innerHTML = `<span class="timestamp">[${timestamp}]</span> ${transcript}`;
            historyList.insertBefore(li, historyList.firstChild);
        }

        processBtn.addEventListener('click', async () => {
            if (sessionRecords.length === 0) return;
            statusDiv.innerText = "Status: Split-processing transcripts...";
            try {
                const response = await fetch('/soccer-grade/split-dataframe', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ data: sessionRecords })
                });
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
                let text = row.transcript || row.Transcript || "";
                let time = row.timestamp || row.Timestamp || "";
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
</html>
"""

# --- PAGE C: DATA UPLOAD MANAGER (WITH SEPARATE UPLOAD PREVIEW) ---
UPLOAD_PAGE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Upload Manager</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 850px; margin: 40px auto; text-align: center; background: #f9f9f9; }
        .back-nav { text-align: left; margin-bottom: 20px; }
        .back-link { text-decoration: none; color: #007bff; font-weight: bold; font-size: 14px; }
        .container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: left; }
        .data-block { background: #f1f3f5; border-radius: 6px; padding: 15px; margin-bottom: 20px; font-size: 14px; max-height: 200px; overflow-y: auto; }
        .upload-block { background: #e3f2fd; border: 1px solid #90caf9; border-radius: 6px; padding: 15px; margin-bottom: 20px; font-size: 14px; max-height: 250px; overflow-y: auto; }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; background: white; }
        th, td { border: 1px solid #dee2e6; padding: 8px; text-align: left; font-size: 12px; }
        th { background-color: #e9ecef; }
        .upload-zone { border: 2px dashed #007bff; padding: 25px; text-align: center; background: #f8f9fa; border-radius: 6px; margin-top: 15px; }
        input[type="file"] { margin-top: 10px; }
    </style>
</head>
<body>
    <div class="back-nav">
        <a href="/" class="back-link">← Back to Hub</a>
    </div>
    <div class="container">
        <h2>Data Upload Manager</h2>
        <p style="color:#666;">Manage your imported spreadsheet files separately alongside your captured active voice logs.</p>
        
        <h3 style="color: #0d47a1;">Active Uploaded Spreadsheet Data</h3>
        <div class="upload-block">
            {% if uploaded_data_table %}
                {{ uploaded_data_table|safe }}
            {% else %}
                <span style="color:#666; font-style: italic;">No uploaded spreadsheet records currently loaded in active space. Use the dropzone below to load a CSV.</span>
            {% endif %}
        </div>

        <h3>Voice Stream Snapshot (Raw)</h3>
        <div class="data-block">
            {% if raw_data_table %}
                {{ raw_data_table|safe }}
            {% else %}
                <span style="color:#999;">No raw recording logs currently loaded in session memory.</span>
            {% endif %}
        </div>

        <h3>Voice Stream Snapshot (Processed & Exploded)</h3>
        <div class="data-block">
            {% if processed_data_table %}
                {{ processed_data_table|safe }}
            {% else %}
                <span style="color:#999;">No processed/split comment structures currently loaded in session memory.</span>
            {% endif %}
        </div>

        <h3>Upload Spreadsheet Metrics</h3>
        <div class="upload-zone">
            <form action="/upload-manager/submit-file" method="POST" enctype="multipart/form-data">
                <label style="font-weight:bold; display:block;">Select CSV File to import:</label>
                <input type="file" name="uploaded_csv" accept=".csv" required><br><br>
                <button type="submit" style="padding:10px 20px; background:#28a745; color:white; border:none; border-radius:4px; font-weight:bold; cursor:pointer;">Analyze & Ingest File</button>
            </form>
        </div>
    </div>
</body>
</html>
"""

# --- PAGE D: CREATE LINE UP INTERFACE ---
LINEUP_PAGE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Create Line Up</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 850px; margin: 40px auto; text-align: center; background: #f9f9f9; }
        .back-nav { text-align: left; margin-bottom: 20px; }
        .back-link { text-decoration: none; color: #007bff; font-weight: bold; font-size: 14px; }
        .container { background: white; padding: 40px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .placeholder-box { border: 2px dashed #ffc107; background: #fffde7; padding: 30px; border-radius: 6px; margin-bottom: 30px; }
        .debug-panel { background: #f1f3f5; border-radius: 6px; padding: 15px; text-align: left; font-size: 12px; }
    </style>
</head>
<body>
    <div class="back-nav">
        <a href="/" class="back-link">← Back to Hub</a>
    </div>
    <div class="container">
        <h2>Line Up Builder Workspace</h2>
        <div class="placeholder-box">
            <h3 style="color: #856404; margin-top: 0;">Content Coming Soon</h3>
            <p style="color: #666; margin-bottom: 0;">The interactive field arranger and drag-and-drop roster builder tools are currently in development.</p>
        </div>
        
        <div class="debug-panel">
            <h4 style="margin-top:0; color:#495057;">Available Environment Session Context Check:</h4>
            <ul>
                <li><strong>Raw Log Array Rows:</strong> {{ raw_count }} rows accessible</li>
                <li><strong>Processed Log Array Rows:</strong> {{ processed_count }} rows accessible</li>
                <li><strong>Uploaded Asset Records:</strong> {{ uploaded_count }} rows accessible</li>
            </ul>
        </div>
    </div>
</body>
</html>
"""

# --- PAGE E: ANALYTICS & REPORTING INTERFACE ---
ANALYTICS_PAGE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Analytics & Reports</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; text-align: center; background: #f9f9f9; }
        .back-nav { text-align: left; margin-bottom: 20px; }
        .back-link { text-decoration: none; color: #007bff; font-weight: bold; font-size: 14px; }
        .container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: left; }
        .data-grid { display: flex; flex-direction: column; gap: 25px; }
        .section-box { background: #fff; border: 1px solid #dee2e6; border-radius: 6px; padding: 20px; }
        .table-wrap { max-height: 250px; overflow-y: auto; margin-top: 10px; border: 1px solid #eee; }
        table { width: 100%; border-collapse: collapse; background: white; }
        th, td { border: 1px solid #dee2e6; padding: 8px; text-align: left; font-size: 12px; }
        th { background-color: #f8f9fa; position: sticky; top: 0; }
        .empty-text { color: #999; font-style: italic; font-size: 13px; }
    </style>
</head>
<body>
    <div class="back-nav">
        <a href="/" class="back-link">← Back to Hub</a>
    </div>
    <div class="container">
        <h2>Analytics & Unified Data Reporting</h2>
        <p style="color:#666; margin-bottom: 25px;">Cross-reference spreadsheet telemetry sets with processed runtime streaming voice buffers directly below.</p>
        
        <div class="data-grid">
            <div class="section-box" style="border-left: 4px solid #007bff;">
                <h3 style="margin-top:0; color:#007bff;">1. Live Voice Stream Ingestion (Raw Logs)</h3>
                <div class="table-wrap">
                    {% if raw_table %} {{ raw_table|safe }} {% else %} <span class="empty-text">No active voice stream capture cached in memory.</span> {% endif %}
                </div>
            </div>

            <div class="section-box" style="border-left: 4px solid #6f42c1;">
                <h3 style="margin-top:0; color:#6f42c1;">2. Exploded Roster Logs (Smart Split Comment Entities)</h3>
                <div class="table-wrap">
                    {% if processed_table %} {{ processed_table|safe }} {% else %} <span class="empty-text">No exploded entity records broken out yet.</span> {% endif %}
                </div>
            </div>

            <div class="section-box" style="border-left: 4px solid #28a745;">
                <h3 style="margin-top:0; color:#28a745;">3. Ingested External Metrics Spreadsheet</h3>
                <div class="table-wrap">
                    {% if uploaded_table %} {{ uploaded_table|safe }} {% else %} <span class="empty-text">No uploaded analytical metrics spreadsheet file parsed yet.</span> {% endif %}
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""

# ===================================================================== #
# SECTION 2: UTILITIES & TEXT SPLITTING ALGORITHM                       #
# ===================================================================== #

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

# ===================================================================== #
# SECTION 3: APPS CONTROLLER ENGINE & ROUTING                           #
# ===================================================================== #

@app.route("/")
def index():
    return render_template_string(LANDING_PAGE_HTML)

@app.route("/soccer-grade")
def soccer_grade():
    return render_template_string(SOCCER_INTERFACE_HTML)

@app.route("/soccer-grade/split-dataframe", methods=["POST"])
def split_dataframe():
    try:
        req_data = request.get_json()
        if not req_data or 'data' not in req_data:
            return jsonify({"status": "error", "message": "No data structure provided"}), 400
        raw_rows = req_data['data']
        if not raw_rows:
            return jsonify({"status": "success", "processed_data": []})
        
        df_raw = pd.DataFrame(raw_rows)
        df_raw.columns = ['Timestamp', 'Transcript']
        session['cached_raw'] = df_raw.to_dict(orient='records')
        
        df_processed = df_raw.copy()
        df_processed['Transcript'] = df_processed['Transcript'].apply(split_comments_smart)
        df_exploded = df_processed.explode('Transcript').reset_index(drop=True)
        session['cached_processed'] = df_exploded.to_dict(orient='records')
        
        return jsonify({"status": "success", "processed_data": df_exploded.to_dict(orient='records')})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/soccer-grade/process-audio", methods=["POST"])
def process_audio():
    if not client:
        return jsonify({"status": "error", "message": "Mistral API key missing."}), 500
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
        return jsonify({"status": "error", "message": f"Transcription failed: {str(e)}"}), 500
        
    if os.path.exists(temp_filename):
        os.remove(temp_filename)
        
    if not detected_text:
        detected_text = "[Unintelligible audio recorded]"
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return jsonify({"status": "success", "transcript": detected_text, "timestamp": current_time})

@app.route("/upload-manager")
def upload_manager():
    raw_session = session.get('cached_raw', [])
    processed_session = session.get('cached_processed', [])
    uploaded_session = session.get('cached_uploaded', [])
    
    raw_data_table = pd.DataFrame(raw_session).to_html(classes='table', index=False) if raw_session else None
    processed_data_table = pd.DataFrame(processed_session).to_html(classes='table', index=False) if processed_session else None
    uploaded_data_table = pd.DataFrame(uploaded_session).to_html(classes='table', index=False) if uploaded_session else None
    
    return render_template_string(UPLOAD_PAGE_HTML, 
                                  raw_data_table=raw_data_table, 
                                  processed_data_table=processed_data_table, 
                                  uploaded_data_table=uploaded_data_table)

@app.route("/upload-manager/submit-file", methods=["POST"])
def submit_uploaded_file():
    if 'uploaded_csv' not in request.files:
        return "No file selected", 400
    file = request.files['uploaded_csv']
    if file.filename == '':
        return "Empty file selection", 400
    try:
        uploaded_df = pd.read_csv(file)
        session['cached_uploaded'] = uploaded_df.to_dict(orient='records')
        return render_template_string("<h3>Ingestion complete!</h3><p>File parsed successfully.</p><script>setTimeout(function(){window.location.href='/upload-manager';}, 1200);</script>")
    except Exception as e:
        return f"Error analyzing data structure: {str(e)}", 500

# --- NEW ROUTE: CREATE LINE UP ---
@app.route("/create-lineup")
def create_lineup():
    raw_session = session.get('cached_raw', [])
    processed_session = session.get('cached_processed', [])
    uploaded_session = session.get('cached_uploaded', [])
    
    return render_template_string(
        LINEUP_PAGE_HTML,
        raw_count=len(raw_session),
        processed_count=len(processed_session),
        uploaded_count=len(uploaded_session)
    )

# --- NEW ROUTE: ANALYTICS ---
@app.route("/analytics")
def analytics():
    raw_session = session.get('cached_raw', [])
    processed_session = session.get('cached_processed', [])
    uploaded_session = session.get('cached_uploaded', [])
    
    raw_table = pd.DataFrame(raw_session).to_html(classes='table', index=False) if raw_session else None
    processed_table = pd.DataFrame(processed_session).to_html(classes='table', index=False) if processed_session else None
    uploaded_table = pd.DataFrame(uploaded_session).to_html(classes='table', index=False) if uploaded_session else None
    
    return render_template_string(
        ANALYTICS_PAGE_HTML,
        raw_table=raw_table,
        processed_table=processed_table,
        uploaded_table=uploaded_table
    )

@app.route("/trading-dashboard")
def trading_dashboard():
    return "<h3>Trading Dashboard Sandbox</h3><a href='/'>← Back Hub</a>"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
