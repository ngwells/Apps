import datetime
import json
import os
import re
from flask import Flask, jsonify, request, render_template_string, session
from mistralai.client import Mistral
import numpy as np
import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "soccer-grade-secret-automation-key")

# ===================================================================== #
# CONFIGURATION: Ensure cookie expires when browser closes              #
# ===================================================================== #
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
)

# Initialize Mistral Client securely using ONLY the system environment variable
API_KEY = os.environ.get("MISTRAL_API_KEY")
client = Mistral(api_key=API_KEY) if API_KEY else None

# ===================================================================== #
# SECTION 1: HTML INTERFACES (FRONTEND UI LAYOUTS)                    #
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
        .system-controls { margin-top: 50px; border-top: 1px solid #ddd; padding-top: 20px; }
        .btn-reset-all { padding: 10px 20px; background: #6c757d; color: white; border: none; border-radius: 6px; font-weight: bold; cursor: pointer; transition: background 0.2s; }
        .btn-reset-all:hover { background: #5a6268; }
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
    <div class="system-controls">
        <button class="btn-reset-all" onclick="clearFullSession()">Reset Environment Roster Cache</button>
    </div>
    <script>
        function clearFullSession() {
            if (confirm("Are you sure you want to completely flush all loaded dataframes, recordings, and generated files?")) {
                fetch('/system/reset-session', { method: 'POST' })
                .then(() => {
                    alert("Cache context cleared successfully.");
                    window.location.reload();
                });
            }
        }
    </script>
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

            fetch('/soccer-grade/sync-raw', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ data: sessionRecords })
            });
        }

        processBtn.addEventListener('click', async () => {
            if (sessionRecords.length === 0) return;
            statusDiv.innerText = "Status: Split-processing transcripts...";
            try {
                const response = await fetch('/soccer-grade/split-dataframe', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ data: sessionRecords.map(r => ({ timestamp: r.Timestamp, transcript: r.Transcript })) })
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

# --- PAGE C: DATA UPLOAD MANAGER ---
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
        .upload-flex-grid { display: flex; gap: 20px; margin-top: 20px; }
        .upload-zone { flex: 1; border: 2px dashed #007bff; padding: 20px; text-align: center; background: #f8f9fa; border-radius: 6px; }
        .upload-zone.override { border-color: #dc3545; background: #fff5f5; }
        input[type="file"] { margin-top: 10px; max-width: 100%; }
        .badge-alert { background: #dc3545; color: white; padding: 3px 6px; font-size: 11px; font-weight: bold; border-radius: 4px; display: inline-block; margin-bottom: 5px; }
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

        <h3>
            Voice Stream Snapshot (Processed & Exploded)
            {% if was_overridden %}<span class="badge-alert">Testing Override Active</span>{% endif %}
        </h3>
        <div class="data-block" {% if was_overridden %}style="border-left: 4px solid #dc3545;"{% endif %}>
            {% if processed_data_table %}
                {{ processed_data_table|safe }}
            {% else %}
                <span style="color:#999;">No processed/split comment structures currently loaded in session memory.</span>
            {% endif %}
        </div>

        <div class="upload-flex-grid">
            <div class="upload-zone">
                <form action="/upload-manager/submit-file" method="POST" enctype="multipart/form-data">
                    <label style="font-weight:bold; display:block; color:#0d47a1;">Upload Spreadsheet Metrics</label>
                    <input type="file" name="uploaded_csv" accept=".csv" required><br><br>
                    <button type="submit" style="padding:8px 16px; background:#28a745; color:white; border:none; border-radius:4px; font-weight:bold; cursor:pointer;">Ingest Standard File</button>
                </form>
            </div>

            <div class="upload-zone override">
                <form action="/upload-manager/override-processed" method="POST" enctype="multipart/form-data">
                    <label style="font-weight:bold; display:block; color:#c0392b;">Sandbox Testing Mock</label>
                    <span style="font-size: 11px; color: #7f8c8d; display:block; margin-bottom:5px;">Forces override of Processed & Exploded dataset frame</span>
                    <input type="file" name="mock_processed_csv" accept=".csv" required><br><br>
                    <button type="submit" style="padding:8px 16px; background:#dc3545; color:white; border:none; border-radius:4px; font-weight:bold; cursor:pointer;">Inject Test Override</button>
                </form>
            </div>
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
        .container { background: white; padding: 40px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: left; }
        .placeholder-box { border: 2px dashed #ffc107; background: #fffde7; padding: 20px; border-radius: 6px; margin-bottom: 30px; text-align: center; }
        
        .format-selector { display: flex; justify-content: center; gap: 15px; margin: 25px 0; }
        .btn-format { padding: 12px 30px; font-size: 16px; font-weight: bold; border: 2px solid #007bff; background: white; color: #007bff; border-radius: 8px; cursor: pointer; transition: all 0.2s; }
        .btn-format.active { background: #007bff; color: white; box-shadow: 0 4px 10px rgba(0,123,255,0.3); }
        .btn-format:hover { background: #e6f0ff; }
        .btn-format.active:hover { background: #007bff; }
        
        .action-container { text-align: center; margin: 20px 0; display: flex; justify-content: center; gap: 15px; }
        .btn-execute { padding: 14px 30px; font-size: 15px; font-weight: bold; background: #28a745; color: white; border: none; border-radius: 6px; cursor: pointer; box-shadow: 0 3px 6px rgba(0,0,0,0.1); }
        .btn-execute:disabled { background: #ccc; cursor: not-allowed; box-shadow: none; }
        .btn-clear-frame { padding: 14px 30px; font-size: 15px; font-weight: bold; background: #dc3545; color: white; border: none; border-radius: 6px; cursor: pointer; box-shadow: 0 3px 6px rgba(0,0,0,0.1); }
        
        .llm-response-box { background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 6px; padding: 20px; margin-top: 25px; min-height: 100px; }
        .llm-response-box table { width: 100%; border-collapse: collapse; margin-top: 15px; background: white; }
        .llm-response-box th, .llm-response-box td { border: 1px solid #dee2e6; padding: 10px; text-align: left; font-size: 13px; }
        .llm-response-box th { background-color: #f1f3f5; font-weight: bold; }
        
        .loader { display: none; text-align: center; font-weight: bold; color: #666; margin: 20px 0; }
        .debug-panel { background: #f1f3f5; border-radius: 6px; padding: 15px; font-size: 12px; margin-top: 30px; }
    </style>
</head>
<body>
    <div class="back-nav">
        <a href="/" class="back-link">← Back to Hub</a>
    </div>
    <div class="container">
        <h2 style="text-align: center; margin-bottom: 5px;">Line Up Builder Workspace</h2>
        <p style="color:#666; text-align: center; margin-bottom: 25px;">Select a configuration grid below to evaluate tactical alignment blueprints via Mistral AI.</p>
        
        <div class="placeholder-box">
            <h3 style="color: #856404; margin-top: 0; font-size: 16px;">Content Coming Soon</h3>
            <p style="color: #666; margin-bottom: 0; font-size: 13px;">The visual field arranger and interactive drag-and-drop roster canvas are currently in validation.</p>
        </div>

        <h3 style="font-size: 16px; color:#333; border-bottom: 1px solid #eee; padding-bottom: 8px;">1. Select Roster Matrix Format</h3>
        <div class="format-selector">
            <button type="button" id="btn-7v7" class="btn-format {% if selected_format == '7v7' %}active{% endif %}" onclick="selectFormat(this, '7v7')">7v7</button>
            <button type="button" id="btn-9v9" class="btn-format {% if selected_format == '9v9' %}active{% endif %}" onclick="selectFormat(this, '9v9')">9v9</button>
            <button type="button" id="btn-11v11" class="btn-format {% if selected_format == '11v11' %}active{% endif %}" onclick="selectFormat(this, '11v11')">11v11</button>
        </div>

        <div class="action-container">
            <button type="button" id="execute-btn" class="btn-execute" onclick="runTacticalPrompt()">Execute Tactical Blueprint Generation</button>
            <button type="button" id="clear-btn" class="btn-clear-frame" onclick="clearBlueprintFrame()">Clear Blueprint Table</button>
        </div>

        <div class="loader" id="loading-spinner">Processing context frames and executing Mistral matrix construction...</div>

        <h3 style="font-size: 16px; color:#333; margin-top: 30px;">2. Generated System Response Dataframe</h3>
        <div class="llm-response-box" id="response-anchor">
            {% if blueprint_table %}
                {{ blueprint_table|safe }}
            {% else %}
                <span style="color:#999; font-style: italic;">No configuration structure generated. Select a match format above and click execute.</span>
            {% endif %}
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

    <script>
        let selectedFormatName = "{{ selected_format|safe }}";

        function selectFormat(clickedButton, formatValue) {
            const buttons = document.querySelectorAll('.btn-format');
            buttons.forEach(btn => btn.classList.remove('active'));
            
            clickedButton.classList.add('active');
            selectedFormatName = formatValue;
            document.getElementById('execute-btn').disabled = false;

            fetch('/create-lineup/select-format', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ format_type: formatValue })
            });
        }

        async function runTacticalPrompt() {
            if (!selectedFormatName) return;
            
            const runButton = document.getElementById('execute-btn');
            const spinner = document.getElementById('loading-spinner');
            const responseAnchor = document.getElementById('response-anchor');
            
            runButton.disabled = true;
            spinner.style.display = "block";
            responseAnchor.innerHTML = '<span style="color:#666; font-style:italic;">Querying Mistral model layers...</span>';
            
            try {
                const response = await fetch('/create-lineup/generate-tactics', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ format_type: selectedFormatName })
                });
                const data = await response.json();
                
                spinner.style.display = "none";
                runButton.disabled = false;
                
                if (data.status === 'success') {
                    responseAnchor.innerHTML = data.html_payload;
                } else {
                    responseAnchor.innerHTML = `<span style="color:#dc3545; font-weight:bold;">Error: ${data.message}</span>`;
                }
            } catch (err) {
                spinner.style.display = "none";
                runButton.disabled = false;
                responseAnchor.innerHTML = '<span style="color:#dc3545; font-weight:bold;">Network pipeline transmission breakdown.</span>';
            }
        }

        function clearBlueprintFrame() {
            fetch('/create-lineup/clear-blueprint', { method: 'POST' })
            .then(res => res.json())
            .then(data => {
                if(data.status === 'success') {
                    document.getElementById('response-anchor').innerHTML = '<span style="color:#999; font-style: italic;">No configuration structure generated. Select a match format above and click execute.</span>';
                }
            });
        }
    </script>
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
        body { font-family: Arial, sans-serif; max-width: 1000px; margin: 40px auto; text-align: center; background: #f9f9f9; }
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
        .badge-alert { background: #dc3545; color: white; padding: 2px 5px; font-size: 10px; font-weight: bold; border-radius: 3px; }
        
        .metric-banner { background: #f1f3f5; border: 1px solid #ccc; padding: 15px; border-radius: 6px; margin-bottom: 25px; display: flex; align-items: center; justify-content: space-between; }
        .action-cluster { display: flex; gap: 10px; }
        .btn-metric { background: #2e7d32; color: white; border: none; padding: 10px 20px; font-weight: bold; border-radius: 4px; cursor: pointer; transition: background 0.2s; }
        .btn-metric:hover { background: #1b5e20; }
        .btn-clear-metrics { background: #dc3545; color: white; border: none; padding: 10px 20px; font-weight: bold; border-radius: 4px; cursor: pointer; transition: background 0.2s; }
        .btn-clear-metrics:hover { background: #bd2130; }
    </style>
</head>
<body>
    <div class="back-nav">
        <a href="/" class="back-link">← Back to Hub</a>
    </div>
    <div class="container">
        <h2>Analytics & Unified Data Reporting</h2>
        <p style="color:#666; margin-bottom: 25px;">Cross-reference spreadsheet telemetry sets with processed runtime streaming voice buffers directly below.</p>
        
        <div class="metric-banner">
            <div>
                <strong style="color:#222; display:block; font-size:15px;">Semantic Player-to-Position Evaluation System</strong>
                <span style="font-size:12px; color:#666;">Triggers Mistral Vector Embeddings, Cosine Similarity calculations, and populates results_df.</span>
            </div>
            <div class="action-cluster">
                <form action="/analytics/compute-metrics" method="POST" style="margin:0;">
                    <button type="submit" class="btn-metric">Initiate Analytics Processing</button>
                </form>
                <form action="/analytics/clear-metrics" method="POST" style="margin:0;">
                    <button type="submit" class="btn-clear-metrics">Clear results_df Frame</button>
                </form>
            </div>
        </div>

        {% if similarity_results_table %}
        <div class="section-box" style="border-left: 4px solid #2e7d32; margin-bottom:25px; background: #fbfdfb;">
            <h3 style="margin-top:0; color:#2e7d32;">🎯 Top Position Fit Recommendations (results_df)</h3>
            <div class="table-wrap" style="max-height: 350px;">
                {{ similarity_results_table|safe }}
            </div>
        </div>
        {% endif %}

        <div class="data-grid">
            <div class="section-box" style="border-left: 4px solid #007bff;">
                <h3 style="margin-top:0; color:#007bff;">1. Live Voice Stream Ingestion (Raw Logs)</h3>
                <div class="table-wrap">
                    {% if raw_table %} {{ raw_table|safe }} {% else %} <span class="empty-text">No active voice stream capture cached in memory.</span> {% endif %}
                </div>
            </div>

            <div class="section-box" style="border-left: 4px solid #6f42c1;">
                <h3 style="margin-top:0; color:#6f42c1;">
                    2. Exploded Roster Logs (Smart Split Comment Entities)
                    {% if was_overridden %}<span class="badge-alert">Testing Override Active</span>{% endif %}
                </h3>
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

            <div class="section-box" style="border-left: 4px solid #ffc107;">
                <h3 style="margin-top:0; color:#b58100;">4. Generated Tactical Blueprint Frame (Mistral AI Core)</h3>
                <div class="table-wrap">
                    {% if blueprint_table %} {{ blueprint_table|safe }} {% else %} <span class="empty-text">No roster configuration blueprint generated yet from the Create Line Up tab.</span> {% endif %}
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""

# ===================================================================== #
# SECTION 2: UTILITIES & VECTOR EMBEDDINGS ENGINES                       #
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
    """Finds a column header closely resembling 'position' via simple character sequence evaluation."""
    target = "position"
    best_col = None
    max_matches = 0
    
    for col in df.columns:
        col_str = str(col).lower().strip()
        if col_str == target:
            return df
        # Simple character matching sequence ratio
        matches = sum(1 for char in target if char in col_str)
        if matches > max_matches and len(col_str) >= 4:
            max_matches = matches
            best_col = col
            
    if best_col is not None and max_matches >= 5:
        df.rename(columns={best_col: "Position"}, inplace=True)
    return df

# ===================================================================== #
# SECTION 3: APPS CONTROLLER ENGINE & ROUTING                           #
# ===================================================================== #

@app.route("/")
def index():
    return render_template_string(LANDING_PAGE_HTML)

@app.route("/system/reset-session", methods=["POST"])
def reset_session():
    session.clear()
    return jsonify({"status": "cleared"})

@app.route("/soccer-grade")
def soccer_grade():
    raw_records = session.get('cached_raw', [])
    processed_records = session.get('cached_processed', [])
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
    req_data = request.get_json()
    if req_data and 'data' in req_data:
        session['cached_raw'] = req_data['data']
    return jsonify({"status": "synchronized"})

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
        session.pop('processed_was_overridden', None)
        
        return jsonify({"status": "success", "processed_data": df_exploded.to_dict(orient='records')})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/soccer-grade/process-audio", methods=["POST"])
def process_audio():
    if not client:
        return jsonify({"status": "error", "message": "Mistral API client initialization error: MISTRAL_API_KEY environment variable missing."}), 500
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
    was_overridden = session.get('processed_was_overridden', False)
    
    raw_data_table = pd.DataFrame(raw_session).to_html(classes='table', index=False) if raw_session else None
    processed_data_table = pd.DataFrame(processed_session).to_html(classes='table', index=False) if processed_session else None
    uploaded_data_table = pd.DataFrame(uploaded_session).to_html(classes='table', index=False) if uploaded_session else None
    
    return render_template_string(UPLOAD_PAGE_HTML, 
                                  raw_data_table=raw_data_table, 
                                  processed_data_table=processed_data_table, 
                                  uploaded_data_table=uploaded_data_table,
                                  was_overridden=was_overridden)

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

@app.route("/upload-manager/override-processed", methods=["POST"])
def override_processed_data():
    if 'mock_processed_csv' not in request.files:
        return "No file selected for testing override", 400
    file = request.files['mock_processed_csv']
    if file.filename == '':
        return "Empty file selection", 400
    try:
        mock_df = pd.read_csv(file)
        session['cached_processed'] = mock_df.to_dict(orient='records')
        session['processed_was_overridden'] = True
        return render_template_string("<h3>Testing Override Applied!</h3><p>Exploded database frame temporarily swapped.</p><script>setTimeout(function(){window.location.href='/upload-manager';}, 1200);</script>")
    except Exception as e:
        return f"Error applying template mock override: {str(e)}", 500

# --- ROUTE: CREATE LINE UP ---
@app.route("/create-lineup")
def create_lineup():
    raw_session = session.get('cached_raw', [])
    processed_session = session.get('cached_processed', [])
    uploaded_session = session.get('cached_uploaded', [])
    
    selected_format = session.get('selected_format', '')
    blueprint_table = session.get('cached_blueprint', None)
    
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
    req_body = request.get_json()
    if req_body and 'format_type' in req_body:
        session['selected_format'] = req_body['format_type']
    return jsonify({"status": "format_cached"})

@app.route("/create-lineup/clear-blueprint", methods=["POST"])
def clear_blueprint():
    session.pop('cached_blueprint', None)
    return jsonify({"status": "success"})

# --- ASYNC ROUTE: MISTRAL TACTICAL LLM PROMPT GENERATOR ---
@app.route("/create-lineup/generate-tactics", methods=["POST"])
def generate_tactics():
    if not client:
        return jsonify({"status": "error", "message": "Mistral API client initialization error: MISTRAL_API_KEY environment variable missing."}), 500
        
    req_body = request.get_json()
    format_type = req_body.get("format_type", "11v11")
    session['selected_format'] = format_type
    
    prompt_instruction = f"""
    give me the characteristics and skills required for a player for each position in {format_type} line up. sperate each position and use the characteristics and skills from all the all time great players for that position. Create a data frame with one column being position and the other column being a narrative description of the characteristics and skills for that position.
    
    CRITICAL OUTPUT RULE: Return ONLY a valid JSON format list of objects representing this dataframe array. No extra commentary prose text.
    Format Example:
    [
      {{"position": "Goalkeeper (GK)", "narrative_description": "Exceptional shot-stopping reflexes..."}}
    ]
    """
    
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
        
        session['cached_blueprint'] = html_table
        return jsonify({"status": "success", "html_payload": html_table})
        
    except Exception as e:
        return jsonify({"status": "error", "message": f"Pipeline failure: {str(e)}"}), 500

# --- ROUTE: ANALYTICS HUB VIEW ---
@app.route("/analytics")
def analytics():
    raw_session = session.get('cached_raw', [])
    processed_session = session.get('cached_processed', [])
    uploaded_session = session.get('cached_uploaded', [])
    blueprint_table = session.get('cached_blueprint', None)
    was_overridden = session.get('processed_was_overridden', False)
    
    raw_table = pd.DataFrame(raw_session).to_html(classes='table', index=False) if raw_session else None
    processed_table = pd.DataFrame(processed_session).to_html(classes='table', index=False) if processed_session else None
    uploaded_table = pd.DataFrame(uploaded_session).to_html(classes='table', index=False) if uploaded_session else None
    similarity_results_table = session.get('similarity_results_html', None)
    
    return render_template_string(
        ANALYTICS_PAGE_HTML,
        raw_table=raw_table,
        processed_table=processed_table,
        uploaded_table=uploaded_table,
        blueprint_table=blueprint_table,
        was_overridden=was_overridden,
        similarity_results_table=similarity_results_table
    )

@app.route("/analytics/clear-metrics", methods=["POST"])
def clear_metrics_dataframe():
    session.pop('similarity_results_html', None)
    return render_template_string("<h3>Results Dataframe Flushed</h3><script>window.location.href='/analytics';</script>")

# --- ROUTE: VECTOR SIMILARITY ENGINE & RESULTS_DF STRUCT BUILDER ---
@app.route("/analytics/compute-metrics", methods=["POST"])
def compute_metrics():
    processed_session = session.get('cached_processed', [])
    uploaded_session = session.get('cached_uploaded', [])
    blueprint_html = session.get('cached_blueprint', None)
    
    if not processed_session:
        return "Error: Processed Player Evaluations dataset frame missing.", 400
    if not uploaded_session and not blueprint_html:
        return "Error: Missing ideal target vectors. Load an external spreadsheet or generate a Line Up blueprint first.", 400

    # 1. Align Player Evaluation Frame (player_evals_df)
    player_evals_df = pd.DataFrame(processed_session)
    if 'Player' not in player_evals_df.columns:
        player_evals_df['Player'] = player_evals_df['Transcript'].apply(
            lambda x: re.search(r'(player\s+\d+|\d+)', str(x), re.I).group(1) if re.search(r'(player\s+\d+|\d+)', str(x), re.I) else "Unknown"
        )
    if 'Description' not in player_evals_df.columns:
        player_evals_df['Description'] = player_evals_df['Transcript']

    # 2. Align Ideal Position Target Frame (ideal_player_df) From Tab
    if uploaded_session:
        ideal_player_df = pd.DataFrame(uploaded_session)
    else:
        try:
            ideal_player_df = pd.read_html(blueprint_html)[0]
        except Exception:
            return "Error parsing system blueprint data frames.", 500

    # Execute fuzzy spelling matcher for Position header
    ideal_player_df = fix_misspelled_position_header(ideal_player_df)

    # Standardize column contexts
    if 'Position' not in ideal_player_df.columns:
        if len(ideal_player_df.columns) >= 1:
            ideal_player_df.rename(columns={ideal_player_df.columns[0]: "Position"}, inplace=True)
            
    if 'Description' not in ideal_player_df.columns:
        if len(ideal_player_df.columns) >= 2:
            ideal_player_df.rename(columns={ideal_player_df.columns[1]: "Description"}, inplace=True)

    try:
        # 3. Compute Embeddings
        player_embeddings = [get_mistral_embeddings(desc) or [0]*1024 for desc in player_evals_df["Description"]]
        ideal_embeddings = [get_mistral_embeddings(desc) or [0]*1024 for desc in ideal_player_df["Description"]]

        # 4. Process Cosine Similarity Matrix Mapping
        similarity_matrix = cosine_similarity(player_embeddings, ideal_embeddings)
        
        # Lock position target strictly from ideal file attributes
        similarity_df = pd.DataFrame(similarity_matrix, index=player_evals_df["Player"], columns=ideal_player_df["Position"])

        # 5. Group by Position and pull top 3 candidates ranked by vector matching score
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
        
        # Enforce column sorting rules: Sorted explicitly by position blocks, then structural score confidence cascading down
        results_df.sort_values(by=["Position", "Confidence Score"], ascending=[True, False], inplace=True)
        
        # Enforce exact structural column presentation configuration layout: position, score, player
        results_df = results_df[["Position", "Confidence Score", "Player"]]
        
        # Save table into context session layout
        session['similarity_results_html'] = results_df.to_html(classes='table', index=False)
        return render_template_string("<h3>Matrix Calculations Completed!</h3><script>window.location.href='/analytics';</script>")
        
    except Exception as e:
        return f"Execution matrix construction failure: {str(e)}", 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
