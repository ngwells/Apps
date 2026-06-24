import datetime
import os
import re
from flask import Flask, jsonify, request, render_template_string
from mistralai.client import Mistral
import pandas as pd

app = Flask(__name__)

# Initialize Mistral Client from environment variable
API_KEY = os.environ.get("MISTRAL_API_KEY")
client = Mistral(api_key=API_KEY) if API_KEY else None

# Enhanced HTML with Split-Processing capabilities and dedicated download handlers
HTML_INTERFACE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Audio Voice Logger</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 700px; margin: 40px auto; text-align: center; background: #f9f9f9; }
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

        // 1. Process Button Client Handler
        processBtn.addEventListener('click', async () => {
            if (sessionRecords.length === 0) return;
            statusDiv.style.color = '#555';
            statusDiv.innerText = "Status: Split-processing transcripts...";
            
            try {
                const response = await fetch('/split-dataframe', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ data: sessionRecords })
                });
                const result = await response.json();
                
                if (result.status === 'success') {
                    processedRecords = result.processed_data;
                    
                    // Refresh UI
                    processedList.innerHTML = '';
                    processedRecords.forEach(row => {
                        const li = document.createElement('li');
                        li.className = 'history-item';
                        li.innerHTML = `<span class="timestamp">[${row.Timestamp}]</span> ${row.Transcript}`;
                        processedList.appendChild(li);
                    });
                    
                    exportProcessedBtn.disabled = false;
                    statusDiv.style.color = 'green';
                    statusDiv.innerText = "Status: Split processing finished!";
                } else {
                    statusDiv.style.color = 'red';
                    statusDiv.innerText = `Processing failed: ${result.message}`;
                }
            } catch (err) {
                statusDiv.style.color = 'red';
                statusDiv.innerText = "Server error during row processing.";
                console.error(err);
            }
        });

        // Universal Download Engine supporting iOS Safaris and Desktops alike
        async function downloadCSV(records, filename) {
            let csvContent = "Timestamp,Transcript\\n";
            records.forEach(row => {
                let text = row.transcript || row.Transcript || "";
                let time = row.timestamp || row.Timestamp || "";
                let cleanTranscript = text.replace(/"/g, '""');
                csvContent += `"${time}","${cleanTranscript}"\\n`;
            });

            if ('showSaveFilePicker' in window) {
                try {
                    const options = {
                        suggestedName: filename,
                        types: [{ description: 'CSV Files', accept: { 'text/csv': ['.csv'] } }],
                    };
                    const handle = await window.showSaveFilePicker(options);
                    const writable = await handle.createWritable();
                    await writable.write(csvContent);
                    await writable.close();
                    statusDiv.style.color = 'green';
                    statusDiv.innerText = `Status: ${filename} saved successfully!`;
                    return;
                } catch (err) {
                    if (err.name === 'AbortError') return;
                    console.warn("File Picker fell back:", err);
                }
            }

            // Universal Blob Fallback
            try {
                const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
                const url = URL.createObjectURL(blob);
                const link = document.createElement("a");
                link.setAttribute("href", url);
                link.setAttribute("download", filename);
                link.style.visibility = 'hidden';
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
                URL.revokeObjectURL(url);
                statusDiv.style.color = 'green';
                statusDiv.innerText = `Status: ${filename} downloaded to device storage!`;
            } catch (err) {
                statusDiv.style.color = 'red';
                statusDiv.innerText = `Export failed: ${err.message}`;
            }
        }

        exportBtn.addEventListener('click', () => downloadCSV(sessionRecords, 'voice_history.csv'));
        exportProcessedBtn.addEventListener('click', () => downloadCSV(processedRecords, 'processed_voice_data.csv'));

        startBtn.addEventListener('click', async () => {
            audioChunks = [];
            statusDiv.style.color = '#555';
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
                    
                    fetch('/process-audio', { method: 'POST', body: formData })
                        .then(response => response.json())
                        .then(data => {
                            if (data.status === 'success') {
                                statusDiv.style.color = 'green';
                                statusDiv.innerText = "Transcribed successfully!";
                                appendToSessionDOM(data.timestamp, data.transcript);
                            } else {
                                statusDiv.style.color = 'red';
                                statusDiv.innerText = `Error: ${data.message}`;
                            }
                        })
                        .catch(err => {
                            statusDiv.style.color = 'red';
                            statusDiv.innerText = "Server communication failed.";
                            console.error(err);
                        });
                };
                mediaRecorder.start();
                statusDiv.innerText = "Status: Recording... speak now.";
                startBtn.disabled = true;
                stopBtn.disabled = false;
            } catch (err) {
                statusDiv.style.color = 'red';
                statusDiv.innerText = "Status: Microphone access denied or unsupported format.";
                console.error(err);
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

# Helper function executing the smart contextual regex splitting logic
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

@app.route("/")
def index():
    return render_template_string(HTML_INTERFACE)

# 2. New Route: Accepts the array of rows, structures it to a DataFrame, splits, explodes, and returns JSON
@app.route("/split-dataframe", methods=["POST"])
def split_dataframe():
    try:
        req_data = request.get_json()
        if not req_data or 'data' not in req_data:
            return jsonify({"status": "error", "message": "No data layout provided"}), 400
            
        raw_rows = req_data['data']
        if not raw_rows:
            return jsonify({"status": "success", "processed_data": []})

        # Convert the received browser array payload into a clean Pandas DataFrame
        df = pd.DataFrame(raw_rows)
        # Match dictionary property keys coming from JS frontend mappings
        df.columns = ['Timestamp', 'Transcript']

        # Apply the multi-row contextual split rule mapping
        df['Transcript'] = df['Transcript'].apply(split_comments_smart)
        
        # Explode array back into longitudinal rows
        df_exploded = df.explode('Transcript').reset_index(drop=True)
        
        # Return converted record mapping to UI container
        processed_json = df_exploded.to_dict(orient='records')
        return jsonify({"status": "success", "processed_data": processed_json})
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/process-audio", methods=["POST"])
def process_audio():
    if not client:
        return jsonify({"status": "error", "message": "Mistral API key missing from environment variables."}), 500
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
