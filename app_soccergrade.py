from flask import Flask, request, jsonify, render_template_string
from mistralai.client import Mistral
import pandas as pd
import datetime
import os

app = Flask(__name__)

# Initialize Mistral Client from environment variable
API_KEY = os.environ.get("MISTRAL_API_KEY")
client = Mistral(api_key=API_KEY) if API_KEY else None

# Embedded HTML with dynamic session list and Local CSV download functionality
HTML_INTERFACE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Audio Voice Logger</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 650px; margin: 50px auto; text-align: center; background: #f9f9f9; }
        .container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        button { padding: 12px 24px; font-size: 16px; font-weight: bold; cursor: pointer; border: none; border-radius: 25px; margin: 10px; transition: background 0.3s; }
        .btn-record { background: #dc3545; color: white; }
        .btn-record:hover { background: #bd2130; }
        .btn-stop { background: #28a745; color: white; }
        .btn-stop:hover { background: #218838; }
        .btn-save { background: #007bff; color: white; }
        .btn-save:hover { background: #0056b3; }
        button:disabled { background: #ccc; cursor: not-allowed; }
        #status { margin: 20px 0; font-weight: bold; color: #555; }
        
        .history-container { margin-top: 30px; text-align: left; }
        .history-list { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 0; list-style: none; max-height: 300px; overflow-y: auto; }
        .history-item { padding: 12px 15px; border-bottom: 1px solid #eee; font-size: 14px; }
        .history-item:last-child { border-bottom: none; }
        .timestamp { color: #888; font-weight: bold; margin-right: 10px; font-size: 12px; }
    </style>
</head>
<body>
<div class="container">
    <h1>Voice Recorder Logger</h1>
    <p>Click "Start Recording" to open your mic, and "Stop & Process" to transcribe.</p>
    
    <button id="start-btn" class="btn-record">Start Recording</button>
    <button id="stop-btn" class="btn-stop" disabled>Stop & Process</button>
    <button id="export-btn" class="btn-save" disabled>Export Session CSV</button>
    
    <div id="status">Status: Idle</div>

    <div class="history-container">
        <h3>Session Text History (Private to You):</h3>
        <ul id="history-list" class="history-list">
            <li class="history-item" id="empty-state" style="color: #aaa; text-align:center;">No recordings logged in this session yet.</li>
        </ul>
    </div>
</div>

<script>
    let mediaRecorder;
    let audioChunks = [];
    const startBtn = document.getElementById('start-btn');
    const stopBtn = document.getElementById('stop-btn');
    const exportBtn = document.getElementById('export-btn');
    const statusDiv = document.getElementById('status');
    const historyList = document.getElementById('history-list');
    
    // In-memory array to store rows for the export function
    let sessionRecords = [];

    function appendToSessionDOM(timestamp, transcript) {
        const emptyState = document.getElementById('empty-state');
        if (emptyState) emptyState.remove();

        // Push data to memory list for csv compilation
        sessionRecords.push({ timestamp, transcript });
        exportBtn.disabled = false; // Enable export button

        const li = document.createElement('li');
        li.className = 'history-item';
        li.innerHTML = `<span class="timestamp">[${timestamp}]</span> ${transcript}`;
        
        // Inserts the newest transcription at the top of the browser view list
        historyList.insertBefore(li, historyList.firstChild);
    }

    // Function to let the user select their local directory and save the file
    exportBtn.addEventListener('click', async () => {
        if (sessionRecords.length === 0) return;

        // Build CSV string layout
        let csvContent = "Timestamp,Transcript\\n";
        sessionRecords.forEach(row => {
            // Escape double quotes inside the text
            let cleanTranscript = row.transcript.replace(/"/g, '""');
            csvContent += `"${row.timestamp}","${cleanTranscript}"\\n`;
        });

        try {
            // Opens native file system save-as browser prompt dialog layout window
            const options = {
                suggestedName: 'voice_history.csv',
                types: [{
                    description: 'CSV Files',
                    accept: { 'text/csv': ['.csv'] },
                }],
            };
            
            const handle = await window.showSaveFilePicker(options);
            const writable = await handle.createWritable();
            await writable.write(csvContent);
            await writable.close();
            
            statusDiv.style.color = 'green';
            statusDiv.innerText = "Status: File saved successfully to your selected directory!";
        } catch (err) {
            if (err.name !== 'AbortError') {
                statusDiv.style.color = 'red';
                statusDiv.innerText = `Export failed: ${err.message}`;
                console.error(err);
            }
        }
    });

    startBtn.addEventListener('click', async () => {
        audioChunks = [];
        statusDiv.style.color = '#555';
        statusDiv.innerText = "Status: Requesting microphone access...";
        
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            
            mediaRecorder.ondataavailable = event => {
                audioChunks.push(event.data);
            };

            mediaRecorder.onstop = async () => {
                statusDiv.innerText = "Status: Transcribing audio file...";
                const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                
                const formData = new FormData();
                formData.append('audio_data', audioBlob, 'recording.webm');

                fetch('/process-audio', {
                    method: 'POST',
                    body: formData
                })
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
            statusDiv.innerText = "Status: Microphone access denied.";
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

@app.route("/")
def index():
    return render_template_string(HTML_INTERFACE)

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
                file={
                    "content": f.read(),
                    "file_name": temp_filename
                }
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
    
    return jsonify({
        "status": "success", 
        "transcript": detected_text,
        "timestamp": current_time
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))