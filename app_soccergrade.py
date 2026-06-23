from flask import Flask, request, jsonify, render_template_string
from mistralai.client import Mistral
import pandas as pd
import datetime
import os

app = Flask(__name__)

# This will create and append to a CSV file right inside your local machine's folder
LOCAL_CSV_FILE = "local_voice_history.csv"

# Initialize Mistral Client from environment variable
API_KEY = os.environ.get("MISTRAL_API_KEY")
client = Mistral(api_key=API_KEY) if API_KEY else None

def save_to_local_directory(text):
    """Appends the transcript and a timestamp to a local CSV file on your machine."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_data = pd.DataFrame([{"Timestamp": timestamp, "Transcript": text}])
    
    if os.path.exists(LOCAL_CSV_FILE):
        new_data.to_csv(LOCAL_CSV_FILE, mode='a', header=False, index=False)
    else:
        new_data.to_csv(LOCAL_CSV_FILE, index=False)

# Embedded HTML: Starts with a completely empty history on fresh page load (Option 1)
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
    <p>Click "Start Recording" to open your mic, and "Stop & Process" to transcribe your speech.</p>
    
    <button id="start-btn" class="btn-record">Start Recording</button>
    <button id="stop-btn" class="btn-stop" disabled>Stop & Process</button>
    
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
    const statusDiv = document.getElementById('status');
    const historyList = document.getElementById('history-list');
    
    // Array to manage rows just for the active session in this browser window
    let sessionRecords = [];

    function appendToSessionDOM(timestamp, transcript) {
        const emptyState = document.getElementById('empty-state');
        if (emptyState) emptyState.remove();

        const li = document.createElement('li');
        li.className = 'history-item';
        li.innerHTML = `<span class="timestamp">[${timestamp}]</span> ${transcript}`;
        
        // Inserts the newest transcription at the top of the browser list view
        historyList.insertBefore(li, historyList.firstChild);
    }

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
                        statusDiv.innerText = "Transcribed and saved locally!";
                        // Add to browser UI interface instantly
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
    # Option 1 behavior: returns a fresh interface without loading any global server history
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

    # Capture the current execution timestamp
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Save to your computer's local log layout directory file
    save_to_local_directory(detected_text)
    
    return jsonify({
        "status": "success", 
        "transcript": detected_text,
        "timestamp": current_time
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))