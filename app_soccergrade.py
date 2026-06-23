from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import datetime
import os

app = Flask(__name__)

CSV_FILE = "voice_data.csv"

def save_to_dataframe(text):
    """Appends the transcript and a timestamp to a CSV file/DataFrame and returns all rows."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_data = pd.DataFrame([{"Timestamp": timestamp, "Transcript": text}])
    
    if os.path.exists(CSV_FILE):
        new_data.to_csv(CSV_FILE, mode='a', header=False, index=False)
    else:
        new_data.to_csv(CSV_FILE, index=False)
    
    # Read the updated file to return full history
    df = pd.read_csv(CSV_FILE)
    return df.to_dict(orient="records")

def get_all_records():
    """Helper to fetch existing records if the page is refreshed."""
    if os.path.exists(CSV_FILE):
        try:
            df = pd.read_csv(CSV_FILE)
            return df.to_dict(orient="records")
        except Exception:
            return []
    return []

# Embedded HTML with dynamic list display
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
    <p>Click "Start Recording" to open your mic, and "Stop & Process" to append to the DataFrame.</p>
    
    <button id="start-btn" class="btn-record">Start Recording</button>
    <button id="stop-btn" class="btn-stop" disabled>Stop & Process</button>
    
    <div id="status">Status: Idle</div>

    <div class="history-container">
        <h3>Processed Text History:</h3>
        <ul id="history-list" class="history-list">
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

    // Function to render rows into the UI list
    function updateHistoryDOM(records) {
        historyList.innerHTML = ""; 
        if (records.length === 0) {
            historyList.innerHTML = '<li class="history-item" style="color: #aaa; text-align:center;">No recordings logged yet.</li>';
            return;
        }
        // Loop backwards to show newest logs on top
        records.slice().reverse().forEach(row => {
            const li = document.createElement('li');
            li.className = 'history-item';
            li.innerHTML = `<span class="timestamp">[${row.Timestamp}]</span> ${row.Transcript}`;
            historyList.appendChild(li);
        });
    }

    // Load initial history when page loads
    const initialHistory = {{ initial_history | tojson }};
    updateHistoryDOM(initialHistory);

    startBtn.addEventListener('click', async () => {
        audioChunks = [];
        statusDiv.innerText = "Status: Requesting microphone access...";
        
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            
            mediaRecorder.ondataavailable = event => {
                audioChunks.push(event.data);
            };

            mediaRecorder.onstop = async () => {
                statusDiv.innerText = "Status: Processing audio and adding row...";
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                
                const formData = new FormData();
                formData.append('audio_data', audioBlob);

                fetch('/process-audio', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        statusDiv.style.color = 'green';
                        statusDiv.innerText = "Row added successfully!";
                        // Update UI with the new full dataframe history
                        updateHistoryDOM(data.history);
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
    # Pass existing items to display on full page load
    history = get_all_records()
    return render_template_string(HTML_INTERFACE, initial_history=history)

@app.route("/process-audio", methods=["POST"])
def process_audio():
    if 'audio_data' not in request.files:
        return jsonify({"status": "error", "message": "No audio data received"}), 400
        
    audio_file = request.files['audio_data']
    
    # Placeholder: Replace this string with actual transcription engines (e.g., OpenAI Whisper API, etc.)
    detected_text = "Sample transcript from recorded audio file" 
    
    # Saves row to CSV and returns full updated contents
    updated_history = save_to_dataframe(detected_text)
    
    return jsonify({
        "status": "success", 
        "transcript": detected_text,
        "history": updated_history
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))