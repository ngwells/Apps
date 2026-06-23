from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import datetime
import os

app = Flask(__name__)

CSV_FILE = "voice_data.csv"

def save_to_dataframe(text):
    """Appends the transcript and a timestamp to a CSV file/DataFrame."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_data = pd.DataFrame([{"Timestamp": timestamp, "Transcript": text}])
    
    if os.path.exists(CSV_FILE):
        new_data.to_csv(CSV_FILE, mode='a', header=False, index=False)
    else:
        new_data.to_csv(CSV_FILE, index=False)

# Embedded HTML so you don't need a separate templates/ folder
HTML_INTERFACE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Audio Voice Logger</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 600px; margin: 50px auto; text-align: center; background: #f9f9f9; }
        .container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        button { padding: 12px 24px; font-size: 16px; font-weight: bold; cursor: pointer; border: none; border-radius: 25px; margin: 10px; transition: background 0.3s; }
        .btn-record { background: #dc3545; color: white; }
        .btn-record:hover { background: #bd2130; }
        .btn-stop { background: #28a745; color: white; }
        .btn-stop:hover { background: #218838; }
        button:disabled { background: #ccc; cursor: not-allowed; }
        #status { margin: 20px 0; font-weight: bold; color: #555; }
    </style>
</head>
<body>
<div class="container">
    <h1>Voice Recorder Logger</h1>
    <p>Click "Start Recording" to open your mic, and "Stop & Process" to save the text.</p>
    
    <button id="start-btn" class="btn-record">Start Recording</button>
    <button id="stop-btn" class="btn-stop" disabled>Stop & Process</button>
    
    <div id="status">Status: Idle</div>
</div>

<script>
    let mediaRecorder;
    let audioChunks = [];
    const startBtn = document.getElementById('start-btn');
    const stopBtn = document.getElementById('stop-btn');
    const statusDiv = document.getElementById('status');

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
                statusDiv.innerText = "Status: Processing audio and saving to DataFrame...";
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                
                // Send audio to backend
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
                        statusDiv.innerText = `Saved: "${data.transcript}"`;
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
        // Stop all tracks to turn off the microphone light
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
    # Renders the HTML directly from the string variable above
    return render_template_string(HTML_INTERFACE)

@app.route("/process-audio", methods=["POST"])
def process_audio():
    if 'audio_data' not in request.files:
        return jsonify({"status": "error", "message": "No audio data received"}), 400
        
    audio_file = request.files['audio_data']
    
    # -------------------------------------------------------------------------
    # NOTE: To convert raw audio bytes into text in Python, you need a model.
    # For now, we will simulate the transcript extraction. You can replace this 
    # with an audio transcription library (like Whisper or SpeechRecognition).
    # -------------------------------------------------------------------------
    detected_text = "Sample transcript from recorded audio file" 
    
    save_to_dataframe(detected_text)
    return jsonify({"status": "success", "transcript": detected_text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))