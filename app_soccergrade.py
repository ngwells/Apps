from flask import Flask, render_template, request, jsonify
import pandas as pd
import datetime
import os

app = Flask(__name__)

CSV_FILE = "voice_data.csv"

def save_to_dataframe(text):
    """Appends the transcript and a timestamp to a CSV file/DataFrame."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_data = pd.DataFrame([{"Timestamp": timestamp, "Transcript": text}])
    
    # If file exists, append without writing the header
    if os.path.exists(CSV_FILE):
        new_data.to_csv(CSV_FILE, mode='a', header=False, index=False)
    else:
        new_data.to_csv(CSV_FILE, index=False)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/save-text", methods=["POST"])
def save_text():
    data = request.get_json()
    user_text = data.get("text", "").strip()
    
    if user_text:
        save_to_dataframe(user_text)
        return jsonify({"status": "success", "message": f"Saved: '{user_text}'"})
    
    return jsonify({"status": "error", "message": "No text received."}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))