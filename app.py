from flask import Flask, request
from mistralai.client import Mistral
import os

app = Flask(__name__)

# Load API key from environment using the updated v2 SDK client import
client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

def ask_mistral(prompt):
    # Updated to the new chat.complete method structure
    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

@app.route("/", methods=["GET", "POST"])
def home():
    user_text = ""
    bot_reply = ""

    if request.method == "POST":
        user_text = request.form.get("user_text", "")
        bot_reply = ask_mistral(user_text)

        # Convert line breaks to <br> for clean HTML formatting
        bot_reply = bot_reply.replace("\n", "<br>")

    return f"""
        <h1>Mistral Chatbot</h1>

        <form method="POST">
            <input type="text" name="user_text" placeholder="Ask a question..."
                   style="padding:8px; width:300px;">
            <button type="submit" style="padding:8px;">Send</button>
        </form>

        <h3>You asked:</h3>
        <p>{user_text}</p>

        <h3>Chatbot says:</h3>

        <div style="
            background:#f1f1f1;
            padding:15px;
            border-radius:10px;
            max-width:500px;
            line-height:1.6;
            font-size:18px;
            margin-top:10px;
        ">
            {bot_reply}
        </div>
    """

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
