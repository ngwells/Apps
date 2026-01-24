# from flask import Flask

# app = Flask(__name__)

# @app.route("/")
# def hello_world():
#     return "<h1>Hello, Coach Edgar!</h1>"

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=10000)



from flask import Flask, request

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    message = ""

    if request.method == "POST":
        message = request.form.get("user_text", "")

    return f"""
        <h1>Hello, Coach Edgar!</h1>

        <form method="POST">
            <input type="text" name="user_text" placeholder="Type something..." style="padding:8px; width:250px;">
            <button type="submit" style="padding:8px;">Submit</button>
        </form>

        <h2>{message}</h2>
    """

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
