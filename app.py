# # from flask import Flask

# # app = Flask(__name__)

# # @app.route("/")
# # def hello_world():
# #     return "<h1>Hello, Coach Edgar!</h1>"

# # if __name__ == "__main__":
# #     app.run(host="0.0.0.0", port=10000)



# # from flask import Flask, request

# # app = Flask(__name__)

# # @app.route("/", methods=["GET", "POST"])
# # def home():
# #     message = ""

# #     if request.method == "POST":
# #         message = request.form.get("user_text", "")

# #     return f"""
# #         <h1>Hello, Coach Edgar!</h1>

# #         <form method="POST">
# #             <input type="text" name="user_text" placeholder="Type something..." style="padding:8px; width:250px;">
# #             <button type="submit" style="padding:8px;">Submit</button>
# #         </form>

# #         <h2>{message}</h2>
# #     """

# # if __name__ == "__main__":
# #     app.run(host="0.0.0.0", port=10000)


# from flask import Flask, request
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# app = Flask(__name__)

# # Load Mistral model (open-source)
# model_name = "mistralai/Mistral-7B-Instruct-v0.2"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.float16,
#     device_map="auto"
# )

# def ask_mistral(prompt):
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#     output = model.generate(
#         **inputs,
#         max_new_tokens=200,
#         temperature=0.7
#     )
#     return tokenizer.decode(output[0], skip_special_tokens=True)

# @app.route("/", methods=["GET", "POST"])
# def home():
#     user_text = ""
#     bot_reply = ""

#     if request.method == "POST":
#         user_text = request.form.get("user_text", "")
#         bot_reply = ask_mistral(user_text)

#     return f"""
#         <h1>Mistral Chatbot</h1>

#         <form method="POST">
#             <input type="text" name="user_text" placeholder="Ask a question..." 
#                    style="padding:8px; width:300px;">
#             <button type="submit" style="padding:8px;">Send</button>
#         </form>

#         <h3>You asked:</h3>
#         <p>{user_text}</p>

#         <h3>Chatbot says:</h3>
#         <p>{bot_reply}</p>
#     """

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=10000)

from flask import Flask, request
from mistralai import Mistral

app = Flask(__name__)

import os
client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])



def ask_mistral(prompt):
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
        <p>{bot_reply}</p>
    """

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))

