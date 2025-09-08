from flask import Flask, render_template, request, jsonify
import chat  # import your chatbot functions
import threading, webbrowser

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def get_bot_response():
    user_text = request.json.get("message")
    ints = chat.predict_class(user_text)
    res = chat.get_response(ints, chat.intents)
    return jsonify({"response": res})

def open_browser():
    webbrowser.open("http://127.0.0.1:5000/")

if __name__ == "__main__":
    # Open browser automatically in a separate thread
    threading.Timer(1.5, open_browser).start()
    app.run(debug=True)
