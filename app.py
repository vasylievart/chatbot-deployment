from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

from chat import get_response

app = Flask(__name__)
CORS(app)

@app.post("/predict")
def predict(): 
  data = request.get_json()
  text = data.get('message')
  lang = data.get('lang', 'en')

  if not text: 
    return jsonify({"answer": "Lo siento, no entendí tu mensaje. Puedes contactarnos por WhatsApp or email."})

  #TODO: check if text is valid
  response = get_response(text, lang)
  message = {"answer": response}
  return jsonify(message)

if __name__ == "__main__":
  app.run(debug=True)