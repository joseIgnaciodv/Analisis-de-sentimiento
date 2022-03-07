from flask import Flask, jsonify, make_response
from flask_cors import CORS
from deep_translator import GoogleTranslator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nrclex import NRCLex

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def inicio():
    return 'Bienvenido, esta API permite analizar el sentimento del texto input'

@app.route('/sentiment', methods=['GET'])
def sentiment():
    return 'Para analizar el sentimiento escoga opción e introduzca texto'

@app.route('/sentiment/sentimentVader/<string:texto>', methods=['GET'])
def sentimientoVader(texto):
    if texto is None:
        make_response(jsonify({'error': 'Texto no encontrado'}), 404)
    traductor = GoogleTranslator(target='en')
    texto_traducido = traductor.translate(texto)
    vs = SentimentIntensityAnalyzer()
    vs_result = vs.polarity_scores(texto_traducido)
    return jsonify({'sentiment': vs_result}), 200

@app.route('/sentiment/sentimentNRCLex/<string:texto>', methods=['GET'])
def sentimientoNRCLex(texto):
    if texto is None:
        make_response(jsonify({'error': 'Texto no encontrado'}), 404)
    traductor = GoogleTranslator(target='en')
    texto_traducido = traductor.translate(texto)
    nrc = NRCLex(texto_traducido)
    return jsonify({'sentiment': nrc.raw_emotion_scores}), 200


if __name__ == '__main__':
    app.run(debug = True)