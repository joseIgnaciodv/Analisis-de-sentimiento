from flask import Flask, jsonify, request, abort, make_response
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nrclex import NRCLex
from google_trans_new import google_translator
import deepl

app = Flask(__name__)

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
    vs = SentimentIntensityAnalyzer()
    vs_result = vs.polarity_scores(texto)
    return jsonify({'sentiment': vs_result})

@app.route('/traductor/<string:texto>', methods=['GET'])
def traducir(texto):
    if texto is None:
        make_response(jsonify({'error': 'Texto no encontrado'}), 404)
    traductor = google_translator()
    traduccion = traductor.translate(texto, lang_src="es", lang_tgt="en")
    return jsonify({'traduccion': traduccion})

# pip install deepl-translate (download version 1.2.0) 
@app.route('/traductor/deepl/<string:texto>', methods=['GET'])
def traducirDeepl(texto):
    if texto is None:
        make_response(jsonify({'error': 'Texto no encontrado'}), 404)
    traduccion = deepl.translate(source_language="ES", target_language="EN", text=texto)
    return jsonify({'traduccion': traduccion})

@app.route('/sentiment/sentimentNRCLex/<string:texto>', methods=['GET'])
def sentimientoNRCLex(texto):
    if texto is None:
        make_response(jsonify({'error': 'Texto no encontrado'}), 404)
    nrc = NRCLex(texto)
    return jsonify({'sentiment': nrc.raw_emotion_scores})


if __name__ == '__main__':
    app.run(debug = True)