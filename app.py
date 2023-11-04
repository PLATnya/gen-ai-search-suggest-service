from flask import Flask, jsonify, request
from transformers import pipeline, set_seed, GPT2Tokenizer
from parrot import Parrot
from nltk.tokenize import word_tokenize
generator = pipeline('text-generation', model='gpt2')
parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5", use_gpu=False)

from flask_cors import CORS, cross_origin
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
@app.route('/', methods=['GET'])
def root():
    return jsonify({'msg' : 'fuck'})

@app.route('/generate', methods=['GET'])
@cross_origin()
def generate():
    input_value = request.args.get('q').strip()
    words_count = word_tokenize(input_value)
    return jsonify({'output' : [gen for gen in generator(input_value, max_length=len(words_count)+3, num_return_sequences=5)]})

@app.route('/paraphrase', methods=['GET'])
@cross_origin()
def paraphrase():
    input_value = request.args.get('q').strip()
    para_phrases = parrot.augment(input_phrase=input_value)
    return jsonify({'output' : [phrase for phrase in para_phrases if phrase != input_value]})


if __name__ == '__main__':
    app.run()