import json
from flask import Flask, request, jsonify
from model import QuestionAnsweringModel
from constants import *

app = Flask(__name__)
app.config["DEBUG"] = False

question_answering_model = QuestionAnsweringModel()

@app.route('/', methods=['GET'])
def home():
    return "Homepage"

@app.route('/api/v1/question-answering', methods=['POST'])
def extract_answer():
	record = json.loads(request.data)
	question = record.get('question')
	passage = record.get('passage')
	model = record.get('model')

	if question is None or passage is None or model is None:
		return jsonify({'error': 'wrong request body format'})
	if model not in AVAILABLE_MODELS:
		return jsonify({'error': f'only accept models {AVAILABLE_MODELS}'})
	
	answers = question_answering_model.question_answer(question, passage, model)
	response = []
	for ans, score in answers:
		response.append({
			"answer": str(ans),
			"score": float(score)
		})
	return jsonify(response)

if __name__ == '__main__':
  	app.run(host='0.0.0.0', port=8080)