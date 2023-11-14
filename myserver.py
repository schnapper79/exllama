from flask import Flask, request, jsonify
import json
from mymodel import ExllamaModel
import argparse, os
from waitress import serve

# Create the parser
parser = argparse.ArgumentParser(description='Flask app with model directory')

# Add an argument
parser.add_argument('-d','--directory', type=str, help='Path to the model directory',default='/workspace/models')

# Parse the arguments
args = parser.parse_args()


app = Flask(__name__)

# Assuming ExllamaModel is already defined as per your provided code
model = None
# Set the model directory in the app config
app.config['MODEL_DIRECTORY'] = args.directory

def get_subdirectories(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]


@app.route('/load_model', methods=['POST'])
def load_model():
    global model
    cfg = request.json

    # Extract the path to model from cfg and then remove it from cfg
    path_to_model = cfg.pop('path_to_model', None)
    if not path_to_model:
        return jsonify({"error": "Path to model is required"}), 400
   
    # cfg.setdefault('truncation_length', cfg['max_seq_len'] - 4)
    

    directory= os.path.join(app.config['MODEL_DIRECTORY'],path_to_model)
    if os.path.exists(directory) and os.path.isdir(directory):
        model = ExllamaModel.from_pretrained(directory, cfg)
        return jsonify({"message": "Model loaded successfully"}), 200
    else:
        return jsonify({"error": "Path to model does not exist"}), 404

    

@app.route('/generate', methods=['POST'])
def generate():
    if not model:
        return jsonify({"error": "Model is not loaded"}), 400

    state = request.json

    # Check if 'prompt' is provided, if not return an error
    if 'prompt' not in state:
        return jsonify({"error": "Prompt is required"}), 400

    prompt = state['prompt']

    # Set defaults for other state members
    state.setdefault('temperature', 0.96)
    state.setdefault('top_p', 0.65)
    state.setdefault('top_k', 40)
    state.setdefault('typical_p', 0.0)
    state.setdefault('repetition_penalty', 1.15)
    state.setdefault('repetition_penalty_range', 256)

    state.setdefault('ban_eos_token', false)
    state.setdefault('custom_token_bans', none)
    state.setdefault('guidance_scale', 1)
    state.setdefault('negative_prompt', none)
    state.setdefault('add_bos_token', false)
    
    state.setdefault('max_new_tokens',200)

    output = model.generate(prompt, state)
    return jsonify({"output": output}), 200

@app.route('/list_models', methods=['GET'])
def list_models():
    try:
        directory = app.config['MODEL_DIRECTORY']
        subdirectories = get_subdirectories(directory)
        return jsonify({"models": subdirectories}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=8080)
