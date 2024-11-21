from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_swagger_ui import get_swaggerui_blueprint
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
import time

app = Flask(__name__)
CORS(app)

# Load tokenizer and model
tokenizer = PreTrainedTokenizerFast.from_pretrained("ainize/bart-base-cnn")
model = BartForConditionalGeneration.from_pretrained("ainize/bart-base-cnn")

SWAGGER_URL = '/api-docs'
API_URL = '/summaryswag.yaml'  # This must match the route serving the YAML file

swaggerui_blueprint = get_swaggerui_blueprint(SWAGGER_URL, API_URL)
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

@app.route("/summaryswag.yaml")
def serve_swagger_yaml():
    """
    Serve the Swagger YAML file.
    """
    return send_from_directory(".", "summaryswag.yaml", mimetype="text/yaml")

@app.route('/summarize', methods=['POST'])
def summarize():
    """
    Endpoint to summarize an article using BartForConditionalGeneration.
    """
    try:
        data = request.json
        article = data.get("article", "")
        article = article.replace('\n', ' ').replace('\t', ' ')
        if not article:
            return jsonify({"error": "No article text provided"}), 400

        # Optional parameters
        max_length = data.get("max_length", 130)
        min_length = data.get("min_length", 30)
        num_beams = data.get("num_beams", 4)

        # Tokenize input
        input_ids = tokenizer.encode(article, return_tensors="pt")

        start_time = time.time()
        # Generate summary
        summary_ids = model.generate(
            input_ids=input_ids,
            bos_token_id=model.config.bos_token_id,
            eos_token_id=model.config.eos_token_id,
            length_penalty=2.0,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        end_time = time.time()

        time_taken = round(end_time - start_time, 2)
        return jsonify({
            "summary": summary,
            "time_taken_seconds": time_taken
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
