
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_swagger_ui import get_swaggerui_blueprint
from transformers import pipeline
import time

app = Flask(__name__)
CORS(app)

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


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
    Endpoint to summarize an article.
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
        do_sample = data.get("do_sample", False)


        start_time = time.time()
        summary = summarizer(article, max_length=max_length, min_length=min_length, do_sample=do_sample)
        end_time = time.time()


        time_taken = round(end_time - start_time, 2)
        return jsonify({
            "summary": summary[0]['summary_text'],
            "time_taken_seconds": time_taken
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
