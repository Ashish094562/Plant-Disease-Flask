from flask import Flask, request, jsonify
from PIL import Image
import io
from model_loader import download_assets, load_model, load_json
from inference import predict
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

# Load everything once at startup
MODEL_PATH, DISEASE_PATH, LABELS_PATH = download_assets()
interpreter = load_model(MODEL_PATH)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

LABELS = load_json(LABELS_PATH)
disease_list = load_json(DISEASE_PATH)
DISEASE_INFO = {d["name"]: d for d in disease_list}

@app.route("/health")
def health():
    return {"status": "ok"}

@app.route("/predict", methods=["POST"])
def predict_disease():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image = Image.open(io.BytesIO(file.read()))

    label, confidence, info, probs, top3 = predict(
        image,
        interpreter,
        input_details,
        output_details,
        LABELS,
        DISEASE_INFO
    )

    response = {
        "disease": label,
        "confidence": float(confidence),
        "cause": info["cause"],
        "cure": info["cure"],
        "top_predictions": [
            {
                "label": LABELS[int(idx)],
                "probability": float(probs[int(idx)])
            }
            for idx in top3
        ]
    }

    return jsonify(response)


if __name__ == "__main__":
    app.run()

