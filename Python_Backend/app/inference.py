

import numpy as np
from preprocessing import preprocess


def predict(image, interpreter, input_details, output_details, labels, disease_info):
    x = preprocess(image)

    interpreter.set_tensor(input_details[0]["index"], x)
    interpreter.invoke()

    probs = interpreter.get_tensor(output_details[0]["index"])[0]

    top_idx = int(np.argmax(probs))
    confidence = float(probs[top_idx])

    label = labels[top_idx]
    info = disease_info.get(label, {"cause": "N/A", "cure": "N/A"})

    top3 = np.argsort(probs)[-3:][::-1]

    return label, confidence, info, probs, top3
