
import json
import streamlit as st
import tensorflow as tf
from huggingface_hub import hf_hub_download
from config import REPO_ID, MODEL_FILE, DISEASE_JSON, LABELS_JSON


@st.cache_resource
def download_assets():
    model_path = hf_hub_download(REPO_ID, MODEL_FILE)
    disease_path = hf_hub_download(REPO_ID, DISEASE_JSON)
    labels_path = hf_hub_download(REPO_ID, LABELS_JSON)
    return model_path, disease_path, labels_path


@st.cache_resource
def load_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


@st.cache_resource
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
