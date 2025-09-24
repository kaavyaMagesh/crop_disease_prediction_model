import os
import io
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras

# --- Keras Backend Initialization (CRITICAL) ---
os.environ["KERAS_BACKEND"] = "tensorflow"
tf.keras.utils.set_random_seed(42)

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app) 

# --- Global Variables & Paths ---
WEIGHTS_PATH = "crop_disease_weights.weights.h5" 
IMG_SIZE = (224, 224)

# --- Disease-to-Advice Mapping ---
ADVICE_DB = {
    "American Bollworm on Cotton": "Apply insecticides like spinosad or emamectin benzoate. Implement a crop rotation schedule to break the life cycle.",
    "Anthracnose on Cotton": "Use a fungicide with active ingredients such as azoxystrobin or pyraclostrobin. Improve air circulation and remove infected plant debris.",
    "Army worm": "Use biological controls like Bacillus thuringiensis (Bt) or insecticides such as chlorantraniliprole. Monitor fields regularly.",
    "Becterial Blight in Rice": "Use copper-based bactericides. Avoid excessive nitrogen fertilizer. Plant resistant varieties.",
    "Brownspot": "Use fungicides like Mancozeb. Manage nitrogen and potassium levels in the soil.",
    "Common_Rust": "Use a fungicide containing azoxystrobin or pyraclostrobin. Plant rust-resistant maize varieties.",
    "Cotton Aphid": "Use insecticidal soap or horticultural oil. For severe infestations, apply a systemic insecticide. Encourage natural predators like ladybugs.",
    "Flag Smut": "Use seed treatments with fungicides like carboxin. Plant resistant varieties and practice crop rotation.",
    "Gray Leaf Spot": "Apply a fungicide. Practice crop rotation and use residue management to reduce inoculum.",
    "Healthy Maize": "Your maize is healthy! No treatment needed. Keep up the good work.",
    "Healthy Wheat": "Your wheat is healthy! No treatment needed. Keep up the good work.",
    "Healthy cotton": "Your cotton is healthy! No treatment needed. Keep up the good work.",
    "Leaf Curl": "No direct chemical cure; focus on controlling the whitefly vector. Use insecticides to manage whitefly populations. Remove and destroy severely infected plants.",
    "Leaf smut": "Use seed treatments with fungicides like carboxin. Practice crop rotation.",
    "Mosaic sugarcane": "No cure; use disease-free planting material. Rogue out and destroy infected plants.",
    "RedRot sugarcane": "Use disease-free sets for planting. Plant resistant varieties and practice crop rotation.",
    "RedRust sugarcane": "Use fungicides and plant resistant sugarcane varieties.",
    "Rice Blast": "Apply a fungicide at early stages. Use resistant varieties and avoid excessive nitrogen.",
    "Sugarcane Healthy": "Your sugarcane is healthy! No treatment needed. Keep up the good work.",
    "Tungro": "No chemical cure for the disease itself; focus on controlling the green leafhopper vector using insecticides.",
    "Wheat Brown leaf Rust": "Apply fungicides. Plant rust-resistant wheat varieties.",
    "Wheat Stem fly": "Use systemic insecticides as a seed or soil treatment.",
    "Wheat aphid": "Use biological controls like ladybugs. For severe cases, apply insecticides.",
    "Wheat black rust": "Use fungicides. Plant rust-resistant varieties, as this rust can be very destructive.",
    "Wheat leaf blight": "Apply fungicides. Practice good sanitation by removing crop residue.",
    "Wheat mite": "Use miticides. Proper irrigation and management of weeds can help.",
    "Wheat powdery mildew": "Apply fungicides. Plant resistant varieties. Improve air circulation.",
    "Wheat scab": "Apply fungicides during flowering. Use crop rotation and manage residue.",
    "Wheat Yellow_Rust": "Apply fungicides at the first sign of disease. Plant resistant varieties.",
    "Wilt": "Improve soil drainage. Use resistant varieties and practice crop rotation. Remove and destroy infected plants.",
    "Yellow Rust Sugarcane": "Use fungicides and plant resistant varieties.",
    "bacterial blight in Cotton": "Use copper-based bactericides. Avoid excessive nitrogen fertilizer. Plant resistant varieties.",
    "bollrot on Cotton": "Apply fungicides, especially during wet conditions. Ensure proper plant spacing and balanced fertilizer application.",
    "bollworm on Cotton": "Use insecticides like pyrethroids or carbamates. Monitor for egg laying and larva presence.",
    "cotton mealy bug": "Use horticultural oils or neem oil. For severe cases, use systemic insecticides. Release parasitic wasps.",
    "cotton whitefly": "Apply insecticides that target whiteflies. Use yellow sticky traps to monitor and reduce populations.",
    "maize ear rot": "Use resistant hybrids. Ensure proper fertilization to avoid plant stress.",
    "maize fall armyworm": "Use insecticides like chlorantraniliprole. Implement biological controls using parasitic wasps.",
    "maize stem borer": "Use systemic insecticides. Promote natural enemies and practice crop rotation.",
    "pink bollworm in cotton": "Use mating disruption with pheromone lures. Apply insecticides in conjunction with other controls.",
    "red cotton bug": "Use contact insecticides. Clean up plant debris to remove overwintering sites.",
    "thirps on cotton": "Use insecticidal sprays with ingredients like spinosad or imidacloprid. Use blue sticky traps to monitor and control adult populations."
}

# --- Model Loading and Prediction Function ---
def load_model():
    """
    Re-builds the model architecture and loads the saved weights.
    Returns the compiled model object.
    """
    try:
        print("Starting model loading process...")
        
        # We've hardcoded the number of classes to match the weights file
        # The error message says the weights have 45 classes
        NUM_CLASSES = 45 

        base_model = tf.keras.applications.MobileNetV2(
            weights='imagenet', 
            include_top=False, 
            input_shape=(*IMG_SIZE, 3)
        )
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        for layer in base_model.layers[-20:]:
            layer.trainable = True

        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        output = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
        model = tf.keras.Model(inputs=base_model.input, outputs=output)
        
        # Check if the weights file exists
        if not os.path.exists(WEIGHTS_PATH):
            print(f"❌ Error: Weights file not found at {WEIGHTS_PATH}")
            return None

        model.load_weights(WEIGHTS_PATH)
        print("✅ Model weights loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ A fatal error occurred during model loading:")
        print(traceback.format_exc())
        return None

MODEL = load_model()
# Make sure this matches the hardcoded number above
CLASS_NAMES = list(ADVICE_DB.keys())

# --- Pre-processing Function ---
def preprocess_image(image_bytes):
    """
    Converts image bytes to a TensorFlow tensor, resizes, and normalizes it.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize(IMG_SIZE)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error during image preprocessing: {e}")
        return None

# --- Main Prediction Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    if MODEL is None:
        return jsonify({"error": "Model not loaded. Check server logs."}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    image_bytes = file.read()
    processed_image = preprocess_image(image_bytes)
    
    if processed_image is None:
        return jsonify({"error": "Failed to process image"}), 400

    predictions = MODEL.predict(processed_image)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    advice = ADVICE_DB.get(predicted_class_name, "No specific advice found for this disease.")

    response = {
        "disease": predicted_class_name,
        #"confidence": float(predictions[0][predicted_class_index]),
        "advice": advice
    }
    
    return jsonify(response)

if __name__ == '__main__':
    if MODEL is not None:
        print("Flask app is starting...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Flask app will not start because the model failed to load.")