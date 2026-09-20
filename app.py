from flask import Flask, render_template, request, jsonify
import os
import uuid
import asyncio
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
import cv2
import edge_tts

from treatment_data import treatment_data


# ============================================================
# FLASK SETUP
# ============================================================

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "uploads"
)

GRADCAM_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "gradcam"
)

AUDIO_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "audio"
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "rice_disease_model.keras"
)

print("Loading trained model...")

model = load_model(MODEL_PATH)

print("Model loaded successfully!")
print("Model name:", model.name)


# ============================================================
# CLASS NAMES
# ============================================================

class_names = [
    "bacterial_leaf_blight",
    "bacterial_leaf_streak",
    "bacterial_panicle_blight",
    "blast",
    "brown_spot",
    "dead_heart",
    "downy_mildew",
    "hispa",
    "normal",
    "tungro"
]


# ============================================================
# FIND MOBILENETV2
# ============================================================

base_model = None
base_index = None

for i, layer in enumerate(model.layers):

    if "mobilenet" in layer.name.lower():

        base_model = layer
        base_index = i
        break

    if hasattr(layer, "layers"):

        for sublayer in layer.layers:

            if "mobilenet" in sublayer.name.lower():

                base_model = layer
                base_index = i
                break

        if base_model is not None:
            break


if base_model is None:

    print(
        "ERROR: MobileNetV2 base model not found!"
    )

else:

    print(
        "Base model found:",
        base_model.name
    )


# ============================================================
# FIND LAST CONVOLUTIONAL LAYER
# ============================================================

last_conv_layer = None

if base_model is not None:

    for layer in reversed(base_model.layers):

        if isinstance(
            layer,
            tf.keras.layers.Conv2D
        ):

            last_conv_layer = layer
            break


if last_conv_layer is None:

    print(
        "ERROR: No Conv2D layer found!"
    )

else:

    print(
        "Grad-CAM layer:",
        last_conv_layer.name
    )


# ============================================================
# CREATE INTERMEDIATE MOBILENET MODEL
# ============================================================

base_grad_model = None

if (
    base_model is not None
    and last_conv_layer is not None
):

    try:

        base_grad_model = Model(
            inputs=base_model.input,
            outputs=[
                last_conv_layer.output,
                base_model.output
            ]
        )

        print(
            "MobileNet Grad-CAM model created successfully!"
        )

    except Exception as e:

        print(
            "MobileNet Grad-CAM creation error:",
            repr(e)
        )


# ============================================================
# GRAD-CAM FUNCTION
# ============================================================

def generate_gradcam(
    img_array,
    predicted_class
):

    if base_grad_model is None:

        print(
            "Grad-CAM model is not available."
        )

        return None

    try:

        # ----------------------------------------------------
        # Input image
        # ----------------------------------------------------

        input_tensor = tf.cast(
            img_array,
            tf.float32
        )

        # ----------------------------------------------------
        # Run layers before MobileNetV2
        # ----------------------------------------------------

        x = input_tensor

        for i in range(base_index):

            layer = model.layers[i]

            x = layer(
                x,
                training=False
            )

        # ----------------------------------------------------
        # Gradient calculation
        # ----------------------------------------------------

        with tf.GradientTape() as tape:

            conv_outputs, base_output = (
                base_grad_model(
                    x,
                    training=False
                )
            )

            tape.watch(conv_outputs)

            # Continue through layers
            # after MobileNetV2

            predictions = base_output

            for i in range(
                base_index + 1,
                len(model.layers)
            ):

                layer = model.layers[i]

                predictions = layer(
                    predictions,
                    training=False
                )

            class_score = predictions[
                :,
                predicted_class
            ]

        # ----------------------------------------------------
        # Calculate gradients
        # ----------------------------------------------------

        grads = tape.gradient(
            class_score,
            conv_outputs
        )

        if grads is None:

            print(
                "Gradients are None."
            )

            return None

        # ----------------------------------------------------
        # Average gradients
        # ----------------------------------------------------

        pooled_grads = tf.reduce_mean(
            grads,
            axis=(1, 2)
        )

        # ----------------------------------------------------
        # Generate heatmap
        # ----------------------------------------------------

        conv_output = conv_outputs[0]

        pooled_grad = pooled_grads[0]

        heatmap = tf.reduce_sum(
            conv_output * pooled_grad,
            axis=-1
        )

        heatmap = tf.maximum(
            heatmap,
            0
        )

        maximum = tf.reduce_max(
            heatmap
        )

        if maximum > 0:

            heatmap = (
                heatmap / maximum
            )

        heatmap = heatmap.numpy()

        # ----------------------------------------------------
        # Resize heatmap
        # ----------------------------------------------------

        heatmap = cv2.resize(
            heatmap,
            (224, 224)
        )

        heatmap = np.uint8(
            255 * heatmap
        )

        # ----------------------------------------------------
        # Create colored heatmap
        # ----------------------------------------------------

        heatmap_color = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET
        )

        # ----------------------------------------------------
        # Original image
        # ----------------------------------------------------

        original = image.img_to_array(
            image.load_img(
                current_uploaded_image,
                target_size=(224, 224)
            )
        )

        original = np.uint8(
            original
        )

        original_bgr = cv2.cvtColor(
            original,
            cv2.COLOR_RGB2BGR
        )

        # ----------------------------------------------------
        # Overlay heatmap
        # ----------------------------------------------------

        superimposed = cv2.addWeighted(
            original_bgr,
            0.6,
            heatmap_color,
            0.4,
            0
        )

        # ----------------------------------------------------
        # Save Grad-CAM image
        # ----------------------------------------------------

        filename = (
            "gradcam_"
            + uuid.uuid4().hex
            + ".jpg"
        )

        filepath = os.path.join(
            GRADCAM_FOLDER,
            filename
        )

        cv2.imwrite(
            filepath,
            superimposed
        )

        print(
            "Grad-CAM generated successfully!"
        )

        return (
            "gradcam/"
            + filename
        )

    except Exception as e:

        print(
            "Grad-CAM error:",
            repr(e)
        )

        return None


# ============================================================
# EDGE TTS VOICES
# ============================================================

VOICE_MAP = {

    "English":
        "en-IN-NeerjaNeural",

    "Telugu":
        "te-IN-ShrutiNeural",

    "Hindi":
        "hi-IN-SwaraNeural"
}


# ============================================================
# GENERATE AUDIO
# ============================================================

async def generate_audio(
    text,
    language,
    output_file
):

    voice = VOICE_MAP.get(
        language,
        "en-IN-NeerjaNeural"
    )

    communicate = edge_tts.Communicate(
        text,
        voice
    )

    await communicate.save(
        output_file
    )


# ============================================================
# VOICE API
# ============================================================

@app.route(
    "/generate_voice",
    methods=["POST"]
)
def generate_voice():

    try:

        data = request.get_json()

        if data is None:

            return jsonify({
                "error":
                    "No voice data received."
            }), 400

        text = data.get(
            "text",
            ""
        )

        language = data.get(
            "language",
            "English"
        )

        if not text:

            return jsonify({
                "error":
                    "No text provided."
            }), 400

        if language not in VOICE_MAP:

            language = "English"

        filename = (
            "voice_"
            + uuid.uuid4().hex
            + ".mp3"
        )

        filepath = os.path.join(
            AUDIO_FOLDER,
            filename
        )

        asyncio.run(
            generate_audio(
                text,
                language,
                filepath
            )
        )

        print(
            "Voice generated:",
            language
        )

        return jsonify({
            "audio_url":
                "/static/audio/"
                + filename
        })

    except Exception as e:

        print(
            "Voice generation error:",
            repr(e)
        )

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# PREDICTION
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    global current_uploaded_image

    try:

        # ----------------------------------------------------
        # Check uploaded file
        # ----------------------------------------------------

        if "leaf_image" not in request.files:

            return (
                "No image uploaded."
            )

        file = request.files[
            "leaf_image"
        ]

        if file.filename == "":

            return (
                "Please select an image."
            )

        # ----------------------------------------------------
        # Language
        # ----------------------------------------------------

        language = request.form.get(
            "language",
            "English"
        )

        # ----------------------------------------------------
        # Save image
        # ----------------------------------------------------

        extension = os.path.splitext(
            file.filename
        )[1]

        filename = (
            "leaf_"
            + uuid.uuid4().hex
            + extension
        )

        filepath = os.path.join(
            UPLOAD_FOLDER,
            filename
        )

        file.save(filepath)

        current_uploaded_image = filepath

        print(
            "Image saved:",
            filepath
        )

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        img = image.load_img(
            filepath,
            target_size=(224, 224)
        )

        img_array = image.img_to_array(
            img
        )

        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        # ----------------------------------------------------
        # MobileNetV2 preprocessing
        # ----------------------------------------------------

        processed_image = (
            tf.keras.applications
            .mobilenet_v2
            .preprocess_input(
                img_array.copy()
            )
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        predictions = model.predict(
            processed_image,
            verbose=0
        )

        predicted_class = int(
            np.argmax(
                predictions[0]
            )
        )

        confidence = float(
            predictions[0][
                predicted_class
            ] * 100
        )

        disease = class_names[
            predicted_class
        ]

        print(
            "Disease:",
            disease
        )

        print(
            "Confidence:",
            round(
                confidence,
                2
            ),
            "%"
        )

        # ----------------------------------------------------
        # Generate Grad-CAM
        # ----------------------------------------------------

        gradcam_image = (
            generate_gradcam(
                processed_image,
                predicted_class
            )
        )

        # ----------------------------------------------------
        # Treatment information
        # ----------------------------------------------------

        if disease in treatment_data:

            info = treatment_data[
                disease
            ]

        else:

            info = {

                "English": {

                    "name":
                        disease,

                    "reason":
                        "The disease was identified from the uploaded rice leaf image.",

                    "treatment":
                        "Consult an agricultural expert for suitable treatment.",

                    "prevention":
                        "Maintain field sanitation and monitor the crop regularly."
                },

                "Telugu": {

                    "name":
                        disease,

                    "reason":
                        "అప్‌లోడ్ చేసిన వరి ఆకు చిత్రం ఆధారంగా వ్యాధిని గుర్తించాం.",

                    "treatment":
                        "సరైన చికిత్స కోసం వ్యవసాయ నిపుణుడిని సంప్రదించండి.",

                    "prevention":
                        "పొలాన్ని పరిశుభ్రంగా ఉంచి పంటను క్రమం తప్పకుండా పరిశీలించండి."
                },

                "Hindi": {

                    "name":
                        disease,

                    "reason":
                        "अपलोड की गई धान की पत्ती की छवि के आधार पर रोग की पहचान की गई है।",

                    "treatment":
                        "उचित उपचार के लिए कृषि विशेषज्ञ से सलाह लें।",

                    "prevention":
                        "खेत की स्वच्छता बनाए रखें और फसल की नियमित निगरानी करें।"
                }
            }

        # ----------------------------------------------------
        # Select language
        # ----------------------------------------------------

        if language not in info:

            language = "English"

        selected_info = info[
            language
        ]

        # ----------------------------------------------------
        # Create voice text
        # ----------------------------------------------------

        disease_name_for_voice = (
            selected_info.get(
                "name",
                disease
            )
        )

        reason_for_voice = (
            selected_info.get(
                "reason",
                ""
            )
        )

        treatment_for_voice = (
            selected_info.get(
                "treatment",
                ""
            )
        )

        prevention_for_voice = (
            selected_info.get(
                "prevention",
                ""
            )
        )

        if language == "Telugu":

            voice_text = (

                f"వ్యాధి: "
                f"{disease_name_for_voice}. "

                f"నమ్మక స్థాయి: "
                f"{round(confidence, 2)} శాతం. "

                f"అంచనా వెనుక కారణం: "
                f"{reason_for_voice}. "

                f"సిఫార్సు చేసిన చికిత్స: "
                f"{treatment_for_voice}. "

                f"నివారణ చర్యలు: "
                f"{prevention_for_voice}."
            )

        elif language == "Hindi":

            voice_text = (

                f"रोग: "
                f"{disease_name_for_voice}। "

                f"विश्वास स्तर: "
                f"{round(confidence, 2)} प्रतिशत। "

                f"पहचान का कारण: "
                f"{reason_for_voice}। "

                f"अनुशंसित उपचार: "
                f"{treatment_for_voice}। "

                f"बचाव के उपाय: "
                f"{prevention_for_voice}।"
            )

        else:

            voice_text = (

                f"Disease: "
                f"{disease_name_for_voice}. "

                f"Confidence: "
                f"{round(confidence, 2)} percent. "

                f"Reason: "
                f"{reason_for_voice}. "

                f"Recommended treatment: "
                f"{treatment_for_voice}. "

                f"Prevention: "
                f"{prevention_for_voice}."
            )

        # ----------------------------------------------------
        # Image URLs
        # ----------------------------------------------------

        original_image = (
            "/static/uploads/"
            + filename
        )

        gradcam_url = None

        if gradcam_image is not None:

            gradcam_url = (
                "/static/"
                + gradcam_image
            )

        # ----------------------------------------------------
        # Result page
        # ----------------------------------------------------

        return render_template(

            "result.html",

            disease=disease,

            confidence=round(
                confidence,
                2
            ),

            language=language,

            disease_name=selected_info.get(
                "name",
                disease
            ),

            reason=selected_info.get(
                "reason",
                ""
            ),

            treatment=selected_info.get(
                "treatment",
                ""
            ),

            prevention=selected_info.get(
                "prevention",
                ""
            ),

            voice_text=voice_text,

            original_image=original_image,

            gradcam_image=gradcam_url
        )

    except Exception as e:

        print(
            "Prediction error:",
            repr(e)
        )

        return f"""
        <html>

        <body>

        <h2>Prediction Error</h2>

        <p>{str(e)}</p>

        <br>

        <a href="/">Go Back</a>

        </body>

        </html>
        """


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print("")

    print(
        "=========================================="
    )

    print(
        " Rice Disease XAI Web Application"
    )

    print(
        "=========================================="
    )

    print(
        "Open: http://127.0.0.1:5000"
    )

    print("")

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )