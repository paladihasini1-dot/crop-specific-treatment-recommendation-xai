Crop-Specific Treatment Recommendation System Using Explainable AI (XAI)

Project Overview

The Crop-Specific Treatment Recommendation System Using Explainable AI (XAI) is an AI-based application that identifies crop diseases from leaf images and provides suitable treatment recommendations.

The system uses deep learning for disease classification and Explainable AI techniques such as Grad-CAM to show the areas of the leaf that influenced the model's prediction.

Objectives

- Detect crop diseases from leaf images.
- Predict the disease using a deep learning model.
- Provide suitable treatment recommendations.
- Explain the prediction using Explainable AI.
- Help users understand why a particular disease was predicted.
- Provide an easy-to-use web interface.

Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- Flask
- HTML
- CSS
- JavaScript
- Grad-CAM
- Explainable AI

System Workflow

User uploads a crop leaf image.

1. Image preprocessing
2. Disease prediction using the trained deep learning model
3. Grad-CAM generates an explanation heatmap
4. The predicted disease is identified
5. Treatment recommendation is displayed
6. The result is presented through the web application

XAI

Grad-CAM (Gradient-weighted Class Activation Mapping) is used to visualize the important regions of the leaf image that contributed to the model's prediction.

This makes the prediction more understandable instead of providing only the disease name.

Project Features

- Crop leaf image upload
- Disease classification
- Treatment recommendation
- Grad-CAM visualization
- Explainable AI
- Web-based interface
- Multilingual result support

Project Structure

crop-specific-treatment-recommendation-xai/
│
├── app.py
├── predict.py
├── train_model.py
├── prepare_dataset.py
├── README.md
├── templates/
└── static/

Future Enhancements

- Support for more crops and diseases
- Mobile application
- Voice-based interaction
- Additional regional languages
- Improved disease classification accuracy
- Integration with real-time agricultural information

  Screenshots

Home Page

"Home Page" (Screenshot 2026-09-20 220305.png)
Disease Prediction
"Disease Prediction" (Screenshot 2026-09-20 220402.png)
Grad-CAM / XAI Explanation
"Grad-CAM Explanation" (Screenshot 2026-09-20 220341.png)
Treatment Recommendation
"Treatment Recommendation" (Screenshot 2026-09-20 220420.png)

Project Status

Academic Major Project – In Development
