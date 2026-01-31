Ethiopian Crop Recommendation System 🌾

This project focuses on empowering smallholder farmers in Ethiopia by providing data-driven, localized crop group recommendations using machine learning. Agriculture employs 70% of the Ethiopian population, yet many farmers struggle with crop selection due to varying soil nutrients, climate conditions, and altitude zones.

🚀 Overview

The system utilizes a real-world Ethiopian dataset to classify optimal crop groups based on soil chemistry and climate data. It addresses the lack of personalized recommendations for farmers and extension workers in diverse regions, from the highlands to the lowlands.

📊 Dataset Highlights

Source: Real Ethiopian dataset (Alemu, 2024, Mendeley Data
)

Size: ~6,000+ processed samples

Features:

Soil: Nitrogen (N), Phosphorous (P), Potassium (K), pH, Zinc (Zn), Sulfur (S)

Climate: Temperature, humidity, rainfall

Engineering: Altitude derived from surface pressure

Target Classes: 4 major crop groups:

Major Cereals

Cereals

Pulses

Specialty Crops

🛠️ Technical Pipeline

Preprocessing: Outlier clipping, handling class imbalance (grouping 12 rare classes into 4), data augmentation using SMOTE

Scaling: StandardScaler normalization

Models Evaluated: Logistic Regression, Decision Tree, Random Forest, BalancedRandomForest, XGBoost

Deployment: FastAPI backend with a static HTML UI for interactive recommendations

📈 Model Performance

Selected Model: XGBoost + SMOTE (best balance of Macro F1 scores)

Metrics:

Test Accuracy: ~0.80

Macro F1 Score: ~0.65

Validation Accuracy: 0.801

Key Findings: Temperature, soil moisture, and rainfall are the top predictors for crop suitability

⚠️ Limitations

Data Imbalance: Major Cereals (like Teff) dominate the data (32%), while rare classes remain challenging despite SMOTE

Geographic Bias: Data limited to specific regions of Ethiopia

Climate Data: Uses NASA MERRA-2 reanalysis rather than direct ground measurements

🔮 Future Roadmap

Integrate yield prediction (regression) alongside crop recommendations

Develop a mobile application for direct field use by farmers

Add Amharic language support to improve local accessibility

Incorporate ground-truth climate validation



visit on https://huggingface.co/spaces/dawit-kassa/crop-recommendation-ethiopia


<img width="1915" height="926" alt="image" src="https://github.com/user-attachments/assets/713dc745-d153-42e7-abed-06f6da21c13a" />

