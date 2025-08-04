"""
Demo script showing how to use the trained models for prediction and explanation
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_models():
    """Load all trained models and preprocessing objects"""
    import os
    
    print("Loading models...")
    
    # Get the base directory (project root)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'models', 'model_artifacts')
    
    try:
        tabular_model = joblib.load(os.path.join(models_dir, 'tabular_model.joblib'))
        multimodal_model = joblib.load(os.path.join(models_dir, 'multimodal_model.joblib'))
        preprocessor = joblib.load(os.path.join(models_dir, 'preprocessor.joblib'))
        metadata = joblib.load(os.path.join(models_dir, 'metadata.joblib'))
        
        print("✅ Models loaded successfully")
        return tabular_model, multimodal_model, preprocessor, metadata
    
    except FileNotFoundError as e:
        print(f"❌ Model files not found: {e}")
        print("Please run the training notebook first to create the models.")
        return None, None, None, None

def create_sample_property():
    """Create a sample property for demonstration"""
    return {
        'accommodates': 4,
        'bedrooms': 2,
        'beds': 2,
        'bathrooms_numeric': 1.0,
        'minimum_nights': 2,
        'maximum_nights': 365,
        'availability_365': 300,
        'number_of_reviews': 25,
        'review_scores_rating': 4.5,
        'calculated_host_listings_count': 1,
        'amenities_count': 15,
        'has_wifi': 1,
        'has_kitchen': 1,
        'has_parking': 0,
        'has_pool': 0,
        'is_superhost_numeric': 0,
        'neighbourhood_cleansed': 'Downtown',
        'room_type': 'Entire home/apt',
        'property_type': 'Apartment'
    }

def predict_price(models, property_data, reviews_text="Great location, clean apartment"):
    """Make price predictions using both models"""
    
    tabular_model, multimodal_model, preprocessor, metadata = models
    
    if not all(models):
        return None
    
    # Create sample data matching training format
    sample_data = {}
    for feature in metadata['feature_names']:
        if feature in property_data:
            sample_data[feature] = property_data[feature]
        elif 'has_' in feature or 'is_' in feature:
            sample_data[feature] = 0
        elif 'count' in feature:
            sample_data[feature] = 1
        elif 'rate' in feature or 'ratio' in feature:
            sample_data[feature] = 1.0
        else:
            sample_data[feature] = 0.0
    
    # Calculate derived features
    sample_data['availability_rate'] = sample_data.get('availability_365', 300) / 365
    sample_data['price_per_person'] = 100 / sample_data.get('accommodates', 4)  # Assume $100 current price
    sample_data['beds_per_bedroom'] = sample_data.get('beds', 2) / max(sample_data.get('bedrooms', 1), 1)
    
    # Create DataFrame
    X_sample = pd.DataFrame([sample_data])
    
    # Make predictions
    tabular_pred = tabular_model.predict(X_sample)[0]
    multimodal_pred = multimodal_model.predict(X_sample, [reviews_text])[0]
    
    # Convert back from log if needed
    y_skewness = metadata.get('y_skewness', 0)
    if abs(y_skewness) > 1:
        tabular_pred = np.expm1(tabular_pred)
        multimodal_pred = np.expm1(multimodal_pred)
    
    return {
        'tabular_prediction': tabular_pred,
        'multimodal_prediction': multimodal_pred,
        'improvement': multimodal_pred - tabular_pred,
        'improvement_percent': ((multimodal_pred - tabular_pred) / tabular_pred) * 100
    }

def explain_prediction(models, property_data, reviews_text="Great location, clean apartment"):
    """Get explanation for a prediction"""
    
    tabular_model, multimodal_model, preprocessor, metadata = models
    
    if not all(models):
        return None
    
    # Create sample data (same as predict_price)
    sample_data = {}
    for feature in metadata['feature_names']:
        if feature in property_data:
            sample_data[feature] = property_data[feature]
        elif 'has_' in feature or 'is_' in feature:
            sample_data[feature] = 0
        elif 'count' in feature:
            sample_data[feature] = 1
        elif 'rate' in feature or 'ratio' in feature:
            sample_data[feature] = 1.0
        else:
            sample_data[feature] = 0.0
    
    # Calculate derived features
    sample_data['availability_rate'] = sample_data.get('availability_365', 300) / 365
    sample_data['price_per_person'] = 100 / sample_data.get('accommodates', 4)
    sample_data['beds_per_bedroom'] = sample_data.get('beds', 2) / max(sample_data.get('bedrooms', 1), 1)
    
    # Create DataFrame
    X_sample = pd.DataFrame([sample_data])
    
    # Get explanation
    try:
        explanation = multimodal_model.explain_prediction(X_sample.iloc[0], reviews_text)
        return explanation
    except Exception as e:
        print(f"Error getting explanation: {e}")
        return None

def main():
    """Main demo function"""
    
    print("🏠 Airbnb Price Predictor Demo")
    print("=" * 40)
    
    # Load models
    models = load_models()
    if not all(models):
        return
    
    # Create sample property
    property_data = create_sample_property()
    reviews_text = "Amazing stay! The apartment was clean, spacious, and in a great location. Host was very responsive and helpful. Would definitely stay again!"
    
    print("\n📊 Sample Property Details:")
    print(f"  • Accommodates: {property_data['accommodates']} guests")
    print(f"  • Bedrooms: {property_data['bedrooms']}")
    print(f"  • Bathrooms: {property_data['bathrooms_numeric']}")
    print(f"  • Room Type: {property_data['room_type']}")
    print(f"  • Property Type: {property_data['property_type']}")
    print(f"  • Reviews: {property_data['number_of_reviews']}")
    print(f"  • Rating: {property_data['review_scores_rating']}")
    
    print(f"\n📝 Sample Review Text:")
    print(f"  '{reviews_text}'")
    
    # Make predictions
    print("\nMaking Predictions...")
    predictions = predict_price(models, property_data, reviews_text)
    
    if predictions:
        print(f"\n💰 Price Predictions:")
        print(f"  • Tabular Model: ${predictions['tabular_prediction']:.2f}")
        print(f"  • Multimodal Model: ${predictions['multimodal_prediction']:.2f}")
        print(f"  • Text Improvement: ${predictions['improvement']:.2f} ({predictions['improvement_percent']:.1f}%)")
    
    # Get explanation
    print("\n🔍 Getting Explanation...")
    explanation = explain_prediction(models, property_data, reviews_text)
    
    if explanation and 'tabular' in explanation:
        print(f"\n📈 Top 5 Features Affecting Price:")
        
        # Sort features by absolute impact
        sorted_features = sorted(explanation['tabular'].items(), 
                               key=lambda x: abs(x[1]), reverse=True)[:5]
        
        for feature, impact in sorted_features:
            impact_sign = "+" if impact > 0 else ""
            print(f"  • {feature.replace('_', ' ').title()}: {impact_sign}{impact:.3f}")
    
    if explanation and 'predictions' in explanation:
        pred_info = explanation['predictions']
        print(f"\nPrediction Breakdown:")
        print(f"  • Tabular Prediction: ${pred_info.get('tabular_prediction', 0):.2f}")
        print(f"  • Final Prediction: ${pred_info.get('final_prediction', 0):.2f}")
        print(f"  • Text Contribution: ${pred_info.get('text_contribution', 0):.2f}")
    
    print("\nDemo completed!")
    print("\nNext Steps:")
    print("  • Run 'streamlit run streamlit_app.py' for the interactive web interface")
    print("  • Check the training notebook for more detailed analysis")
    print("  • Modify this script to test your own properties")

if __name__ == "__main__":
    main()
