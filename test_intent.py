import pandas as pd
import requests
from sklearn.metrics import classification_report
import time

def test_intent_classification():
    # Configuration
    API_URL = "http://127.0.0.1:8000/chat"
    CSV_FILE = "IntentDetection.csv"
    SIMILARITY_THRESHOLD = 0.5
    
    # Load the dataset
    df = pd.read_csv(CSV_FILE)
    print(f"Loaded {len(df)} prompts")
    
    # Lists to store results
    true_labels = []
    predicted_labels = []
    
    print(f"\nTesting {len(df)} prompts...")
    print("-" * 50)
    
    # Process each prompt
    for idx, row in df.iterrows():
        prompt = row['Prompt']
        true_category = row['Category']
        
        print(f"Processing {idx+1}/{len(df)}: {prompt[:50]}...")
        
        # Send request to FastAPI
        try:
            payload = {
                "prompt": prompt,
                "similarity_threshold": SIMILARITY_THRESHOLD
            }
            
            response = requests.post(API_URL, json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                predicted_category = data['response'].strip()
                
                # Check if prediction is correct
                is_correct = predicted_category.lower() == true_category.lower()
                status = "✅" if is_correct else "❌"
                print(f"  {status} True: {true_category} | Predicted: {predicted_category}")
                
                # Store for classification report
                true_labels.append(true_category)
                predicted_labels.append(predicted_category)
                
            else:
                print(f"  🔥 API Error: {response.status_code}")
                true_labels.append(true_category)
                predicted_labels.append("API_ERROR")
                
        except Exception as e:
            print(f"  🔥 Request failed: {e}")
            true_labels.append(true_category)
            predicted_labels.append("REQUEST_ERROR")
        
        # Small delay to avoid overwhelming the server
        time.sleep(0.5)
    
    # Generate results
    print("\n" + "="*60)
    print("CLASSIFICATION RESULTS")
    print("="*60)
    
    # Calculate accuracy
    correct_count = sum(1 for t, p in zip(true_labels, predicted_labels) if t.lower() == p.lower())
    total_tests = len(true_labels)
    accuracy = correct_count / total_tests if total_tests > 0 else 0
    
    print(f"Total Prompts Tested: {total_tests}")
    print(f"Correct Predictions: {correct_count}")
    print(f"Incorrect Predictions: {total_tests - correct_count}")
    print(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    # Classification report
    print(f"\nDETAILED CLASSIFICATION REPORT:")
    print("-" * 40)
    try:
        report = classification_report(true_labels, predicted_labels, zero_division=0)
        print(report)
    except Exception as e:
        print(f"Error generating report: {e}")
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'Prompt': df['Prompt'],
        'True_Category': true_labels,
        'Predicted_Category': predicted_labels
    })
    results_df.to_csv('classification_results2.csv', index=False)
    print(f"\nResults saved to: classification_results.csv")

if __name__ == "__main__":
    test_intent_classification()