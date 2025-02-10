import os
import joblib
import pandas as pd
import re
import http.server
import socketserver
from geolocation_model import GeolocationModel
from nearby_services_model import NearbyServicesModel

def load_model_and_vectorizer(model_path, vectorizer_path):
    """Load the model and vectorizer from joblib files."""
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

def load_remedies_and_otc():
    """Load home remedies and OTC medicines from CSV files."""
    remedies_df = pd.read_csv('REMEDIES.csv')  # Ensure this file exists
    otc_df = pd.read_csv('Book1__OTC.csv')      # Ensure this file exists
    return remedies_df, otc_df

def get_remedies_for_disease(disease, remedies_df):
    """Get home remedies for the specified disease from the DataFrame."""
    remedies_row = remedies_df[remedies_df['DISEASE NAME'].str.lower().str.strip() == disease.lower().strip()]
    if not remedies_row.empty:
        remedies = remedies_row.iloc[0, 1:7].dropna().tolist()  # Get columns HOMEREMEDY1 to HOMEREMEDY6
        return remedies
    return []

def get_otc_for_disease(disease, otc_df):
    """Get OTC medicines for the specified disease from the DataFrame."""
    otc_row = otc_df[otc_df['Diseases'].str.lower().str.strip() == disease.lower().strip()]
    if not otc_row.empty:
        otc_medicines = otc_row.iloc[0, 1:5].dropna().tolist()  # Get columns OTC1 to OTC4
        return otc_medicines
    return []

def safe_input(prompt, default=""):
    """Wrapper for input() to handle EOFError."""
    try:
        return input(prompt).strip()
    except EOFError:
        return default  # Provide a fallback value

def process_user_input():
    """Main function to process user input and provide health-related suggestions."""
    # Load models and vectorizers
    symptom_model, symptom_vectorizer = load_model_and_vectorizer('naive_bayes_modelS.pkl.gz', 'tfidf_vectorizerS.pkl')
    geolocation_model = GeolocationModel()
    services_model = NearbyServicesModel(api_key=os.getenv("HERE_API_KEY"))

    # Load remedies and OTC medicines from CSV
    remedies_df, otc_df = load_remedies_and_otc()

    print("Hello! I am your health assistant.")

    # Loop until a valid symptom input is given
    while True:
        user_input = safe_input("Please tell me your symptoms (e.g., headache, fever, etc.): ", default="fever")

        # Check if input contains only numbers
        if re.fullmatch(r'\d+', user_input):
            print("Invalid input! Please enter valid symptoms (not numbers).")
            continue  # Ask again

        # Transform user input using TF-IDF vectorizer and use the model to predict symptoms
        user_input_vectorized = symptom_vectorizer.transform([user_input])
        matched_symptoms = symptom_model.predict(user_input_vectorized)

        if matched_symptoms:
            detected_symptoms = [symptom for symptom in matched_symptoms if isinstance(symptom, str)]
            if detected_symptoms:
                break  # Exit loop if valid symptoms are found
            else:
                print("Invalid input! Please enter recognizable symptoms.")
        else:
            print("I couldn't recognize those symptoms. Please try again.")

    disease = detected_symptoms[0]  # Assuming the first matched symptom is the disease
    print(f"I think you might have {disease}.")

    # Get home remedies
    remedies = get_remedies_for_disease(disease, remedies_df)
    if remedies:
        print("Here are some home remedies:")
        for remedy in remedies:
            print(f"- {remedy}")
    else:
        print("No home remedies found for this disease.")

    # Get OTC medicines
    otc_medicines = []  # Initialize to an empty list
    otc_choice = safe_input("Do you want to know about OTC medicines for this? (yes/no): ", default="no").lower()

    if otc_choice == 'yes':
        otc_medicines = get_otc_for_disease(disease, otc_df)
        if otc_medicines:
            print("Here are some OTC medicines you can try:")
            for otc in otc_medicines:
                print(f"- {otc}")
        else:
            print("No OTC medicines found.")
    elif otc_choice == 'no':
        print("No OTC medicines will be displayed.")
    else:
        print("Invalid input. Please enter 'yes' or 'no'.")

    # Get user's location for nearby medical services
    location = safe_input("Please provide your address or city for nearby medical services: ", default="New York")
    lat, lng = geolocation_model.get_geolocation(location)
    if lat and lng:
        services_model.get_nearby_services(lat, lng)
    else:
        print("Sorry, I couldn't find that location. Please try again.")

# Run the main function in the background
import threading
conversation_thread = threading.Thread(target=process_user_input)
conversation_thread.daemon = True
conversation_thread.start()

# Bind to a port to keep Render deployment active
PORT = int(os.environ.get("PORT", 8080))

class KeepAliveHandler(http.server.SimpleHTTPRequestHandler):
    """Simple HTTP server to keep Render service alive."""
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Service is running...")

with socketserver.TCPServer(("", PORT), KeepAliveHandler) as httpd:
    print(f"Serving at port {PORT}")
    httpd.serve_forever()


