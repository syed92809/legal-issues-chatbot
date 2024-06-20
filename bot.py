import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv
import os
import google.generativeai as genai
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the .env file
load_dotenv()

# Access the API key
api_key = os.getenv('API_KEY')

# Initialize Firebase Admin SDK
cred_path = "C:/Users/farru/PythonProjects/legal_chatbot/self-learning/lawyerfinder-86944-firebase-adminsdk-5llwt-fbbcf19d2a.json"

try:
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
except Exception as e:
    print(f"Error initializing Firebase Admin SDK: {e}")
    exit()

def capitalize_first_letter(text):
    return ' '.join(word.capitalize() for word in text.split())

def normalize_keyword(keyword):
    if keyword.endswith('s'):
        return keyword[:-1].lower()
    return keyword.lower()

def get_lawyer_profiles(specialization=None, court=None, min_rating=None, max_price=None):
    try:
        # Query the Firestore database for lawyer profiles
        lawyers_ref = db.collection('lawyers_information')
        query = lawyers_ref

        if specialization:
            # Fetch lawyer IDs with the required specialization
            specialization = capitalize_first_letter(specialization)
            specialization_docs = db.collection('lawyer_specialization').where('specializations', 'array_contains', specialization).stream()
            specialization_ids = [doc.id for doc in specialization_docs]
            if specialization_ids:
                query = query.where('__name__', 'in', specialization_ids)

        if court:
            # Fetch lawyer IDs with the required court
            court = capitalize_first_letter(court)
            court_docs = db.collection('lawyer_court').where('selectedCourts', 'array_contains', court).stream()
            court_ids = [doc.id for doc in court_docs]
            if court_ids:
                query = query.where('__name__', 'in', court_ids)

        lawyers = []
        for lawyer in query.stream():
            lawyer_dict = lawyer.to_dict()
            lawyer_id = lawyer.id

            if max_price and int(lawyer_dict.get('lowPrice', 0)) > max_price:
                continue

            # Fetch specialization for the lawyer
            spec_doc = db.collection('lawyer_specialization').document(lawyer_id).get()
            if spec_doc.exists:
                specializations = spec_doc.to_dict().get('specializations', [])
                lawyer_dict['specialization'] = ', '.join(specializations) if specializations else 'N/A'
            else:
                lawyer_dict['specialization'] = 'N/A'

            # Fetch rating for the lawyer
            rating_docs = db.collection('lawyer_rating').where('lawyer_id', '==', lawyer_id).stream()
            ratings = [doc.to_dict() for doc in rating_docs]
            if ratings:
                average_rating = sum(rating['rating'] for rating in ratings) / len(ratings)
                lawyer_dict['rating'] = round(average_rating, 1)
            else:
                lawyer_dict['rating'] = 'N/A'

            if min_rating and (lawyer_dict['rating'] == 'N/A' or lawyer_dict['rating'] < min_rating):
                continue

            lawyers.append(lawyer_dict)

        return lawyers
    except Exception as e:
        print(f"Error fetching lawyer profiles: {e}")
        return []

def format_lawyer_profiles(lawyers):
    if not lawyers:
        return "No lawyers found matching your criteria."
    profile_text = "Here are some recommended lawyers:\n"
    for lawyer in lawyers:
        profile_text += f"Name: {lawyer.get('firstName', 'N/A')} {lawyer.get('lastName', 'N/A')}, "
        profile_text += f"Specialization: {lawyer.get('specialization', 'N/A')}, "
        profile_text += f"Rating: {lawyer.get('rating', 'N/A')}, "
        profile_text += f"Description: {lawyer.get('description', 'N/A')}, "
        profile_text += f"Country: {lawyer.get('country', 'N/A')}, "
        profile_text += f"City: {lawyer.get('city', 'N/A')}, "
        profile_text += f"Phone: {lawyer.get('phone', 'N/A')}\n"
    return profile_text

def find_best_match(user_input, keywords):
    # Combine user input with keywords for vectorization
    texts = keywords + [user_input]

    # Vectorize the texts
    vectorizer = TfidfVectorizer().fit_transform(texts)
    vectors = vectorizer.toarray()

    # Calculate cosine similarity between the user input and all keywords
    cosine_matrix = cosine_similarity(vectors[-1].reshape(1, -1), vectors[:-1])
    similar_indices = cosine_matrix.argsort()[0][-1]

    return keywords[similar_indices], cosine_matrix[0, similar_indices]

def parse_user_input_cosine(user_input, specialization_keywords, court_keywords):
    best_specialization_match, best_specialization_similarity = find_best_match(user_input, specialization_keywords)
    best_court_match, best_court_similarity = find_best_match(user_input, court_keywords)
    
    # Set thresholds for similarity to ensure only relevant matches are considered
    similarity_threshold = 0.1  # Adjust this threshold based on your needs

    specialization = best_specialization_match if best_specialization_similarity >= similarity_threshold else None
    court = best_court_match if best_court_similarity >= similarity_threshold else None

    # Extract ratings and prices as in the original code
    user_input_lower = user_input.lower()
    rating_match = re.search(r'(\d+(\.\d+)?)\s*(star|rating|above|greater than)', user_input_lower)
    if rating_match:
        min_rating = float(rating_match.group(1))
    else:
        min_rating = None

    price_match = re.search(r'(\d+)\s*(dollars|usd|price range|below|under|less than)', user_input_lower)
    if price_match:
        max_price = int(price_match.group(1))
    else:
        max_price = None

    return specialization, court, min_rating, max_price

if api_key:
    # Initialize the API client with the API key
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel('gemini-1.5-flash')

    print("You can now chat with the legal information bot. Type 'exit' to end the chat.")
    
    specialization_keywords = [
        'Admiralty Lawyer', 'Business Lawyer', 'Constitutional Lawyer', 'Environmental Lawyer', 'Property Lawyer', 
        'Patent Lawyer', 'Healthcare Lawyer', 'Criminal Defense Lawyer', 'Family Lawyer', 'Personal Injury Lawyer', 
        'Immigration Lawyer', 'Labor and Employment Lawyer', 'Intellectual Property Lawyer', 'Estate Planning Lawyer', 
        'Real Estate Lawyer', 'Bankruptcy Lawyer', 'Tax Lawyer', 'Divorce Lawyer', 'Civil Rights Lawyer', 
        'Corporate Lawyer', 'Entertainment Lawyer', 'Social Security Disability Lawyer'
    ]

    court_keywords = [
        'Local/Municipal Courts', 'District Courts', 'Superior Courts', 'High Courts', 'Supreme Courts', 
        'Constitutional Courts', 'Federal Courts', 'International Courts'
    ]

    while True:
        user_question = input("You: ")
        
        if user_question.lower() in ['exit', 'quit', 'bye']:
            print("Exiting the chat. Goodbye!")
            break
        
        # Capitalize the first letter of each word in the user's input
        user_question_capitalized = capitalize_first_letter(user_question)
        print("***************************\n", user_question_capitalized, "\n*********************************")
        
        prompt = (
            f"Pretend as a professional lawyer who has knowledge of all legal issues in the United States and can give educational advice. "
            f"If the user asks for lawyer profiles, provide the relevant profiles from the database. "
            f"Don't refer external websites or resources to the user, only convince the user to consult lawyers through our app. "
            f"Don't recommend user to use our search or filter feature, they can only find lawyer by talking to you."
            f"If user ask any other question other than legal issue or try to talk on any other topic than simply deny asking that question by saying I am not programmed to answer on this topic."
            f"Question: {user_question_capitalized}"
        )
        
        response = model.generate_content(prompt)

        # Check if the user is asking for a lawyer before using cosine similarity
        if 'lawyer' in user_question.lower() or 'lawyers' in user_question.lower():
            # Parse user question for criteria
            specialization, court, min_rating, max_price = parse_user_input_cosine(user_question_capitalized, specialization_keywords, court_keywords)

            # Fetch and format lawyer profiles if criteria match
            if specialization or court or min_rating or max_price:
                try:
                    lawyers = get_lawyer_profiles(specialization=specialization, court=court, min_rating=min_rating, max_price=max_price)
                    lawyer_profiles = format_lawyer_profiles(lawyers)
                    modified_response = response.text + "\n\n" + lawyer_profiles + "\nNote: This information is for educational purposes only and should not be considered legal advice."
                except Exception as e:
                    modified_response = response.text + "\n\nError fetching lawyer profiles: " + str(e) + "\nNote: This information is for educational purposes only and should not be considered legal advice."
            else:
                modified_response = response.text + "\n\nNote: This information is for educational purposes only and should not be considered legal advice."
        else:
            modified_response = response.text + "\n\nNote: This information is for educational purposes only and should not be considered legal advice."
        
        print("Law GPT:", modified_response, "\n************************************************")

else:
    print("API Key not found. Please check your .env file.")
