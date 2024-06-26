# import firebase_admin
# from firebase_admin import credentials, firestore
# from dotenv import load_dotenv
# import os
# import google.generativeai as genai
# import re

# # Load the .env file
# load_dotenv()

# # Access the API key
# api_key = os.getenv('API_KEY')

# # Initialize Firebase Admin SDK
# cred = credentials.Certificate("C:/Users/farru/PythonProjects/legal_chatbot/self-learning/lawyerfinder-86944-firebase-adminsdk-5llwt-15f2c84b6c.json")
# firebase_admin.initialize_app(cred)

# # Initialize Firestore
# db = firestore.client()

# def capitalize_first_letter(text):
#     return ' '.join(word.capitalize() for word in text.split())

# def get_lawyer_profiles(specialization=None, court=None, min_rating=None, max_price=None):
#     # Query the Firestore database for lawyer profiles
#     lawyers_ref = db.collection('lawyers_information')
#     query = lawyers_ref

#     if specialization:
#         # Fetch lawyer IDs with the required specialization
#         specialization = capitalize_first_letter(specialization)
#         specialization_docs = db.collection('lawyer_specialization').where('specializations', 'array_contains', specialization).stream()
#         specialization_ids = [doc.id for doc in specialization_docs]
#         if specialization_ids:
#             query = query.where('__name__', 'in', specialization_ids)

#     if court:
#         # Fetch lawyer IDs with the required court
#         court = capitalize_first_letter(court)
#         court_docs = db.collection('lawyer_court').where('selectedCourts', 'array_contains', court).stream()
#         court_ids = [doc.id for doc in court_docs]
#         if court_ids:
#             query = query.where('__name__', 'in', court_ids)

#     lawyers = []
#     for lawyer in query.stream():
#         lawyer_dict = lawyer.to_dict()
#         lawyer_id = lawyer.id

#         if max_price and int(lawyer_dict.get('lowPrice', 0)) > max_price:
#             continue

#         # Fetch specialization for the lawyer
#         spec_doc = db.collection('lawyer_specialization').document(lawyer_id).get()
#         if spec_doc.exists:
#             specializations = spec_doc.to_dict().get('specializations', [])
#             lawyer_dict['specialization'] = ', '.join(specializations) if specializations else 'N/A'
#         else:
#             lawyer_dict['specialization'] = 'N/A'

#         # Fetch rating for the lawyer
#         rating_docs = db.collection('lawyer_rating').where('lawyer_id', '==', lawyer_id).stream()
#         ratings = [doc.to_dict() for doc in rating_docs]
#         if ratings:
#             average_rating = sum(rating['rating'] for rating in ratings) / len(ratings)
#             lawyer_dict['rating'] = round(average_rating, 1)
#         else:
#             lawyer_dict['rating'] = 'N/A'

#         if min_rating and (lawyer_dict['rating'] == 'N/A' or lawyer_dict['rating'] < min_rating):
#             continue

#         lawyers.append(lawyer_dict)

#     return lawyers

# def format_lawyer_profiles(lawyers):
#     if not lawyers:
#         return "No lawyers found matching your criteria."
#     profile_text = "Here are some recommended lawyers:\n"
#     for lawyer in lawyers:
#         profile_text += f"Name: {lawyer.get('firstName', 'N/A')} {lawyer.get('lastName', 'N/A')}, "
#         profile_text += f"Specialization: {lawyer.get('specialization', 'N/A')}, "
#         profile_text += f"Rating: {lawyer.get('rating', 'N/A')}, "
#         profile_text += f"Description: {lawyer.get('description', 'N/A')}, "
#         profile_text += f"Country: {lawyer.get('country', 'N/A')}, "
#         profile_text += f"City: {lawyer.get('city', 'N/A')}, "
#         profile_text += f"Phone: {lawyer.get('phone', 'N/A')}\n"
#     return profile_text

# def parse_user_input(user_input):
#     specialization_keywords = [
#         'Admiralty Lawyer', 'Business Lawyer', 'Constitutional Lawyer', 'Environmental Lawyer', 'Property Lawyer', 
#         'Patent Lawyer', 'Healthcare Lawyer', 'Criminal Defense Lawyer', 'Family Lawyer', 'Personal Injury Lawyer', 
#         'Immigration Lawyer', 'Labor and Employment Lawyer', 'Intellectual Property Lawyer', 'Estate Planning Lawyer', 
#         'Real Estate Lawyer', 'Bankruptcy Lawyer', 'Tax Lawyer', 'Divorce Lawyer', 'Civil Rights Lawyer', 
#         'Corporate Lawyer', 'Entertainment Lawyer', 'Social Security Disability Lawyer'
#     ]
#     court_keywords = [
#         'Local/Municipal Courts', 'District Courts', 'Superior Courts', 'High Courts', 'Supreme Courts', 
#         'Constitutional Courts', 'Federal Courts', 'International Courts'
#     ]
#     rating_keywords = ['top rated', 'rating', 'above', 'greater than']
#     price_keywords = ['price range', 'below', 'under', 'less than']

#     specialization, court, min_rating, max_price = None, None, None, None

#     user_input_lower = user_input.lower()

#     # Check for specialization keywords
#     for keyword in specialization_keywords:
#         if keyword.lower() in user_input_lower:
#             specialization = keyword
#             break

#     # Check for court keywords
#     for keyword in court_keywords:
#         if keyword.lower() in user_input_lower:
#             court = keyword
#             break

#     # Check for rating keywords
#     rating_match = re.search(r'(\d+(\.\d+)?)\s*(star|rating|above|greater than)', user_input_lower)
#     if rating_match:
#         min_rating = float(rating_match.group(1))

#     # Check for price keywords
#     price_match = re.search(r'(\d+)\s*(dollars|usd|price range|below|under|less than)', user_input_lower)
#     if price_match:
#         max_price = int(price_match.group(1))

#     return specialization, court, min_rating, max_price

# if api_key:
#     # Initialize the API client with the API key
#     genai.configure(api_key=api_key)

#     model = genai.GenerativeModel('gemini-1.5-flash')

#     print("You can now chat with the legal information bot. Type 'exit' to end the chat.")
    
#     while True:
#         user_question = input("You: ")
        
#         if user_question.lower() in ['exit', 'quit', 'bye']:
#             print("Exiting the chat. Goodbye!")
#             break
        
#         # Capitalize the first letter of each word in the user's input
#         user_question_capitalized = capitalize_first_letter(user_question)
#         print("***************************\n",user_question_capitalized,"\n*********************************")
        
#         prompt = (
#             f"Pretend as a professional lawyer who has knowledge of all legal issues in the United States and can give educational advice. "
#             f"If the user asks for lawyer profiles, provide the relevant profiles from the database. "
#             f"Don't refer external websites or resources to the user, only convince the user to consult lawyers through our app. "
#             f"Don't recommend user to use our search or filter feature, they can only find lawyer by talking to you."
#             f"If user ask any other question other than legal issue or try to talk on any other topic than simply deny asking that question by saying I am not programmed to answer on this topic."
#             f"Question: {user_question_capitalized}"
#         )
        
#         response = model.generate_content(prompt)

#         # Parse user question for criteria
#         specialization, court, min_rating, max_price = parse_user_input(user_question_capitalized)

#         # Fetch and format lawyer profiles if criteria match
#         if specialization or court or min_rating or max_price:
#             lawyers = get_lawyer_profiles(specialization=specialization, court=court, min_rating=min_rating, max_price=max_price)
#             lawyer_profiles = format_lawyer_profiles(lawyers)
#             modified_response = response.text + "\n\n" + lawyer_profiles + "\nNote: This information is for educational purposes only and should not be considered legal advice."
#         else:
#             modified_response = response.text + "\n\nNote: This information is for educational purposes only and should not be considered legal advice."
        
#         print("Law GPT:", modified_response, "\n************************************************")

# else:
#     print("API Key not found. Please check your .env file.")



# import nltk
# import numpy as np
# import os
# import random
# import string
# from nltk.corpus import stopwords
# from nltk.corpus import wordnet
# from nltk.tokenize import word_tokenize, sent_tokenize
# from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# # Download necessary NLTK data
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')

# # Initialize lemmatizer and stop words
# lemmatizer = WordNetLemmatizer()
# stop_words = set(stopwords.words('english'))

# # Load initial learning data
# learning_data_file = 'learning_data.txt'
# if not os.path.exists(learning_data_file):
#     with open(learning_data_file, 'w') as f:
#         f.write("Hello!\nHi there!\nHow are you?\nI'm doing great, thank you!\n")

# with open(learning_data_file, 'r') as file:
#     learning_data = file.readlines()

# learning_data = [line.strip() for line in learning_data]

# def preprocess_text(text):
#     # Tokenization
#     tokens = word_tokenize(text.lower())
#     # Remove punctuation and stop words, and lemmatize
#     tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
#     return ' '.join(tokens)

# def get_response(user_input):
#     user_input = preprocess_text(user_input)
#     all_texts = learning_data + [user_input]
    
#     vectorizer = TfidfVectorizer().fit_transform(all_texts)
#     vectors = vectorizer.toarray()
    
#     cosine_matrix = cosine_similarity(vectors)
#     similarity_scores = cosine_matrix[-1][:-1]
    
#     if max(similarity_scores) < 0.1:
#         return "I don't understand that yet, but I'm learning!"
    
#     best_match_idx = np.argmax(similarity_scores)
#     return learning_data[best_match_idx]

# def update_learning_data(user_input, bot_response):
#     with open(learning_data_file, 'a') as file:
#         file.write(user_input + '\n')
#         file.write(bot_response + '\n')
#     learning_data.append(user_input)
#     learning_data.append(bot_response)

# def chatbot():
#     print("Chatbot: Hi! How can I help you today?")
#     while True:
#         user_input = input("You: ")
#         if user_input.lower() in ['exit', 'quit', 'bye']:
#             print("Chatbot: Goodbye!")
#             break
        
#         bot_response = get_response(user_input)
#         print(f"Chatbot: {bot_response}")
        
#         update_learning_data(user_input, bot_response)

# # Run the chatbot
# if __name__ == "__main__":
#     chatbot()
