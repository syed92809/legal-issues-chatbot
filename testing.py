getIn = "high"

# Print the first character of the entered word
print("You entered:", getIn)

# Convert the first character to uppercase and concatenate with the rest of the string
filtered = getIn[0].upper() + getIn[1:]
print(filtered)


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
