from dotenv import load_dotenv
import os
import google.generativeai as genai

# Load the .env file
load_dotenv()

# Access the API key
api_key = os.getenv('API_KEY')

if api_key:
    # Initialize the API client with the API key
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel('gemini-1.5-flash')

    print("You can now chat with the legal information bot. Type 'exit' to end the chat.")
    
    while True:
        user_question = input("User: ")
        
        if user_question.lower() == 'exit':
            print("Exiting the chat. Goodbye!")
            break
        
        prompt = (
            f"Pretend as a professional lawyer who has knowledge of all legal issues in the United States and can give educational advice. "
            f"Question: {user_question}"
        )
        
        response = model.generate_content(prompt)
        
        # Add custom text to the end of the response
        modified_response = response.text + "\n\nNote: This information is for educational purposes only and should not be considered legal advice."
        
        print("Gemini:", modified_response,"\n************************************************")

else:
    print("API Key not found. Please check your .env file.")
