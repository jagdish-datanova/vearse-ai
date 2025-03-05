from flask import Flask, request, jsonify
# from flask_pymongo import PyMongo
from pymongo import MongoClient
from openai import OpenAI
import json
import os
from werkzeug.utils import secure_filename
import re
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# MongoDB Configuration
database_url = os.getenv('DATABASE_URL')
mongo_client = MongoClient(database_url)

# Create (or get) the database
db = mongo_client["dialogue_memory"]

# Ensure the conversations collection exists
if "conversations" not in db.list_collection_names():
    db.create_collection("conversations")

# File Upload Directory
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# OpenAI API Key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)

# OpenAI Prompt Template
prompt_template = '''
You are an expert dialogue designer for interactive storytelling in video games. Your task is to generate custom dialogue using the provided query, chat history, uploaded file data and optional context.

### Chat History: {chat_history}

### Uploaded file Data: {uploaded_file_data}

### Context: {context}

### Query: {query}

JSON Input Template:
{{
"Label": {{"type": "label", "id": "a0", "description": "First Interaction"}},
"Condition": {{"type": "condition", "condition": "A", "target": "41"}},
"Item Gain": {{"type": "item_gain", "item": "flippers"}},
"End": {{"type": "end"}}
}}

Instructions:
- Start with a label (e.g., $a0: First Interaction) and write compelling descriptions.
- Use conditions to create branching paths. Each condition must target another label and provide immersive choices (e.g., “Yes | No” or “Help | Ignore”).
- Ensure all branches lead to another label, an item gain, or an end state.
- Include optional item_gain or loop conditions if applicable.
- Dialogue must feel logical and reflective of player choices.
- Output should be in JSON format without adding any extra tags and, symbols.

example Output:
  [{{
    "id": "a0",
    "dialogue": "The merchant waves at you. 'Come closer! I have an adventure to offer. Will you hear me out?'",
    "options": [
      {{
        "condition": "Accept",
        "dialogue": "'Splendid! Here's the quest.'",
        "target": "41"
      }},
      {{
        "condition": "Decline",
        "dialogue": "'Suit yourself. The opportunity may not come again.'",
        "target": "51"
      }}
    ]
  }},
  {{
    "id": "41",
    "dialogue": "'The golden amulet must be retrieved from the ancient ruins to the north. Will you take on the challenge?'",
    "options": [
      {{
        "condition": "Agree to help",
        "dialogue": "'You're a true hero. Good luck!'",
        "target": "83"
      }},
      {{
        "condition": "Change mind",
        "dialogue": "'Perhaps you're not the adventurer I thought you were.'",
        "target": "51"
      }}
    ]
  }},
  {{
    "id": "83",
    "dialogue": "You hand over the golden amulet. 'This will secure my fortune. You have my thanks!'",
    "reward": "Golden Amulet"
  }},
  {{
    "id": "51",
    "dialogue": "The merchant walks away, disappointed. 'I hope you reconsider next time.'"
  }}]
'''

def extract_json(text):
    """
    Robustly extract JSON from text, with support for complex nested structures.
    
    Args:
        text (str): The text potentially containing JSON.
    
    Returns:
        dict or list: Parsed JSON object or list, or None if extraction fails.
    """
    import json
    import re

    def clean_and_parse(text):
        # Remove any leading/trailing whitespace or non-JSON characters
        text = text.strip()
        
        # Remove potential markdown code block markers
        text = text.replace('```json', '').replace('```', '').strip()
        
        try:
            # First, try direct parsing
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try more aggressive parsing strategies
        try:
            # Find JSON-like content using regex
            json_match = re.search(r'\[.*\]|\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except (json.JSONDecodeError, AttributeError):
            pass

        # Attempt to find and parse the most promising JSON-like structure
        try:
            # Look for the most comprehensive JSON-like content
            match = re.search(r'(\[(?:\{[^{}]*\}(?:,)?)*\])', text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
        except (json.JSONDecodeError, AttributeError):
            pass

        # # Last resort: print debugging information
        # print("JSON Extraction Failed. Problematic Text:")
        # print(text)
        return None

    # Main extraction attempt
    result = clean_and_parse(text)
    
    # If extraction failed, try removing any leading/trailing text
    if result is None:
        # Try extracting JSON from between first and last valid JSON brackets
        match = re.search(r'(\[.*\])', text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

    return result

# Function to get or create a conversation for a user
def get_or_create_conversation(user_id):
    conversation = db.conversations.find_one({"user_id": user_id})
    if not conversation:
        db.conversations.insert_one({"user_id": user_id, "history": [], "file_paths": []})
        conversation = db.conversations.find_one({"user_id": user_id})
    return conversation

def get_summary_from_openai(user_id, query, context=None):
    """Generate a response using OpenAI and maintain MongoDB-based session memory"""
    try:
      
      conversation = get_or_create_conversation(user_id)
      
      chat_history = conversation["history"] if conversation and "history" in conversation else []
      
      # Ensure file_paths exists; otherwise, use an empty list
      file_paths = conversation["file_paths"] if conversation and "file_paths" in conversation else []

      uploaded_file_data = []
      for file_path in file_paths:
          if os.path.exists(file_path):
              with open(file_path, 'r', encoding='utf-8') as f:
                  uploaded_file_data.append(json.load(f))
      
      # Format chat history and context separately
      formatted_chat_history = json.dumps(chat_history, indent=2) if chat_history else "No previous chat history."
      formatted_context = json.dumps(context, indent=2) if context else "No additional context provided."
      uploaded_file_data = json.dumps(uploaded_file_data, indent=2) if uploaded_file_data else "No uploaded file data."

      # Prepare the final prompt with separated sections
      formatted_prompt = prompt_template.format(
          chat_history=formatted_chat_history,
          context=formatted_context,
          uploaded_file_data=uploaded_file_data,
          query=query
      )
      
      messages = [{"role": "system", "content": formatted_prompt}]
      
      response = client.chat.completions.create(
          model="gpt-4o", temperature=0.3, messages=messages
      )
      result = response.choices[0].message.content
      
      # Extract JSON from response
      extracted_json = extract_json(result)
      
      # Store updated conversation history in MongoDB
      new_entry = [
          {"role": "user", "content": query},
          {"role": "assistant", "content": extracted_json}
      ]
      
      db.conversations.update_one(
          {"user_id": user_id},
          {"$push": {"history": {"$each": new_entry}}}
          )

      return extracted_json if extracted_json else {"error": "No valid JSON found in response"}
    except Exception as e:
        return f"Error generating dialogue: {str(e)}"


@app.route('/generate-dialogue', methods=['POST'])
def generate_dialogue():
    """Generate dialogue from user query with optional file as context"""
    try:
      user_id = request.form.get('user_id') or request.json.get('user_id')
      query = request.form.get('input_data') or request.json.get('input_data')
      files = request.files.getlist('file')
      context = []

      if not user_id:
          return jsonify({"error": "Please Provide an user_id"}), 400
      
      if not query:
          return jsonify({"error": "Query is required. "}), 400
      
      user_folder =os.path.join(app.config["UPLOAD_FOLDER"], user_id)
      os.makedirs(user_folder, exist_ok=True)
      
      file_paths = []
      for file in files:
        if file and file.filename.endswith('.json'):
            filename = secure_filename(file.filename)
            file_path = os.path.join(user_folder, filename)
            file.save(file_path)
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                context.append(json_data)

            file_paths.append(file_path)

      
      # Update MongoDB with the list of file paths
      if file_paths:
        db.conversations.update_one(
          {"user_id": user_id},
          {"$push": {"file_paths": {"$each": file_paths}}},
          upsert = True
        )
      
      response = get_summary_from_openai(user_id, query, context)
      return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

@app.route('/clear-session', methods=['POST'])
def clear_session():
    """Clear the conversation history for a user in MongoDB"""
    try:
      data = request.json
      if not data or 'user_id' not in data:
          return jsonify({"error": "Missing user_id"}), 400

      user_id = data['user_id']
      conversation = db.conversations.find_one({"user_id": user_id})
      
      if conversation and "file_paths" in conversation:
        for file_path in conversation["file_paths"]:
          if os.path.exists(file_path):
            os.remove(file_path)
      
      # Remove user directory
      user_folder = os.path.join(app.config["UPLOAD_FOLDER"], user_id)
      if os.path.exists(user_folder):
        os.rmdir(user_folder)
        
      # Delete MongoDB entry
      db.conversations.delete_one({"user_id": user_id})

      return jsonify({"message": "Session cleared successfully"})
    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)