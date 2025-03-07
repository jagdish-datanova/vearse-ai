from flask import Flask, request, jsonify
from flask_cors import CORS
from motor.motor_asyncio import AsyncIOMotorClient
import openai
import json
import os
from werkzeug.utils import secure_filename
import re
from dotenv import load_dotenv
import asyncio
import boto3
from botocore.exceptions import NoCredentialsError
import requests
from pydantic import BaseModel, Field
from typing import List, Optional

load_dotenv()

app = Flask(__name__)
CORS(app)

# MongoDB Configuration
database_url = os.getenv('DATABASE_URL')
mongo_client = AsyncIOMotorClient(database_url)

# Create (or get) the database
db = mongo_client["dialogue_memory"]

async def create_db_collection():
    existing_collections = await db.list_collection_names()
    if "conversations" not in existing_collections:
        await db.create_collection("conversations")

# OpenAI API Key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)

class DialogueOption(BaseModel):
    condition: Optional[str] = None
    dialogue: str
    target: str

class DialogueNode(BaseModel):
    id: str
    dialogue: str
    options: List[DialogueOption] = Field(default_factory=list)  # Default empty list
    reward: Optional[str] = None  # Optional field

class DialogueResponse(BaseModel):
    dialogues: List[DialogueNode] = Field(default_factory=list)  # Default empty list


# AWS S3 Configuration
AWS_REGION = os.getenv("AWS_REGION")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_BUCKET = os.getenv("AWS_BUCKET")
AWS_URL = f"https://{AWS_BUCKET}.s3.{AWS_REGION}.amazonaws.com"
AWS_PRESIGNED_URL_EXPIRES= int(os.getenv("AWS_PRESIGNED_URL_EXPIRES", 3600))

# Initialize S3 Client
s3_client = boto3.client(
    "s3",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

async def upload_file_to_s3(file, user_id):
  """Uploads a file to S3 and returns the presigned URL"""
  try:
    # Upload the file to S3
    filename= secure_filename(file.filename)
    s3_key = f"user_uploaded_files/{user_id}/{filename}"
    s3_client.upload_fileobj(file, AWS_BUCKET, s3_key, ExtraArgs={'ContentType': file.content_type})
    
    return s3_key

  except NoCredentialsError:
    return None

def generate_presigned_url(s3_key):
    """Generate a temporary pre-signed URL for accessing a private file."""
    try:
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': AWS_BUCKET, 'Key': s3_key},
            ExpiresIn=AWS_PRESIGNED_URL_EXPIRES
        )
        return presigned_url
    except Exception:
        return None

# - If the user input is a greeting (e.g., "hi", "hello", "welcome", "hey", "good morning"), respond with ->"How Vearse Dialogue Generator assists you today?"
# OpenAI Prompt Template
prompt_template = '''
- You are an expert dialogue designer for interactive storytelling in video games. Your task is to generate custom dialogue using the provided query, chat history, previous files data and uploaded files.

### Chat History: {chat_history}

### previous files data: {previsous_file_data}

### uploaded files data: {uploaded_file}

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

# Function to get or create a conversation for a user
async def get_or_create_conversation(user_id):

    # Create collection if not exist
    await create_db_collection()
    conversation = await db.conversations.find_one({"user_id": user_id})
    if not conversation:
        await db.conversations.insert_one({"user_id": user_id, "history": []})
        conversation = await db.conversations.find_one({"user_id": user_id})
    return conversation

async def get_summary_from_openai(user_id, query, context=None):
    """Generate a response using OpenAI and maintain MongoDB-based session memory"""
    try:
      
      conversation =await get_or_create_conversation(user_id)
      chat_history = conversation["history"] if conversation and "history" in conversation else []      
      uploaded_file_data = []
      # Extract file paths from history where file_path is not null
      file_keys = [entry["file_path"] for entry in chat_history if entry.get("file_path")]
      # Fetch JSON content from each S3 file using pre-signed URLs
      for s3_key in file_keys:
          try:
              presigned_url = generate_presigned_url(s3_key)
              if presigned_url:
                  response = requests.get(presigned_url)
                  if response.status_code == 200:
                      json_data = response.json()  # Parse JSON data
                      uploaded_file_data.append(json_data)
          except Exception as e:
              print(f"Failed to load file from {s3_key}: {e}")
      
      # Format chat history and context separately
      formatted_chat_history = json.dumps(chat_history, indent=2) if chat_history else "No previous chat history."
      uploaded_file = json.dumps(context, indent=2) if context else "No additional context provided."
      previsous_file_data = json.dumps(uploaded_file_data, indent=2) if uploaded_file_data else "No uploaded file data."

      # Prepare the final prompt with separated sections
      formatted_prompt = prompt_template.format(
          chat_history=formatted_chat_history,
          uploaded_file=uploaded_file,
          previsous_file_data=previsous_file_data,
          query=query
      )
      
      messages = [{"role": "system", "content": formatted_prompt}]
      
      response = await client.beta.chat.completions.parse(
                                                model="gpt-4o",
                                                temperature=0.3,
                                                messages=messages,
                                                response_format=DialogueResponse)
      result = response.choices[0].message.content
    #   print(f"Result: {result}")
      return result if result else {"error": "No valid JSON found in response"}
    except Exception as e:
        return f"Error generating dialogue: {str(e)}"

@app.route('/generate-dialogue', methods=['POST'])
async def generate_dialogue():
    """Generate dialogue from user query with optional file as context"""
    try:
      user_id = request.form.get('user_id') or (await request.json).get('user_id')
      query = request.form.get('input_data') or (await request.json).get('input_data')
      files = request.files.getlist('file')
      context = []
      file_path = None

      if not user_id:
          return jsonify({"error": "Please Provide an user_id"}), 400
      
      if not query:
          return jsonify({"error": "Query is required. "}), 400

      file_paths = []
      for file in files:
        if file and file.filename.endswith('.json'):
            file_content = file.read()
            json_data = json.loads(file_content)
            file.seek(0)
            file_url = await upload_file_to_s3(file, user_id)
            if file_url:
              file_paths.append(file_url)
              context.append(json_data)

      # Save user message with file_path (or null if no file)
      user_message = {
          "role": "user",
          "file_path": file_paths,  # Store file path if available, else null
          "content": query
      }
      
      response = await get_summary_from_openai(user_id, query, context)
      assistant_message = {
            "role": "assistant",
            "content": response
        }

      # Update MongoDB history
      await db.conversations.update_one(
          {"user_id": user_id},
          {"$push": {"history": {"$each": [user_message, assistant_message]}}},
          upsert=True
      )

      return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

@app.route('/get-user-files', methods=['POST'])
async def get_user_files():
    """Retrieve user's uploaded file URLs from MongoDB."""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        page = int(data.get('page', 1))
        items_per_page = 10

        if not user_id:
            return jsonify({"error": "Missing user_id"}), 400

        conversation = await db.conversations.find_one({"user_id": user_id})
        if not conversation or "history" not in conversation:
            return jsonify({
                "history": [],
                "page": 1,
                "total_pages": 0
            })
        chat_history = conversation.get("history", [])
        total_items = len(chat_history)
        total_pages = (total_items + items_per_page - 1) // items_per_page

        # Calculate pagination indexes
        start_idx = (page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, total_items)

        # Get paginated history
        paginated_history = chat_history[start_idx:end_idx]

        # Process file paths to generate proper URLs
        for entry in paginated_history:
            if entry.get("file_path") and isinstance(entry["file_path"], list):
                entry["file_path"] = [f"{AWS_URL}/{file}" for file in entry["file_path"]]

        # Include pagination info in the response
        response_data = {
            "history": paginated_history,
            "page": page,
            "total_pages": total_pages,
            "total_items": total_items,
            "items_per_page": items_per_page
        }

        return jsonify({"chat_history": response_data})

    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

@app.route('/clear-session', methods=['POST'])
async def clear_session():
    """Clear the conversation history for a user in MongoDB"""
    try:
      data = await request.json
      if not data or 'user_id' not in data:
          return jsonify({"error": "Missing user_id"}), 400

      user_id = data['user_id']
      conversation = await db.conversations.find_one({"user_id": user_id})
      # Delete associated S3 files from history
      if conversation and "history" in conversation:
          for entry in conversation["history"]:
              file_path = entry.get("file_path")
              if file_path:
                  s3_client.delete_object(Bucket=AWS_BUCKET, Key=file_path)

      # Delete MongoDB entry
      await db.conversations.delete_one({"user_id": user_id})

      return jsonify({"message": "Session cleared successfully"})
    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500
      
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

async def close_mongo_connection():
    """Explicitly close MongoDB connection when the server stops"""
    if mongo_client:
        await mongo_client.close()
        print("MongoDB connection closed")

if __name__ == '__main__':
    import asyncio
    from hypercorn.asyncio import serve
    from hypercorn.config import Config
    from asgiref.wsgi import WsgiToAsgi

    config = Config()
    config.bind = ["0.0.0.0:5001"]
    config.use_reloader = False
    
    # Add cleanup to Hypercorn's on_exit handler
    config.on_exit = [close_mongo_connection]

    try:
        # Wrap the Flask app and run the server
        asgi_app = WsgiToAsgi(app)
        asyncio.run(serve(asgi_app, config))
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    finally:
        # Ensure cleanup even if unexpected errors occur
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(close_mongo_connection())