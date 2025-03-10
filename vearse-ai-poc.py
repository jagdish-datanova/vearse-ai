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
from datetime import datetime

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
        
# Ensure indexes for faster queries
async def setup_indexes():
    await db.conversations.create_index([("user_id", 1), ("created_at", -1)])

# Ensure schema validation for MongoDB
async def setup_schema():
    validation_schema = {
        "bsonType": "object",
        "required": ["user_id", "role", "content", "created_at"],
        "properties": {
            "user_id": {"bsonType": "string"},
            "role": {"bsonType": "string", "enum": ["user", "assistant"]},
            "content": {"bsonType": "string"},
            "file_path": {"bsonType": "array", "items": {"bsonType": "string"}},
            "created_at": {"bsonType": "string"}
        }
    }
    await db.command("collMod", "conversations", validator={"$jsonSchema": validation_schema})

async def initialize_db():
    await setup_indexes()
    await setup_schema()

# OpenAI API Key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)

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

# OpenAI Prompt Template
prompt_template = '''
- You are a gaming story design expert. I will give you a prompt generated for GameUI. This prompt has dialogue for a game in specific format. User will ask for a specific request like changing the tone or storyline of the JSON. Your job is to provide feedback and improvement points for the json dialogue file. Dont correct the JSON yourself, just provide points of improvement.

### Chat History: {chat_history}

### previous files data: {previsous_file_data}

### uploaded files data: {uploaded_file}

### Query: {query}

'''
# print(f"prompt template: {prompt_template}")
# Function to get or create a conversation for a user
async def get_user_conversations(user_id):
    """Retrieve all conversation messages for a user, sorted by latest first."""
    conversations = await db.conversations.find({"user_id": user_id}).sort("created_at", -1).to_list(length=None)

    for entry in conversations:
        entry["_id"] = str(entry["_id"])  # Convert ObjectId to string
        if "created_at" in entry and isinstance(entry["created_at"], datetime):
            entry["created_at"] = entry["created_at"].isoformat()  # Convert datetime to ISO format string

    return conversations
  
async def get_user_files_paginated(user_id, page=1, limit=10):
    skip_count = (page - 1) * limit
    cursor = db.conversations.find({"user_id": user_id}).sort("created_at", -1).skip(skip_count).limit(limit)
    messages = await cursor.to_list(length=limit)
    for entry in messages:
        entry["_id"] = str(entry["_id"])
        if "created_at" in entry and isinstance(entry["created_at"], datetime):
            entry["created_at"] = entry["created_at"].isoformat()
        if entry.get("file_path") and isinstance(entry["file_path"], list):
            entry["file_path"] = [f"{AWS_URL}/{file}" for file in entry["file_path"]]
    total_items = await db.conversations.count_documents({"user_id": user_id})
    total_pages = (total_items + limit - 1) // limit
    return {"history": messages, "page": page, "total_pages": total_pages, "total_items": total_items, "items_per_page": limit}

async def get_summary_from_openai(user_id, query, context=None):
    """Generate a response using OpenAI and maintain MongoDB-based session memory"""
    try:
      
      conversation =await get_user_conversations(user_id)
      for entry in conversation:
            entry["_id"] = str(entry["_id"])
      chat_history = conversation if conversation else []
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
      # print(f"chat history: {formatted_chat_history}")
      uploaded_file = json.dumps(context, indent=2) if context else "No additional context provided."
      # print(f"uploaded files: {uploaded_file}")
      previsous_file_data = json.dumps(uploaded_file_data, indent=2) if uploaded_file_data else "No uploaded file data."
      # print(f"previous files data: {previsous_file_data}")

      
      # Prepare the final prompt with separated sections
      formatted_prompt = prompt_template.format(
          chat_history=formatted_chat_history,
          uploaded_file=uploaded_file,
          previsous_file_data=previsous_file_data,
          query=query
      )
      # print(f"formatted prompt: {formatted_prompt}")
      messages = [{"role": "system", "content": formatted_prompt}]
      
      response = await client.chat.completions.create(
                                                model="gpt-4o",
                                                temperature=0.3,
                                                messages=messages)
      result = response.choices[0].message.content
    #   print(f"Result: {result}")
      return result if result else {"error": "No valid JSON found in response"}
    except Exception as e:
        return f"Error generating dialogue: {str(e)}"

@app.route('/generate-dialogue', methods=['POST'])
async def generate_dialogue():
    """Generate dialogue from user query with optional file as context"""
    try:
      user_id = request.form.get('user_id')
      query = request.form.get('input_data')
      files = request.files.getlist('file')
      context = []

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
            "user_id": user_id,
            "role": "user",
            "file_path": file_paths,
            "content": query,
            "created_at": datetime.utcnow().isoformat() 
        }
      await db.conversations.insert_one(user_message)
      
      response = await get_summary_from_openai(user_id, query, context)
      assistant_message = {
            "user_id": user_id,
            "role": "assistant",
            "content": response,
            "created_at": datetime.utcnow().isoformat()
        }
      await db.conversations.insert_one(assistant_message)

      return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

@app.route('/get-user-files', methods=['POST'])
async def get_user_files():
    try:
        if not request.is_json:
            return jsonify({"error": "Expected JSON data"}), 400
        data = request.json
        user_id = data.get('user_id')
        page = int(data.get('page', 1))
        items_per_page = int(data.get('items_per_page', 10))
        if not user_id:
            return jsonify({"error": "Missing user_id"}), 400
        response = await get_user_files_paginated(user_id, page, items_per_page)
        return jsonify(response)
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

# if __name__ == '__main__':
#     import asyncio
#     from hypercorn.asyncio import serve
#     from hypercorn.config import Config
#     from asgiref.wsgi import WsgiToAsgi

#     config = Config()
#     config.bind = ["0.0.0.0:5001"]
#     config.use_reloader = False
    
#     # Add cleanup to Hypercorn's on_exit handler
#     config.on_exit = [close_mongo_connection]

#     try:
#         # Wrap the Flask app and run the server
#         asgi_app = WsgiToAsgi(app)
#         asyncio.run(serve(asgi_app, config))
#     except KeyboardInterrupt:
#         print("\nServer stopped by user")
#     finally:
#         # Ensure cleanup even if unexpected errors occur
#         loop = asyncio.get_event_loop()
#         if loop.is_running():
#             loop.create_task(close_mongo_connection())

if __name__ == '__main__':
    import asyncio
    from hypercorn.asyncio import serve
    from hypercorn.config import Config
    from asgiref.wsgi import WsgiToAsgi

    config = Config()
    config.bind = ["0.0.0.0:5001"]
    config.use_reloader = False
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(initialize_db())

    asgi_app = WsgiToAsgi(app)

    try:
        loop.run_until_complete(serve(asgi_app, config))
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    finally:
        if not loop.is_closed():
            loop.run_until_complete(close_mongo_connection())
            loop.close()
