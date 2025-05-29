import os
import logging
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import google.generativeai as genai
import json

load_dotenv()

app = Flask(__name__)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Directory for storing conversations 
CONVERSATIONS_DIR = 'conversations'
if not os.path.exists(CONVERSATIONS_DIR):
    os.makedirs(CONVERSATIONS_DIR)
    logging.info(f"Created conversations directory at {CONVERSATIONS_DIR}")

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
WASENDER_API_TOKEN = os.getenv('WASENDER_API_TOKEN')
WASENDER_API_URL = "https://wasenderapi.com/api/send-message"

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    logging.error("GEMINI_API_KEY not found in environment variables. The application might not work correctly.")

@app.errorhandler(Exception)
def handle_global_exception(e):
    """Global handler for unhandled exceptions."""
    logging.error(f"Unhandled Exception: {e}", exc_info=True)
    return jsonify(status='error', message='An internal server error occurred.'), 500

# --- Load Persona ---
PERSONA_FILE_PATH = 'persona.json'
PERSONA_DESCRIPTION = "You are a helpful assistant." # Default persona
PERSONA_NAME = "Assistant"
BASE_PROMPT = "You are a helpful and concise AI assistant replying in a WhatsApp chat. Do not use Markdown formatting. Keep your answers short, friendly, and easy to read. If your response is longer than 3 lines, split it into multiple messages using \n every 3 lines. Each \n means a new WhatsApp message. Avoid long paragraphs or unnecessary explanations."

try:
    with open(PERSONA_FILE_PATH, 'r') as f:
        persona_data = json.load(f)
        custom_description = persona_data.get('description', PERSONA_DESCRIPTION)
        base_prompt = persona_data.get('base_prompt', BASE_PROMPT)
        PERSONA_DESCRIPTION = f"{base_prompt}\n\n{custom_description}"
        PERSONA_NAME = persona_data.get('name', PERSONA_NAME)
    logging.info(f"Successfully loaded persona: {PERSONA_NAME}")
except FileNotFoundError:
    logging.warning(f"Persona file not found at {PERSONA_FILE_PATH}. Using default persona.")
except json.JSONDecodeError:
    logging.error(f"Error decoding JSON from {PERSONA_FILE_PATH}. Using default persona.")
except Exception as e:
    logging.error(f"An unexpected error occurred while loading persona: {e}. Using default persona.")
# --- End Load Persona ---

def load_conversation_history(user_id):
    """Loads conversation history for a given user_id."""
    file_path = os.path.join(CONVERSATIONS_DIR, f"{user_id}.json")
    try:
        with open(file_path, 'r') as f:
            history = json.load(f)
            # Ensure history is a list of dictionaries (pairs of user/assistant messages)
            if isinstance(history, list) and all(isinstance(item, dict) and 'role' in item and 'parts' in item for item in history):
                return history
            else:
                logging.warning(f"Invalid history format in {file_path}. Starting fresh.")
                return []
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from {file_path}. Starting fresh.")
        return []
    except Exception as e:
        logging.error(f"Unexpected error loading history from {file_path}: {e}")
        return []

def save_conversation_history(user_id, history):
    """Saves conversation history for a given user_id."""
    file_path = os.path.join(CONVERSATIONS_DIR, f"{user_id}.json")
    try:
        with open(file_path, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        logging.error(f"Error saving conversation history to {file_path}: {e}")
def split_message(text, max_lines=3, max_chars_per_line=100):
    """Split a long message into smaller chunks for better WhatsApp readability."""
    # First split by existing newlines
    paragraphs = text.split('\\n')
    chunks = []
    current_chunk = []
    current_line_count = 0
    
    for paragraph in paragraphs:
        # Split long paragraphs into smaller lines
        if len(paragraph) > max_chars_per_line:
            words = paragraph.split()
            current_line = []
            current_length = 0
            
            for word in words:
                if current_length + len(word) + 1 <= max_chars_per_line:
                    current_line.append(word)
                    current_length += len(word) + 1
                else:
                    if current_line_count >= max_lines:
                        chunks.append('\n'.join(current_chunk))
                        current_chunk = []
                        current_line_count = 0
                    current_chunk.append(' '.join(current_line))
                    current_line_count += 1
                    current_line = [word]
                    current_length = len(word)
            
            if current_line:
                if current_line_count >= max_lines:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                    current_line_count = 0
                current_chunk.append(' '.join(current_line))
                current_line_count += 1
        else:
            if current_line_count >= max_lines:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_line_count = 0
            current_chunk.append(paragraph)
            current_line_count += 1
    
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    return chunks

def get_gemini_response(message_text, conversation_history=None):
    """Generates a response from Gemini using the google-generativeai library, including conversation history."""
    if not GEMINI_API_KEY:
        logging.error("Gemini API key is not configured.")
        return "Sorry, I'm having trouble connecting to my brain right now (API key issue)."

    try:
        # Using Gemini 2.0 Flash model with system instruction for persona
        model_name = 'gemini-2.0-flash'
        model = genai.GenerativeModel(model_name, system_instruction=PERSONA_DESCRIPTION)
        
        logging.info(f"Sending prompt to Gemini (system persona active): {message_text[:200]}...")

        if conversation_history:
            # Use chat history if available
            chat = model.start_chat(history=conversation_history)
            response = chat.send_message(message_text)
        else:
            # For first message with no history
            response = model.generate_content(message_text)

        # Extract the text from the response
        if response and hasattr(response, 'text') and response.text:
            return response.text.strip()
        elif response and response.candidates:
            # Fallback if .text is not directly available but candidates are
            try:
                return response.candidates[0].content.parts[0].text.strip()
            except (IndexError, AttributeError, KeyError) as e:
                logging.error(f"Error parsing Gemini response candidates: {e}. Response: {response}")
                return "I received an unusual response structure from Gemini. Please try again."
        else:
            logging.error(f"Gemini API (google-generativeai) returned an empty or unexpected response: {response}")
            return "I received an empty or unexpected response from Gemini. Please try again."

    except Exception as e:
        logging.error(f"Error calling Gemini API with google-generativeai: {e}", exc_info=True)
        return "I'm having trouble processing that request with my AI brain. Please try again later."

def send_whatsapp_message(recipient_number, message_content, message_type='text', media_url=None):
    """Sends a message via WaSenderAPI. Supports text and media messages."""
    if not WASENDER_API_TOKEN:
        logging.error("WaSender API token is not set. Please check .env file.")
        return False

    headers = {
        'Authorization': f'Bearer {WASENDER_API_TOKEN}',
        'Content-Type': 'application/json'
    }
    
    # Sanitize recipient_number to remove "@s.whatsapp.net"
    if recipient_number and "@s.whatsapp.net" in recipient_number:
        formatted_recipient_number = recipient_number.split('@')[0]
    else:
        formatted_recipient_number = recipient_number

    payload = {
        'to': formatted_recipient_number
    }

    if message_type == 'text':
        payload['text'] = message_content
    elif message_type == 'image' and media_url:
        payload['imageUrl'] = media_url
        if message_content:
            payload['text'] = message_content 
    elif message_type == 'video' and media_url:
        payload['videoUrl'] = media_url
        if message_content:
            payload['text'] = message_content
    elif message_type == 'audio' and media_url:
        payload['audioUrl'] = media_url
    elif message_type == 'document' and media_url:
        payload['documentUrl'] = media_url
        if message_content:
            payload['text'] = message_content
    else:
        if message_type != 'text':
             logging.error(f"Media URL is required for message type '{message_type}'.")
             return False
        logging.error(f"Unsupported message type or missing content/media_url: {message_type}")
        return False
    
    logging.debug(f"Attempting to send WhatsApp message. Payload: {payload}")

    try:
        response = requests.post(WASENDER_API_URL, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        logging.info(f"Message sent to {recipient_number}. Response: {response.json()}")
        return True
    except requests.exceptions.RequestException as e:
        status_code = e.response.status_code if e.response is not None else "N/A"
        response_text = e.response.text if e.response is not None else "N/A"
        logging.error(f"Error sending WhatsApp message to {recipient_number} (Status: {status_code}): {e}. Response: {response_text}")
        if status_code == 422:
            logging.error("WaSenderAPI 422 Error: This often means an issue with the payload (e.g., device_id, 'to' format, or message content/URL). Check the payload logged above and WaSenderAPI docs.")
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred while sending WhatsApp message: {e}")
        return False

@app.route('/webhook', methods=['POST'])
def webhook():
    """Handles incoming WhatsApp messages via webhook."""
    data = request.json
    logging.info(f"Received webhook data (first 200 chars): {str(data)[:200]}")

    try:
        if data.get('event') == 'messages.upsert' and data.get('data') and data['data'].get('messages'):
            message_info = data['data']['messages']
            
            # Check if it's a message sent by the bot itself
            if message_info.get('key', {}).get('fromMe'):
                logging.info(f"Ignoring self-sent message: {message_info.get('key', {}).get('id')}")
                return jsonify({'status': 'success', 'message': 'Self-sent message ignored'}), 200

            sender_number = message_info.get('key', {}).get('remoteJid')
            
            incoming_message_text = None
            message_type = 'unknown'

            # Extract message content based on message structure
            if message_info.get('message'):
                msg_content_obj = message_info['message']
                if 'conversation' in msg_content_obj:
                    incoming_message_text = msg_content_obj['conversation']
                    message_type = 'text'
                elif 'extendedTextMessage' in msg_content_obj and 'text' in msg_content_obj['extendedTextMessage']:
                    incoming_message_text = msg_content_obj['extendedTextMessage']['text']
                    message_type = 'text'

            if message_info.get('messageStubType'):
                stub_params = message_info.get('messageStubParameters', [])
                logging.info(f"Received system message of type {message_info['messageStubType']} from {sender_number}. Stub params: {stub_params}")
                return jsonify({'status': 'success', 'message': 'System message processed'}), 200

            if not sender_number:
                logging.warning("Webhook received message without sender information.")
                return jsonify({'status': 'error', 'message': 'Incomplete sender data'}), 400

            # Sanitize sender_number to use as a filename
            safe_sender_id = "".join(c if c.isalnum() else '_' for c in sender_number)

            if message_type == 'text' and incoming_message_text:
                logging.info(f"Processing text message from {sender_number} ({safe_sender_id}): {incoming_message_text}")
                
                # Load conversation history
                conversation_history = load_conversation_history(safe_sender_id)
                
                # Get Gemini's reply, passing the history
                gemini_reply = get_gemini_response(incoming_message_text, conversation_history)
                
                if gemini_reply:
                    # Split the response into chunks and send them sequentially
                    message_chunks = split_message(gemini_reply)
                    for chunk in message_chunks:
                        if not send_whatsapp_message(sender_number, chunk, message_type='text'):
                            logging.error(f"Failed to send message chunk to {sender_number}")
                            break
                        # Delay between messages
                        import random
                        import time
                        if i < len(message_chunks) - 1:
                            delay = random.uniform(0.55, 1.5)
                            time.sleep(delay)
                    # Save the new exchange to history
                    # Ensure history format is compatible with genai: list of {'role': 'user'/'model', 'parts': ['text']}
                    conversation_history.append({'role': 'user', 'parts': [incoming_message_text]})
                    conversation_history.append({'role': 'model', 'parts': [gemini_reply]})
                    save_conversation_history(safe_sender_id, conversation_history)
            elif incoming_message_text:
                logging.info(f"Received '{message_type}' message from {sender_number}. No text content. Full data: {message_info}")
            elif message_type != 'unknown':
                 logging.info(f"Received '{message_type}' message from {sender_number}. No text content. Full data: {message_info}")
            else:
                logging.warning(f"Received unhandled or incomplete message from {sender_number}. Data: {message_info}")
        elif data.get('event'):
            logging.info(f"Received event '{data.get('event')}' which is not 'messages.upsert'. Data: {str(data)[:200]}")

        return jsonify({'status': 'success'}), 200
    except Exception as e:
        logging.error(f"Error processing webhook: {e}")
        return jsonify({'status': 'error', 'message': 'Internal server error'}), 500

if __name__ == '__main__':
    # For development with webhook testing via ngrok
    app.run(debug=True, port=5001, host='0.0.0.0')
