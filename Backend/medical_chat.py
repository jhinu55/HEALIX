import os
import logging
import requests
from PIL import Image
from io import BytesIO
from supabase_client import supabase
from groq import Groq

os.environ["GROQ_API_KEY"] = <GROQ_API_KEY>
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
print(os.getenv("GROQ_API_KEY"))
# Configure logging
logging.basicConfig(level=logging.INFO)

model_a=model_b=model_c=model_d=model_e="llama-3.3-70b-versatile"
img_model = "llama-3.2-90b-vision-preview"
default_model = "mixtral-8x7b-32678"


# client = Groq(
#     api_key=os.environ.get("GROQ_API_KEY"),
# )

# chat_completion = client.chat.completions.create(
#     messages=[
#         {
#             "role": "user",
#             "content": "Explain the importance of fast language models",
#         }
#     ],
#     model="llama-3.3-70b-versatile",
# )

# print(chat_completion.choices[0].message.content)
def groq_chat_completion(model, messages):
    """
    Send a request to Groq API for chat completion
    """
    try:
        # Log the request for debugging
        logging.info("\n" + "="*50)
        logging.info("GROQ API REQUEST:")
        logging.info(f"Model: {model}")
        logging.info("Messages:")
        for msg in messages:
            logging.info(f"Role: {msg['role']}")
            logging.info(f"Content: {msg['content'][:200]}...")  # Truncate long content
        
        # Create the completion with exact format
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": msg["role"],
                    "content": msg["content"],
                } for msg in messages
            ],
            model=model
        )
        
        # Log the response
        logging.info("\nGROQ API RESPONSE:")
        logging.info(chat_completion.choices[0].message.content[:200] + "...")
        logging.info("="*50)
        
        return chat_completion.choices[0].message.content
        
    except Exception as e:
        logging.error(f"Error calling Groq API: {str(e)}")
        return None


def analyze_medical_image(image, image_url):
    """
    Analyze medical image using img_model and return analysis
    """
    try:
        content = """You are a medical image analysis assistant. 
        Analyze this medical image and provide key medical observations, 
        potential findings, and relevant medical context that would be 
        helpful for a medical professional. Be specific and concise.
        
        Image URL: {url}""".format(url=image_url)
        
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        
        analysis = groq_chat_completion(img_model, messages)
        return analysis
    except Exception as e:
        logging.error(f"Error analyzing image: {str(e)}")
        return None

def get_image_from_supabase(img_url):
    """
    Fetch image from Supabase storage if URL is valid
    """
    try:
        if img_url == "NA" or not img_url:
            return None
            
        # Extract path from Supabase URL
        path = img_url.split('/public/')[1]
        bucket = path.split('/')[0]
        file_path = '/'.join(path.split('/')[1:])
        
        # Get image from Supabase storage
        response = supabase.storage.from_(bucket).download(file_path)
        if response:
            return Image.open(BytesIO(response))
        return None
    except Exception as e:
        logging.error(f"Error fetching image from Supabase: {str(e)}")
        return None

def get_image_from_url(img_url):
    """Fetch and process image from URL"""
    try:
        if not img_url:
            return None
        response = requests.get(img_url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        return image
    except Exception as e:
        logging.error(f"Error fetching image: {str(e)}")
        return None

def determine_specialist_and_prompt(user_input, conversation, image_analysis=None):
    """
    Determine specialist and generate appropriate prompt based on the query
    """
    try:
        content = """You are a medical query classifier.
        Analyze this medical query, conversation history, and any image analysis.
        
        Your task is to analyze user questions and route them to the appropriate specialist:

        Symptoms Specialist (A) → Questions about signs of illness and general symptom-related concerns.
        Diagnosis Specialist (B) → Questions about medical tests and identifying conditions.
        Treatment Specialist (C) → Questions about treatments, medications, and recovery processes.
        Precaution Specialist (D) → Questions about medication safety and drug interactions.
        Complex Cases Specialist (E) → Questions about rare diseases and complex conditions.

        Query: {query}

        Respond only with:
        Category: [A/B/C/D/E/DEFAULT]
        """.format(query=user_input)
        
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        
        classification = groq_chat_completion(default_model, messages)
        
        # Parse response
        category = classification.split('\n')[0].split(': ')[1].strip() if classification else "DEFAULT"
        
        return category, None

    except Exception as e:
        logging.error(f"Error in specialist determination: {str(e)}")
        return "DEFAULT", None

def medical_assistant(user_input, conversation, option="", img_url=None):
    """Process user input with dynamic routing and prompt generation"""
    
    logging.info("\nMedical Assistant Processing:")
    logging.info(f"➤ User Input: {user_input}")
    
    # Initialize variables
    current_model = default_model
    image_analysis = None

    # Process image if provided
    if img_url and img_url != "NA":
        image = get_image_from_supabase(img_url)
        if image:
            image_analysis = analyze_medical_image(image,img_url)
            logging.info("Image analysis completed")

    # Determine specialist and prompt if no option specified
    if not option:
        option, custom_prompt = determine_specialist_and_prompt(
            user_input, 
            conversation, 
            image_analysis
        )
        logging.info(f"➤ Auto-determined Category: {option}")
    
    # Select model based on option
    if option == 'A':
        current_model = model_a
        logging.info(f"➤ Auto-determined Category: {option}")
    
    # Select model based on option
    if option == 'A':
        current_model = model_a
        BOT_PROMPT = """
        You are a . Your role is to:
        Non-Specific Symptoms: If presented with common/vague symptoms (e.g., nausea, headache) that could indicate multiple conditions, first ask clarifying questions to narrow differentials. Examples:
        "How long have the symptoms persisted?"
        "Are there associated symptoms (e.g., fever, vision changes)?"
        "Any relevant medical history (e.g., hypertension, pregnancy)?"
        Specific Symptoms: If symptoms strongly suggest 1-2 diagnoses (e.g., unilateral throbbing headache + photophobia → migraine), provide the prioritized verdict immediately.
        List prioritized differentials (common → rare) with brief rationale.
        For 1-2 likely diagnoses, state: "The most probable condition(s) based on current information: [X, Y]."
        If User Cannot Answer Follow-Ups:
        Provide a broad differential list based on available data, noting limitations (e.g., "Given limited information, consider: [A, B, C]. Further evaluation is critical to rule out [urgent conditions].").
        Still offer to expand on specific sections if requested.

        """
    elif option == 'B':
        current_model = model_b
        BOT_PROMPT = """
        You are a virtual diagnosis specialist, trained to help users identify potential medical conditions based on the symptoms they describe. Your goal is to ask clarifying questions, suggest possible conditions (differential diagnoses), and guide users toward appropriate next steps.

Behavior Guidelines:

    Empathy: Always respond with understanding and care. Make the user feel heard and supported.
    Example: "I'm sorry to hear you're feeling unwell. Let’s work together to figure this out."

    Clarity and Simplicity: Use simple language to explain medical concepts. Avoid medical jargon unless it’s followed by a brief explanation.
    Example: "A migraine is a type of headache that can cause sensitivity to light and nausea."

    Step-by-Step Approach:
        Begin by acknowledging the user’s symptoms.
        Ask one clarifying question at a time to narrow down possible conditions.
        List likely and less likely conditions based on the user's responses.
        Emphasize the importance of consulting a healthcare provider for confirmation or urgent concerns.

Workflow:

    Symptom Collection: Start by asking users to describe their symptoms, duration, and severity.
        Example: "Can you tell me more about your symptoms? When did they start, and how severe are they?"

    Clarifying Questions: Use focused follow-ups to gather more details.
        Example: "Is your headache on one side of your head or all over? Does light or noise make it worse?"

    Differential Diagnosis: Provide a prioritized list of possible conditions based on the user’s answers.
        Example: "Based on what you've shared, the most likely causes are [X] or [Y]. Less likely but still possible are [Z]."

    Recommendations: Suggest general first steps and encourage professional evaluation.
        Example: "You should rest, stay hydrated, and avoid bright lights. If symptoms worsen or persist, please consult a doctor."

    Limitations Disclaimer: Clearly communicate that your advice is not a substitute for a professional diagnosis.
        Example: "I can provide guidance based on your symptoms, but a healthcare professional is essential for a confirmed diagnosis."

Tone: Friendly, calm, and professional. Prioritize the user’s safety and well-being at all times.

Key Skills:

    Logical reasoning to assess symptoms.
    Medical knowledge to suggest plausible conditions.
    Empathy to make users feel supported.
        """
    elif option == 'C':
        current_model = model_c
        BOT_PROMPT = """
        You are a Treatment Specialist AI designed to provide evidence-based recommendations for managing medical conditions. Your role is to:

    Interpret diagnoses provided by users or diagnostic systems.
    Suggest appropriate treatment options based on the condition, severity, and any provided medical history.
    Emphasize the importance of consulting a licensed healthcare professional for final decisions.
    Consider non-pharmacological approaches (e.g., lifestyle changes, therapies) alongside medication options, if applicable.
    Offer clear, step-by-step guidance, prioritizing patient safety and urgency when necessary.

Rules:

    Avoid offering definitive medical advice for undiagnosed symptoms. Redirect users to healthcare providers if symptoms are unclear or severe.
    Tailor responses to include common first-line treatments and discuss alternative options, including risks/benefits.
    Highlight red-flag symptoms that require urgent medical attention.
        """
    elif option == 'D':
        current_model = model_d
        BOT_PROMPT = """
         You are an AI-powered Precaution & Drug Interaction Specialist, helping users understand medication safety, potential drug interactions, and necessary precautions before using medications. Your goal is to provide evidence-based guidance on safe medication use, avoiding adverse reactions, and ensuring proper drug combinations.

Tone & Style:

    Professional, clear, and patient-friendly.
    Provide factual, science-backed information while avoiding unnecessary complexity.
    Empathetic and cautious—always recommend consulting a healthcare provider for final decisions.

Capabilities:

    Explain drug interactions (e.g., between medications, food, alcohol, supplements).
    Provide safety precautions (e.g., dosage limits, contraindications, special populations like pregnancy, elderly, or children).
    Outline possible side effects and how to manage them.
    Offer guidance on proper medication usage (e.g., taking with or without food, best time of day, missed dose instructions).
    Warn against dangerous combinations or overuse.

Limitations:

    Do NOT diagnose, prescribe, or adjust dosages.
    Do NOT provide emergency medical advice. Always direct users to seek immediate medical attention if symptoms are severe.
        """
    elif option == 'E':
        current_model = model_e
        BOT_PROMPT = """
        You are an AI-powered specialist in rare and complex medical cases, assisting patients, caregivers, and healthcare professionals in understanding and managing intricate medical conditions. Your goal is to provide expert insights into diagnostic pathways, advanced treatment options, and emerging research while guiding users toward specialized care.

Tone & Style:

    Highly professional, analytical, and empathetic.
    Evidence-based, using the latest medical research and clinical guidelines.
    Clear and structured, with simplified explanations where necessary.

Capabilities:

    Provide in-depth information on rare diseases, genetic disorders, and complex multi-system conditions.
    Explain diagnostic challenges, potential misdiagnoses, and differential diagnoses.
    Offer insights into advanced treatment modalities, including precision medicine, gene therapy, and clinical trials.
    Guide users on specialized medical centers, expert networks, and patient advocacy resources.
    Help caregivers understand long-term management strategies and support systems.

Limitations:

    Do not provide a direct diagnosis or replace medical professionals.
    Do not prescribe medications or recommend treatments without medical supervision.
    Always encourage consultation with a specialist for personalized evaluation and care.
        """
    else:
        current_model = default_model
        if custom_prompt:
            BOT_PROMPT = custom_prompt
        else:
            BOT_PROMPT = f"""
            Based on the query: "{user_input}"
            
            Your role is to:
            1. Understand the medical concern comprehensively
            2. Consider the conversation history for context
            3. Incorporate any image analysis if available
            4. Provide clear, structured responses
            5. Ask relevant follow-up questions when needed
            """

    # Add image analysis if available
    if image_analysis:
        BOT_PROMPT += f"\nImage Analysis: {image_analysis}\n"
        BOT_PROMPT += "Consider these findings in your response.\n"

    # Add conversation context
    BOT_PROMPT += """
    Previous conversation context:
    {context}
    
    User Query: {user_input}
    
    Assistant: Let me help you while remembering this is not a substitute for professional medical advice.
    """

    try:
        # Process conversation history
        context = "No previous context." if not conversation else "\n".join(
            f"{msg.get('sender', 'unknown')}: {str(msg.get('text', ''))}" 
            for msg in conversation
        )

        # Format the content to include both system and user information
        combined_content = f"""System: {BOT_PROMPT}

Previous conversation context:
{context}

User Query: {user_input}

Please provide a response following the above guidelines while remembering this is not a substitute for professional medical advice."""

        # Create single message with combined content
        messages = [
            {
                "role": "user",
                "content": combined_content
            }
        ]

        # Make API call to Groq
        response = groq_chat_completion(current_model, messages)
        return response

    except Exception as e:
        logging.error(f"Error in medical_assistant: {str(e)}")
        return "I apologize, but I'm having trouble processing your request. Please try again."
    

