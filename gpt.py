# gpt.py
import json
import openai

def create_openai_client(api_version, api_key, api_endpoint):
    """
    Python function to create openai client

    Returns:
    class: openai AzureOpenAI class instance 
    """
    client = openai.AzureOpenAI(
        api_version=api_version,
        api_key=api_key,
        azure_endpoint=api_endpoint
    )
    return client

# Function to classify the customer complaint based on the image description
def classify_with_gpt(azure_secrets, complaint, categories_file):
    """
    Classifies the customer complaint into a category/subcategory based on the image description.

    Returns:
    str: The category and subcategory of the complaint.
    """
    # load categories from the json file
    with open(categories_file, 'r') as f:
        categories = json.load(f)

    # Create a prompt that includes the image description and other relevant details.
    prompt = f"Classify this complaint Description: {complaint}\nProvide categories and subcategories from below json file.\n {categories}."
    
    # crete openai client
    client = create_openai_client(
        azure_secrets['GPT_API_VERSION'],
        azure_secrets['AZURE_API_KEY'],
        azure_secrets['AZURE_ENDPOINT']
    )

    # Call the GPT model to classify the complaint based on the prompt.
    response = client.chat.completions.create(
        model=azure_secrets['GPT_DEPLOYMENT'],
        messages=[
            {"role": "system", "content": "You are a classification expert."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    # {"type": "image_url", "image_url": {"url": data_url}}
                ]
            }
        ],
        max_tokens=1024
    )
    # Extract the description and return it.
    classification = response.choices[0].message.content
   
    # save the classification result
    with open('./output/classification.txt', 'w') as text_file:
        text_file.write(classification)
        
    print(f"Classification completed....")
    print(f"classification result saved to: ./output/classification.txt")

    return classification


# Example Usage (for testing purposes, remove/comment when deploying):
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    azure_secrets = {
    'AZURE_ENDPOINT'     : os.getenv('AZURE_ENDPOINT'),
    'AZURE_API_KEY'      : os.getenv('AZURE_API_KEY'),
    'WHISPER_API_VERSION': os.getenv('WHISPER_API_VERSION'),
    'WHISPER_DEPLOYMENT' : os.getenv('WHISPER_DEPLOYMENT'),
    'DALLE_API_VERSION'  : os.getenv('DALLE_API_VERSION'),
    'DALLE_DEPLOYMENT'   : os.getenv('DALLE_DEPLOYMENT'),
    'GPT_API_VERSION'    : os.getenv('GPT_API_VERSION'),
    'GPT_DEPLOYMENT'     : os.getenv('GPT_DEPLOYMENT'),
    }

    
    cat = json.load(open('categories.json', 'r'))
    image_description = "The image shows a laptop screen displaying an overheating issue warning message. The setup includes a notebook, pen, and a cup of coffee on a desk."
    
    classification = classify_with_gpt(azure_secrets, image_description, 'categories.json')
    print(classification)
