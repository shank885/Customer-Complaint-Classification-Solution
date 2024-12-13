# dalle.py

import openai
import requests
import os
import json
from PIL import Image

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

# Function to generate an image representing the customer complaint
def generate_image(azure_secrets: dict, prompt: str, size: str, quality: str, style: str):
    """
    Generates an image based on a prompt using OpenAI's DALL-E model.

    Returns:
    str: The path to the generated image.
    """
    # create openai client
    client = create_openai_client(
        azure_secrets['DALLE_API_VERSION'],
        azure_secrets['AZURE_API_KEY'],
        azure_secrets['AZURE_ENDPOINT']
    )

    # Create a prompt to represent the customer complaint.


    # Call the DALL-E model to generate an image based on the prompt.
    result = client.images.generate(
        model=azure_secrets['DALLE_DEPLOYMENT'],
        prompt=prompt,
        size=size,
        quality=quality,
        style=style
    )
    print(f"Image generated....")
    json_response = json.loads(result.model_dump_json())
    image_url = json_response['data'][0]['url']

    # Download the generated image and save it locally.
    image = Image.open(requests.get(image_url, stream=True).raw)
    image_path = './output/generated_image.png'
    image.save(image_path)
    print(f'image saved at: {image_path}')
    print(f'image URL: {image_url}')

    # return image path
    return image_path, image_url

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
    
    prompt = f"Generate a minimal image to represent the customer complaint: Hi, this is regarding a serious issue with a laptop I purchased from your store just two months ago. It's a MacBook Air M2 and it's started acting up within a few weeks of use. The laptop keeps overheating even during light tasks like browsing and internet. And now it's randomly shutting down without warning. I use this device for work and these interruptions are causing me significant inconvenience. I have tried basic troubleshooting like updating the software and cleaning the vents, but nothing seems to work. I even contacted your customer support previously, but the solution they suggested didn't resolve the issue. It's frustrating to have a product fail this quickly. I am requesting an urgent resolution, either a replacement or a full refund. Please get back to me as soon as possible."
    image_path, image_url = generate_image(
        azure_secrets, 
        prompt, 
        '1024x1024', 
        'hd', 
        'natural'
    )
    print(f"Generated image saved at: {image_path}")
