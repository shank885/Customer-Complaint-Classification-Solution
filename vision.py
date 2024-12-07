# vision.py
import os
import openai


def create_openai_client(api_version, api_key, api_endpoint):
    client = AzureOpenAI(
        api_version=api_version,
        api_key=api_key,
        azure_endpoint=api_endpoint
    )
    return client

# Function to describe the generated image and annotate issues
def local_image_to_data_url(image_path):
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'

    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(
            image_file.read()).decode('utf-8')
    return f"data:{mime_type};base64,{base64_encoded_data}"

def describe_image(azure_secrets, image_path, prompt):
    """
    Describes an image and identifies key visual elements related to the customer complaint.

    Returns:
    str: A description of the image, including the annotated details.
    """
    # Load the generated image.
    data_url = local_image_to_data_url(image_path)

    # crete openai client
    client = create_openai_client(
        azure_secrets['GPT_API_VERSION',
        azure_secrets['AZURE_API_KEY'],
        azure_secrets['AZURE_ENDPOINT']
    )

    # Call the model to describe the image and identify key elements.
    response = client.chat.completions.create(
        model=azure_secrets['GPT_DEPLOYMENT'],
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]
            }
        ],
        max_tokens=1024
    )
    # Extract the description and return it.
    return response.choices[0].message.content

# Example Usage (for testing purposes, remove/comment when deploying):
# if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    azure_secrets = {
    'AZURE_ENDPOINT': os.getenv('AZURE_ENDPOINT'),
    'AZURE_API_KEY': os.getenv('AZURE_API_KEY'),
    'WHISPER_API_VERSION': os.getenv('WHISPER_API_VERSION'),
    'WHISPER_DEPLOYMENT': os.getenv('WHISPER_DEPLOYMENT'),
    'DALLE_API_VERSION': os.getenv('DALLE_API_VERSION'),
    'DALLE_DEPLOYMENT': os.getenv('DALLE_DEPLOYMENT'),
    'GPT_API_VERSION': os.getenv('GPT_API_VERSION'),
    'GPT_DEPLOYMENT': os.getenv('GPT_DEPLOYMENT'),
    }
    
    image_path = './output/generated_image.png'
    prompt = 'identifies key visual elements related to the customer \
        complaint in the image. Mark the bounding box of the region in the image with any notable defects.'
    
    description = describe_image(azure_secrets, image_path)
    print(description)
