# vision.py
import os
import ast
import json
import openai
import base64
from mimetypes import guess_type


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

def local_image_to_data_url(image_path):
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'

    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(
            image_file.read()).decode('utf-8')
    return f"data:{mime_type};base64,{base64_encoded_data}"


# Function to describe the generated image and annotate issues
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
        azure_secrets['GPT_API_VERSION'],
        azure_secrets['AZURE_API_KEY'],
        azure_secrets['AZURE_ENDPOINT']
    )

    # Call the model to describe the image and identify key elements.
    response = client.chat.completions.create(
        model=azure_secrets['GPT_DEPLOYMENT'],
        response_format={ "type": "json_object" },
        messages=[
            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
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
    image_description = response.choices[0].message.content
    print(image_description)
    
    desc_json = ast.literal_eval(image_description)
    
    with open('./output/image_description.txt', 'w') as desc_file:
        desc_file.write(str(desc_json))
    print(f"image description saved to :./output/image_description.txt")
    
    with open('./output/image_description_annotation.json', 'w') as annot_file:
        json.dump(desc_json, annot_file, sort_keys=True, indent=4)
    print(f"image annotation info saved to: ./output/image_description_annotation.json")
    
    # Extract the description and return it.
    return desc_json

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
    
    image_path = './output/generated_image.png'
    
    description_prompt = """
    Identify key visual elements related to the customer complaint in the image. 
    Return a brief description along with the bounding box details for the key elements 
    in the image in a JSON format with the following structure:

    {
        "description": "<image description>",
        "annotation": [
            {"bbox": (<x_min>, <y_min>, <x_max>, <y_max>), 'label': <object_in_region>},
            {"bbox": (<x_min>, <y_min>, <x_max>, <y_max>), 'label': <object_in_region>}
        ]
    }
    """
    description = describe_image(azure_secrets, image_path, description_prompt)
    
    # print(description)
