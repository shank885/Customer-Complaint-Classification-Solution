# main.py

# Import functions from other modules
import os
import requests
from whisper import transcribe_audio
from dalle import generate_image
from vision import describe_image
from gpt import classify_with_gpt
from dotenv import load_dotenv
from PIL import Image, ImageDraw

# get azure secrets from .env file
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

def annotate_image(image_path: str, annotations: list) -> str:
    """
    Draws bounding boxes or annotations on the image and saves it locally.
    Returns the path to the annotated image.
    """
    # response = requests.get(image_url)
    image = Image.open(image_path)

    draw = ImageDraw.Draw(image)
    for annotation in annotations:
        box = annotation['bbox']
        label = annotation['label']
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0], box[1]), label, fill="red", font_size=25)

    annotated_path = "./output/annotated_image.png"
    image.save(annotated_path)
    return annotated_path

# Main function to orchestrate the workflow
def main():
    """
    Orchestrates the workflow for handling customer complaints.
    
    Steps include:
    1. Transcribe the audio complaint.
    2. Create a prompt from the transcription.
    3. Generate an image representing the issue.
    4. Describe the generated image.
    5. Annotate the reported issue in the image.
    6. Classify the complaint into a category/subcategory pair.
    
    Returns:
    None
    """
    # Call the function to transcribe the audio complaint.
    print('\n\nTranscribing Audio...\n')
    
    audio_file = './audio/customer_complaint.wav'
    audio_transcript = transcribe_audio(azure_secrets, audio_file)
    print(audio_transcript)
    print('\n\n')
    
    # Create a prompt from the transcription.
    print('\n\nGenerating Image...\n')

    # transcript_prompt = f'Generate a realistic image to represent the customer complaint: {audio_transcript}'
    transcript_prompt = f"""I have a customer complaint which is: {audio_transcript}.\nGenerate a highly detailed, high quality image which should represent the customer complaint with the product.\nTarget specifically on the issue faced by the customer. Keep a clear non-discractive background with proper focus on product with the issue. The issue with the product should be clearly interpretable by looking at the image. Focus on highlighting the location of issue in the product. Avoid putting any additional items in the which is not relevant to the product.
    """
    
    # Generate an image based on the prompt.
    image_path, image_url = generate_image(
        azure_secrets, 
        transcript_prompt, 
        '1024x1024', 
        'hd', 
        'natural'
    )
    print('\n\n')

    # Describe the generated image.
    print('\n\nDescribing Image...\n')
    
    description_prompt = """
    Identify key visual elements related to the customer complaint in the image. 
    Return a brief description along with the bounding box detail for the key element 
    in the image in a JSON format with the following structure. The image size is 1024x1024:

    {
        "description": "<image description>",
        "annotation": [
            {"bbox": (<x_min>, <y_min>, <x_max>, <y_max>), 'label': <object_in_region>}
        ]
    }
    """

    image_description = describe_image(
        azure_secrets, 
        image_path, 
        description_prompt
    )
    print('\n\n')

    # Annotate the reported issue in the image.
    annotations = image_description["annotation"]
    annotated_image_path = annotate_image(image_path, annotations)
    print("Annotated Image Path:", annotated_image_path)
    print('\n\n')

    # Classify the complaint based on the image description.
    print('\n\nClassifying description...\n')
    classification = classify_with_gpt(
        azure_secrets,
        image_description['description'],
        'categories.json'
    )
    print(classification)
    print('\n\n')


# Example Usage (for testing purposes, remove/comment when deploying):
if __name__ == "__main__":
    main()