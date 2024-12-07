# main.py

# Import functions from other modules
import os
from whisper import transcribe_audio
from dalle import generate_image
from vision import describe_image
from gpt import classify_with_gpt
from dotenv import load_dotenv

# get azure secrets from .env file
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

def annotate_image(image_url: str, annotations: list) -> str:
    """
    Draws bounding boxes or annotations on the image and saves it locally.
    Returns the path to the annotated image.
    """
    response = requests.get(image_url)
    image = Image.open(response.raw)

    draw = ImageDraw.Draw(image)
    for annotation in annotations:
        box = annotation['bbox']
        label = annotation['label']
        draw.rectangle(box, outline="red", width=2)
        draw.text((box[0], box[1]), label, fill="red")

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
    audio_transcript = transcribe_audio(azure_secrets, audio_file)

    # Create a prompt from the transcription.
    prompt = f'Generate a visual representation of the customer \
               complaint: {audio_transcript}'

    # Generate an image based on the prompt.
    image_path, image_url = generate_image(
        azure_secrets, 
        prompt, 
        '1024x1024', 
        'hd', 
        'natural'
    )

    # TODO: Describe the generated image.
    description_prompt = 'identifies key visual elements related to the customer \
        complaint in the image, include the bounding box details for \
        the key elements in the image.'
    description = describe_image(
        azure_secrets, 
        image_path, 
        description_prompt
    )

    # TODO: Annotate the reported issue in the image.
    annotations = [{"bbox": (50, 50, 200, 200), "label": "Example Label"}]
    annotated_image_path = annotate_image(image_url, annotations)
    print("Annotated Image Path:", annotated_image_path)


    # TODO: Classify the complaint based on the image description.
    classification = classify_with_gpt(
        azure_secrets,
        description,
        'categories.json'
    )

    # TODO: Print or store the results as required.

    pass  # Replace this with your implementation

# Example Usage (for testing purposes, remove/comment when deploying):
if __name__ == "__main__":
    main()
