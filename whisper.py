# whisper.py
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

# Function to transcribe customer audio complaints using the Whisper model
def transcribe_audio(azure_secrets: dict, audio_file:str):
    """
    Transcribes an audio file into text using OpenAI's Whisper model.

    Returns:
    str: The transcribed text of the audio file.
    """
    # create openai client
    client = create_openai_client(
        azure_secrets['WHISPER_API_VERSION'],
        azure_secrets['AZURE_API_KEY'],
        azure_secrets['AZURE_ENDPOINT']
    )
    try:
        # Load the audio file.
        with open(audio_file, 'rb') as audio_file:
            
            # Call the Whisper model to transcribe the audio file.
            transcription = client.audio.transcriptions.create( # translations
                model=azure_secrets['WHISPER_DEPLOYMENT'],
                file=audio_file,
                probability=0.5
            )

        # save the transcribed audio
        with open('./output/transcription.txt', 'w') as text_file:
            text_file.write(transcription.txt)
        
        # Extract the transcription and return it.
        return transcription.text
    
    except Exception as e:
        print(f"An error has occured: {e}")
        return None


# Example Usage (for testing purposes, remove/comment when deploying):
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    # load environment variables
    load_dotenv()

    # extract azure secrets
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
    
    # audio file path
    audio_file = './audio/katiesteve.wav'

    # generate transcript
    transcription = transcribe_audio(azure_secrets, audio_file)
    
    # print transcript
    print(transcription)
