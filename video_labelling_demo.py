import argparse
import os
from dotenv import load_dotenv
import cv2
import base64
from langchain_openai import AzureChatOpenAI
import pandas as pd
import numpy as np

def process_video(video_path, seconds_per_frame=2):
    base64Frames = []

    video = cv2.VideoCapture(video_path)
    
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frames_to_skip = int(fps * seconds_per_frame)
    curr_frame=0

    # Loop through the video and extract frames at specified sampling rate
    while curr_frame < total_frames - 1:
        video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        curr_frame += frames_to_skip
    video.release()


    print(f"\tExtracted {len(base64Frames)} frames")
    return base64Frames

if __name__ == '__main__':
    # read the API keys
    load_dotenv()
    
    parser = argparse.ArgumentParser(description='Video Lablling using OpenAI')
    parser.add_argument('--seconds_per_frame', type=float, default=1, help='parameter to change the sampling rate')
    parser.add_argument('--input_csv', type=str, default='input.csv', help='CSV file with filenames of videos')
    args = parser.parse_args()
    
    # Define the LLM
    llm = AzureChatOpenAI(
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_deployment=os.getenv("AZURE_OPENAI_GPT4O_DEPLOYMENT_NAME"),
        temperature=0,
    )
    
    # Prompt
    
    ##
    data = pd.read_csv(args.input_csv)
    for i, row in data.iterrows():
        # path to video is on the second column
        print('Processing Video -', row.iloc[0], ': ')
        base64Frames = process_video(row.iloc[1], seconds_per_frame=1)
        
        prompt=[
            {"role": "system", "content": "You are classifying videos of cows into the following categories: walking, standing, sleeping/sitting, and head-butting. Respond with only the category name for the given video frames"},
            {"role": "user", "content": [
                {"type": "text", "text": "These are the frames from the video."},
                *map(lambda x: {"type": "image_url", 
                                "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "low"}}, base64Frames[:9])
                ],
            }
            ]
        ai_message = llm.invoke(prompt)
        print('\tPredicted class:', ai_message.content)
        print('\tActual class:', row.iloc[2])
        print()