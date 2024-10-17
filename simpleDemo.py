import os

from dotenv import load_dotenv
import cv2
from moviepy.editor import VideoFileClip
import time
import base64


load_dotenv()
VIDEO_PATH = "../Data4QA/HeadButting_1.mp4"

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


    print(f"Extracted {len(base64Frames)} frames")
    return base64Frames

# Extract 1 frame per second. You can adjust the `seconds_per_frame` parameter to change the sampling rate
base64Frames = process_video(VIDEO_PATH, seconds_per_frame=1)

from langchain_openai import AzureChatOpenAI
llm = AzureChatOpenAI(
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_deployment=os.getenv("AZURE_OPENAI_GPT4O_DEPLOYMENT_NAME"),
    temperature=0,
)

# from langchain_core.prompts.chat import ChatPromptTemplate
# chat_template = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are a helpful assistant. Help me with my math homework!"),
#         ("human", "{user_input}" ),
#     ]
# )
# messages = chat_template.format_messages(
#   user_input="Hello! Could you solve 2+2?"
# )
# ai_message = llm.invoke(messages)
# print(ai_message.content)

# visual summary
# azure gpt-4o max image limit is 10
messages=[
    {"role": "system", "content": "You are classifying videos of cows into the following categories: walking, standing, sleeping or sitting, and head-butting. Respond with only the category name for the given video frames"},
    {"role": "user", "content": [
        {"type": "text", "text": "These are the frames from the video."},
        *map(lambda x: {"type": "image_url", 
                        "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "low"}}, base64Frames[:9])
        ],
    }
    ]
ai_message = llm.invoke(messages)
print(ai_message.content)