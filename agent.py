import logging
import os
import json
import aiohttp
import base64
import requests
from os.path import join, dirname
from datetime import datetime
from typing import Annotated

from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import deepgram, openai, silero, turn_detector
from livekit import rtc
from livekit.agents.llm import ChatMessage, ChatImage
from openai import AsyncOpenAI
from livekit.agents.utils.images.image import encode, EncodeOptions
dotenv_path = join(dirname(__file__), '.env.prod')
load_dotenv(dotenv_path)

import cv2  # OpenCV for better YUV conversion
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
print(LIVEKIT_API_KEY, '<<LIVEKIT_API_KEY')
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
# LIVEKIT_SERVER_URL = "wss://personal-assistant-cv4hqx25.livekit.cloud" 
LIVEKIT_SERVER_URL = os.getenv("LIVEKIT_URL")
API_URL=os.getenv("API_URL")

# api_client = livekit.api.RoomServiceClient(LIVEKIT_SERVER_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET)

logger = logging.getLogger("voice-assistant")
logger.setLevel(logging.INFO)

def encode_image(image):
  with open(image, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  
async def get_latest_image_as_base64(room: rtc.Room) -> str:
    """Capture a single frame from the video track and return it as a base64-encoded string."""
    video_frame = await get_latest_image(room)
    if video_frame is None:
        logger.error("No frame captured.")
        return None

    try:
        # ✅ Encode the video frame using LiveKit's built-in function
        image_bytes = encode(video_frame, EncodeOptions(format="JPEG", quality=85))

        # ✅ Convert image bytes to Base64
        base64_str = base64.b64encode(image_bytes).decode("utf-8")

        logger.info("Successfully converted image to Base64.")
        return base64_str

    except Exception as e:
        logger.error(f"Error encoding image to Base64: {e}")
        return None

async def save_latest_image(room: rtc.Room, filename="captured_frame.jpg"):
    """Capture a single frame from the video track and save it as a JPEG file."""
    video_frame = await get_latest_image(room)
    if video_frame is None:
        logger.error("No frame captured.")
        return "Error: No frame captured."

    try:
        # ✅ Use LiveKit's encode function to process the video frame
        image_bytes = encode(video_frame, EncodeOptions(format="JPEG", quality=85))

        # ✅ Save the processed image as a file
        with open(filename, "wb") as f:
            f.write(image_bytes)

        logger.info(f"Image saved as {filename}")
        return f"Image saved: {filename}"

    except Exception as e:
        logger.error(f"Error encoding or saving frame: {e}")
        return "Error: Failed to process the video frame."

async def get_video_track(room: rtc.Room):
        """Find and return the first available remote video track in the room."""
        for participant_id, participant in room.remote_participants.items():
            for track_id, track_publication in participant.track_publications.items():
                if track_publication.track and isinstance(
                    track_publication.track, rtc.RemoteVideoTrack
                ):
                    logger.info(
                        f"Found video track {track_publication.track.sid} "
                        f"from participant {participant_id}"
                    )
                    return track_publication.track
        raise ValueError("No remote video track found in the room")
    
async def get_latest_image(room: rtc.Room):
    """Capture and return a single frame from the video track."""
    video_stream = None
    try:
        video_track = await get_video_track(room)
        video_stream = rtc.VideoStream(video_track)
        async for event in video_stream:
            logger.debug("Captured latest video frame")
            return event.frame
    except Exception as e:
        logger.error(f"Failed to get latest image: {e}")
        return None
    finally:
        if video_stream:
            await video_stream.aclose()

class AssistantFnc(llm.FunctionContext):
    """
    This class defines all assistant functions that the LLM can call.
    """
    email: str = None
    room = {}
    

    @llm.ai_callable()
    async def get_todays_events(
        self,
    
    ):
        """Fetch today's events from Google Calendar for the authenticated user, please check metadata for email"""
        logger.info(f"Fetching today's events for {self.email}")

        if not self.email:
            logger.warning("⚠️ No email provided!")
            return "I couldn't retrieve your events because I don't have your email."

        URL = f"{API_URL}/calendar/{self.email}/today"
        logger.info(f"📡 Requesting calendar data from {API_URL}/calendar/{self.email}/today")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(URL) as response:
                    logger.info(f"📩 Received response: {response.status}")
                    if response.status == 200:
                        events_data = await response.json()
                        if events_data:
                            return "\n".join(
                                f"- {event.get('summary', 'No Title')} at {event.get('start', 'Unknown time')}"
                                for event in events_data
                            )
                        return "There are no events scheduled for today."
                    else:
                        logger.error(f"❌ API Error: {response.status}")
                        return "I couldn't retrieve today's events due to a server issue."
        except aiohttp.ClientError as e:
            logger.error(f"❌ Network error: {e}")
            return "I couldn't retrieve today's events due to a network issue."
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
            return "An unexpected error occurred while retrieving today's events."

    @llm.ai_callable()
    def get_time(self):
        """Called to retrieve the current local time."""
        return datetime.now().strftime("%H:%M:%S")

    @llm.ai_callable()
    def get_date(self):
        """Called to get today's date."""
        return datetime.today().strftime('%Y-%m-%d')
    
    @llm.ai_callable()
    async def get_text_from_image(self):
        """If you can't read text from an image use this this function, it will return all text found"""
        # await save_latest_image(self.room)
        base64_image = await get_latest_image_as_base64(self.room)
        try:
            ocr_response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Please extract the text from this document image and return the result as a single string, no other output.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
            )

            logger.info("Received OCR response from OpenAI")
            logger.info("OCR response was: ", ocr_response.choices[0].message.content)
            # Extract text from the API response
            return ocr_response.choices[0].message.content 

        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            return "Error: Failed to process OCR."
        
    @llm.ai_callable()    
    def get_personal_contacts(self):
        """
        If user wants to call someone don't ask them to confirm the user is in their contacts just use this.
        If a user wants to call someone use this to get the email of the contact then use send_web_rtc_contact
        you will find their email here "telecom": [
        {
            "system": "phone",
            "value": "07939043476",
            "use": "mobile",
            "_id": "67bf4a0ff718291342b6bc2c"
        },
        {
            "system": "email",
            "value": "kevinsmith262626@gmail.com",
            "use": "home",
            "_id": "67bf4a0ff718291342b6bc2d"
        }
        ]
        """
        logger.info(f"Getting contacts for: {self.email}<<<<<<")
        url = "{API_URL}/relatedPerson/email/ai"
        cookies = {"email": self.email}  # Setting the email cookie

        try:
            response = requests.get(url, cookies=cookies)
            response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)
            print(response.json(), '<<<<<')
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

    @llm.ai_callable()    
    def send_web_rtc_contact(self, email: Annotated[
            str, llm.TypeInfo(description="the email used for webrtc conatct information")
        ]):
        """
            Use this to send the webrtc contact information to the user so they can initiate the webrtc call
        """
        logger.info(f"Sending info for : {email}<<<<<<<<")
        url = "{API_URL}/send-webrtc-message"
        headers = {"Content-Type": "application/json"}
        data = {
            "toEmail": email,
            "fromEmail": self.email,
            "message": "Start a webrtc call"
        }

        try:
            response = requests.post(url, json=data, headers=headers)
            response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)
            return json.dumps(response.json())
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}


def prewarm_process(proc: JobProcess):
    """Pre-warm any necessary models."""
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    """Entrypoint for the LiveKit assistant."""
    await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)
    fnc_ctx = AssistantFnc()  # Create our function context instance
    room = {}

    # ✅ Set up initial assistant behavior
    initial_chat_ctx = llm.ChatContext().append(
        text=(
            "You are a personal assistant created by LiveKit. Your interface with users is voice. "
            "You can fetch calendar events, tell the time, and give the current date. "
            "You can help a person call someone by using get_personal_contacts and send_web_rtc_contact."
            "When a user asks to call someone, "
            "you must first call `get_personal_contacts` to find their email. "
            "Then, immediately call `send_web_rtc_contact` using that email. "
            "DO NOT wait for the user to say anything after confirming. "
            "You MUST execute both functions automatically and immediately."
        ),
        role="system",
    )

    # ✅ Wait for participant
    participant = await ctx.wait_for_participant()
    logger.info(f"Participant connected: {participant.identity}")

    # ✅ Extract metadata (email)
    try:
        metadata = json.loads(participant.metadata) if participant.metadata else {}
        email = metadata.get("email")
        fnc_ctx.email = email

        if email:
            logger.info(f"✅ Storing User Email: {email}")
        else:
            logger.warning("⚠️ No email found in participant metadata!")

    except json.JSONDecodeError:
        logger.error("❌ Failed to decode participant metadata!")
        email = None

    async def before_llm_cb(assistant: VoicePipelineAgent, chat_ctx: llm.ChatContext):
        """
        Callback that runs right before the LLM generates a response.
        Captures the current video frame and adds it to the conversation context.
        """
        latest_image = await get_latest_image(room)
        if latest_image:
            image_content = [ChatImage(image=latest_image)]
            chat_ctx.messages.append(ChatMessage(role="user", content=image_content))
            logger.debug("Added latest frame to conversation context")
            
    
    # ✅ Initialize Voice Assistant
    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(),
        llm=openai.LLM(model="gpt-4o"),
        tts=openai.TTS(voice="onyx"), #["alloy", "echo", "fable", "onyx", "nova", "shimmer"] 
        fnc_ctx=fnc_ctx,
        chat_ctx=initial_chat_ctx,
        max_nested_fnc_calls=5,
        before_llm_cb=before_llm_cb,
        turn_detector=turn_detector.EOUModel(),
        min_endpointing_delay=0.2 
    )

    fnc_ctx.room = ctx.room
    room = ctx.room

    # ✅ Start the assistant
    agent.start(ctx.room, participant)

    await agent.say("Hello! How can I assist you today?")


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm_process,
        ),
    )