import asyncio
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
from livekit.agents.llm.chat_context import ChatContext


from utils.imageHelpers import get_latest_image, get_latest_image_as_base64

dotenv_path = join(dirname(__file__), '.env.prod')
load_dotenv(dotenv_path)

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
print(LIVEKIT_API_KEY, '<<LIVEKIT_API_KEY')
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
# LIVEKIT_SERVER_URL = "wss://personal-assistant-cv4hqx25.livekit.cloud" 
LIVEKIT_SERVER_URL = os.getenv("LIVEKIT_URL")
API_URL=os.getenv("API_URL")


logger = logging.getLogger("voice-assistant")
logger.setLevel(logging.INFO)

def get_first_name(full_name: str) -> str:
    """Returns the first name from a full name string."""
    if not full_name.strip():
        return ""
    return full_name.strip().split()[0]

class AssistantFnc(llm.FunctionContext):
    """
    This class defines all assistant functions that the LLM can call.
    """
    def __init__(self, agent: VoicePipelineAgent, chat_ctx: ChatContext):
        self.agent = agent  # Use the agent to invoke RPC calls
        self.chat_ctx = chat_ctx  # Store image references
        super().__init__()

    email: str = None
    room = {}
    contacts = []

    @llm.ai_callable()
    async def request_image(
        self,
        reason: Annotated[
            str, llm.TypeInfo(description="Reason for requesting an image from the user's video stream.")
        ],
        image: Annotated[
            str, llm.TypeInfo(description="The base64-encoded image or image URL provided by the frontend.")
        ],
    ):
        """Use this if you need to see through the camera you don't need to confirm with the user"""
        """Processes an image provided by the frontend when the LLM determines it is needed for context."""
        logger.info(f"Processing an image due to: {reason}")
        print("getting image from frontend<<<<<<<<<")
        image = await get_latest_image_as_base64(self.room)
        
        if not image:
            return "I tried to capture an image, but it was unavailable."

        # ✅ Store the image in the chat context so the LLM can reference it
        # self.chat_ctx.add_message(
        #     role="assistant",
        #     content="I've received an image to assist with your request.",
        #     attachments=[{"type": "image", "url": image}]
        # )
        self.chat_ctx.messages.append(ChatMessage(role="user", content=image))

        return f"I have received the image. Here is the reference: {image}."
    
    @llm.ai_callable()
    async def get_weeks_events(self):
        """Fetch this week's events from Google Calendar for the authenticated user, please check metadata for email"""
        logger.info(f"Fetching this week's events for {self.email}")

        if not self.email:
            logger.warning("⚠️ No email provided!")
            return "I couldn't retrieve your events because I don't have your email."

        URL = f"{API_URL}/calendar/{self.email}/week"
        logger.info(f"📡 Requesting calendar data from {URL}")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(URL) as response:
                    logger.info(f"📩 Received response: {response.status}")
                    if response.status == 200:
                        events_data = await response.json()
                        if events_data:
                            return "\n".join(
                                f"- {event.get('summary', 'No Title')} on {event.get('start', 'Unknown time')}"
                                for event in events_data
                            )
                        return "There are no events scheduled for this week."
                    else:
                        logger.error(f"❌ API Error: {response.status}")
                        return "I couldn't retrieve this week's events due to a server issue."
        except aiohttp.ClientError as e:
            logger.error(f"❌ Network error: {e}")
            return "I couldn't retrieve this week's events due to a network issue."
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
            return "An unexpected error occurred while retrieving this week's events."
        

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
    async def get_events_for_date(
        self,
        date: Annotated[
            str,
            llm.TypeInfo(description="The date to fetch events for in YYYY-MM-DD format")
        ],
    ):
        """Fetch events for a specific date from Google Calendar for the authenticated user."""
        logger.info(f"📅 Fetching events for {date} for {self.email}")

        if not self.email:
            logger.warning("⚠️ No email provided!")
            return "I couldn't retrieve your events because I don't have your email."

        URL = f"{API_URL}/calendar/{self.email}/date?date=${date}"
        logger.info(f"📡 Requesting calendar data from {URL}")

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
                        return f"There are no events scheduled for {date}."
                    else:
                        logger.error(f"❌ API Error: {response.status}")
                        return f"I couldn't retrieve events for {date} due to a server issue."
        except aiohttp.ClientError as e:
            logger.error(f"❌ Network error: {e}")
            return f"I couldn't retrieve events for {date} due to a network issue."
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
            return f"An unexpected error occurred while retrieving events for {date}."

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
        """THIS IS ONLY FOR READING TEXT  - If you can't read text from an image use this this function, it will return all text found"""
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
    def get_personal_contacts(self, name: Annotated[
            str, llm.TypeInfo(description="Name of the contact the user is looking for")
        ]):
        """
        If user wants to call someone don't ask them to confirm the user is in their contacts just use this.
        If a user wants to call someone use this to get the email of the contact then use send_web_rtc_contact
        """
        logger.info(f"Looking for name: {name}<<<<<<")
        logger.info(f"Getting contacts for: {self.email}<<<<<<")
        url = f"{API_URL}/relatedPerson/email/ai"
        cookies = {"email": self.email}  # Setting the email cookie

        try:
            print(f"{API_URL}/relatedPerson/email/ai", '<<<<<<',cookies )
            response = requests.get(url, cookies=cookies)
            response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)
            print(response.json(), '<<<<<')
            self.contacts = response.json()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(e, '<<<<<ERROR<<')
            return {"error": str(e)}

    @llm.ai_callable()    
    def send_web_rtc_contact(self, email: Annotated[
            str, llm.TypeInfo(description="email found in the results from get_personal_contacts")
        ]):
        """
            Use this to send the webrtc contact information to the user so they can initiate the webrtc call
        """
        logger.info(f"Calling from : {self.email}<<<<<<<<")
        logger.info(f"Sending info for : {email}<<<<<<<<")
        url = f"{API_URL}/send-webrtc-message"
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
            print(e, '<<<<<<<ERROR<<<')
            return {"error": str(e)}

    @llm.ai_callable()    
    def playSomething(self, music_request: Annotated[
            str, llm.TypeInfo(description="The words the users used to describe what they want to play")
        ]):
        """
            Use this if the user request to play some music or play something
        """
        logger.info(f"Calling from : {self.email}<<<<<<<<")

        url = f"{API_URL}/spotify/requestMusic"
        headers = {"Content-Type": "application/json"}
        data = {
            "toEmail": self.email,
            "request": music_request,
        }
        print(music_request, '<<<<MUSIC REQUEST')
        try:
            response = requests.post(url, json=data, headers=headers)
            response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)
            return json.dumps(response.json())
        except requests.exceptions.RequestException as e:
            print(e, '<<<<<<<playSomething ERROR<<<')
            return {"error": str(e)}
        
    @llm.ai_callable()    
    def getHelpEmergencyCall(self):
        """
            Use this to get help for the user
        """
        logger.info(f"Calling from : {self.email}<<<<<<<<")
        url = f"{API_URL}/send-webrtc-message/emergencyCall"
        headers = {"Content-Type": "application/json"}
        data = {
            "toEmail": self.email,
        }
        try:
            response = requests.post(url, json=data, headers=headers)
            response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)
            return json.dumps(response.json())
        except requests.exceptions.RequestException as e:
            print(e, '<<<<<<<ERROR<<<')
            return {"error": str(e)}
        
    @llm.ai_callable()
    def userFinishedSoDisconnect(self):
        """Use this function when a user is finished and it will disconnect them"""
        url = f"{API_URL}/modes/"
        headers = {"Content-Type": "application/json"}
        data = {
            "toEmail": self.email,
            "mode": 'idle'
        }
        try:
            response = requests.post(url, json=data, headers=headers)
            response.raise_for_status()
            return json.dumps(response.json())
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}


def prewarm_process(proc: JobProcess):
    """Pre-warm any necessary models."""
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    """Entrypoint for the LiveKit assistant."""
    await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)
    fnc_ctx = AssistantFnc(agent=None, chat_ctx=None)  # Create our function context instance
    room = {}

    # ✅ Set up initial assistant behavior
    initial_chat_ctx = llm.ChatContext().append(
        text=(
            "You are a personal assistant call Winston, you are here to help in anyway you can."
            "You can fetch calendar events, tell the time, and give the current date. "
            "You can help a person call someone by using get_personal_contacts and send_web_rtc_contact."
            "When a user asks to call someone, "
            "you must first call `get_personal_contacts()` to find a list of emails. "
            "Then, immediately call `send_web_rtc_contact()` using the correct email from the list. "
            "DO NOT wait for the user to say anything after confirming. "
            "You MUST execute both functions automatically and immediately."
            "If you user asks for help use getHelpEmergencyCall()"
            "You can help with a "
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
        # fnc_ctx.chat_ctx = initial_chat_ctx


        if email:
            logger.info(f"✅ Storing User Email: {email}")
        else:
            logger.warning("⚠️ No email found in participant metadata!")

    except json.JSONDecodeError:
        logger.error("❌ Failed to decode participant metadata!")
        email = None

    # async def before_llm_cb(assistant: VoicePipelineAgent, chat_ctx: llm.ChatContext):
    #     """
    #     Callback that runs right before the LLM generates a response.
    #     Captures the current video frame and adds it to the conversation context.
    #     """
    #     latest_image = await get_latest_image(room)
    #     if latest_image:
    #         image_content = [ChatImage(image=latest_image)]
    #         chat_ctx.messages.append(ChatMessage(role="user", content=image_content))
    #         logger.debug("Added latest frame to conversation context")
            
    
    # ✅ Initialize Voice Assistant
    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(),
        llm=openai.LLM(model="gpt-4o"),
        tts=openai.TTS(voice="fable"), #["alloy", "echo", "fable", "onyx", "nova", "shimmer"] 
        fnc_ctx=fnc_ctx,
        chat_ctx=initial_chat_ctx,
        max_nested_fnc_calls=5,
        # before_llm_cb=before_llm_cb,
        turn_detector=turn_detector.EOUModel(),
        min_endpointing_delay=0.2 
    )
    fnc_ctx.agent = agent 
    fnc_ctx.room = ctx.room
    fnc_ctx.chat_ctx = agent.chat_ctx

    # ✅ Start the assistant
    agent.start(ctx.room, participant)
    initial_prompt = metadata.get("initialPrompt")
    logger.info(f"🧠 Metadata received: {metadata}")

    if initial_prompt:
        logger.info(f"🧠 Scheduling injection of initial prompt: {initial_prompt}")

        async def inject_prompt_when_ready():
            while agent._agent_output is None:
                await asyncio.sleep(0.1)  # Wait until agent is ready
            logger.info(f"🧠 Injecting initial prompt: {initial_prompt}")
            agent._transcribed_text = initial_prompt
            agent._validate_reply_if_possible()

        asyncio.create_task(inject_prompt_when_ready())
    else:
        # print()
        await agent.say(f"Hi {get_first_name(metadata.get('name'))}! How can I help?")


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm_process,
        ),
    )