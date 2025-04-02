import base64
import logging

from livekit import rtc
from livekit.agents.utils.images.image import encode, EncodeOptions
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
        image_bytes = encode(video_frame, EncodeOptions(format="JPEG", quality=50))

        # ✅ Convert image bytes to Base64
        base64_str = base64.b64encode(image_bytes).decode("utf-8")

        logger.info("Successfully converted image to Base64.")
        print(base64_str, "<<<<<")
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