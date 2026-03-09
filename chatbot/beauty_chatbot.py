import chainlit as cl
import dotenv
import re

from openai.types.responses import ResponseTextDeltaEvent
from agents import Runner, SQLiteSession

from beauty_agent import beauty_agent
from advanced_skin_analyzer import analyze_skin

dotenv.load_dotenv()


@cl.on_chat_start
async def on_chat_start():

    session = SQLiteSession("conversation_history")
    cl.user_session.set("agent_session", session)

    await cl.Message(
        content="""
👋 Hi! I'm your AI Beauty Assistant.

You can:
• Ask skincare questions
• Ask about ingredients
• Ask where to buy cosmetics
• Upload a selfie to analyze your skin
"""
    ).send()


@cl.on_message
async def on_message(message: cl.Message):

    session = cl.user_session.get("agent_session")

    # IMAGE ANALYSIS
    if message.elements:

        image = message.elements[0]
        image_path = image.path

        await cl.Message(content="📷 Analyzing your skin...").send()

        analysis = analyze_skin(image_path)

        prompt = f"""
A computer vision system already analyzed the user's face.

DO NOT say you cannot analyze images.

Skin Analysis Report:
{analysis}

Tasks:
1. Identify the skin type
2. Explain detected skin conditions
3. Recommend a skincare routine
4. Suggest helpful ingredients
"""

        result = await Runner.run(
            beauty_agent,
            prompt,
            session=session
        )

        clean_output = re.sub(
            r"<thinking>.*?</thinking>",
            "",
            result.final_output,
            flags=re.DOTALL
        )

        await cl.Message(content=clean_output).send()
        return


    # NORMAL CHAT

    result = Runner.run_streamed(
        beauty_agent,
        message.content,
        session=session
    )

    msg = cl.Message(content="")

    async for event in result.stream_events():

        if event.type == "raw_response_event" and isinstance(
            event.data, ResponseTextDeltaEvent
        ):
            await msg.stream_token(token=event.data.delta)

    await msg.update()