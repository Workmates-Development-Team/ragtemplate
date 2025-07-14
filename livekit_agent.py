# # livekit_agent.py

# import os
# import logging
# import asyncio
# import openai as _openai_sdk
# from dotenv import load_dotenv

# from livekit import agents
# from livekit.agents import AgentSession, Agent, WorkerOptions, JobContext, RoomInputOptions
# from livekit.plugins import openai, sarvam, noise_cancellation, silero

# from rag_utils import embed_text_or_image, retrieve_similar_chunks

# load_dotenv()
# logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
# logger = logging.getLogger("rag-agent")

# # The very first thing the agent should say
# SESSION_INSTRUCTION = (
#     "Hello, what do you want me to ask from the files that have been uploaded? "
#     "Use the context of files in your RAG database."
# )

# class Assistant(Agent):
#     # no custom _init_; we'll pass instructions/tools when instantiating
#     async def on_enter(self):
#         # fires once, right when the session starts
#         await self.session.generate_reply(instructions=self._instructions)


# def build_rag_prompt(user_q: str) -> str:
#     emb = embed_text_or_image(user_q, content_type="text")
#     top = retrieve_similar_chunks(emb, top_k=3)
#     context = "\n\n".join(chunk_text for *_, chunk_text in top)
#     return f"Context:\n{context}\n\nQuestion: {user_q}\nAnswer:"

# async def _on_conversation(ev, session: AgentSession):
#     role = ev.item.role.name.upper() if hasattr(ev.item.role, "name") else ""
#     text = "".join(ev.item.content)
#     if role == "USER":
#         prompt = build_rag_prompt(text)
#         await session.generate_reply(instructions=prompt)
#     elif role == "ASSISTANT":
#         print("\n🗣  Agent:", text)

# async def entrypoint(ctx: JobContext):
#     # 1) Build the OpenAI SDK client
#     sdk_client = _openai_sdk.AsyncClient(api_key=os.getenv("OPENAI_API_KEY"))

#     # 2) Set up STT → LLM → TTS with Sarvam & Silero VAD
#     session = AgentSession(
#         stt=sarvam.STT(
#             language=os.getenv("SARVAM_STT_LANG", "en-IN"),
#             model=os.getenv("SARVAM_STT_MODEL", "saarika:v2.5"),
#         ),
#         llm=openai.LLM(client=sdk_client, model="gpt-4o", temperature=0.3),
#         tts=sarvam.TTS(
#             target_language_code=os.getenv("SARVAM_TTS_LANG", "en-IN"),
#             model=os.getenv("SARVAM_TTS_MODEL", "bulbul:v2"),
#             speaker=os.getenv("SARVAM_SPEAKER", "anushka"),
#         ),
#         vad=silero.VAD.load(),
#     )

#     # 3) Connect to the LiveKit room (audio only)
#     await ctx.connect()

#     # 4) Register a synchronous callback that schedules our async handler
#     session.on("conversation_item_added", lambda ev: asyncio.create_task(_on_conversation(ev, session)))

#     # 5) Instantiate your Agent *with* the required instructions and a tools list
#     assistant = Assistant(
#         instructions=SESSION_INSTRUCTION,  # <-- required
#         tools=[],                          # <-- tools is keyword‐only and required
#     )

#     # 6) Start the voice session – this will activate STT/TTS and block until you quit with [Q]
#     await session.start(
#         agent=assistant,
#         room=ctx.room,
#         room_input_options=RoomInputOptions(
#             noise_cancellation=noise_cancellation.BVCTelephony(),
#             video_enabled=False,
#         ),
#     )

#     # 7) Kick off the very first greeting
#     await session.generate_reply(instructions=SESSION_INSTRUCTION)

# if __name__ == "__main__":
#     agents.cli.run_app(
#         WorkerOptions(
#             entrypoint_fnc=entrypoint,
#             agent_name=os.getenv("AGENT_NAME", "rag-voice-agent"),
#         )
#     )





import os
import logging
import asyncio
import openai as _openai_sdk
from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentSession, Agent, WorkerOptions, JobContext, RoomInputOptions
from livekit.plugins import openai, sarvam, noise_cancellation, silero

from rag_utils import embed_text_or_image, retrieve_similar_chunks

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("rag-agent")

# 1) Initial system greeting
SESSION_INSTRUCTION = (
    "Hello, what do you want me to ask from the files that have been uploaded? "
    "Use the context of files in your RAG database."
)

# 2) Strict rule: only answer from context
SYSTEM_RULES = (
    "You are a helpful assistant. You must answer using ONLY the information in the provided context. "
    "Do NOT add or infer any information beyond that. If the context does not contain the answer, "
    "respond with: \"I don't know based on the provided files.\""
)

class Assistant(Agent):
    async def on_enter(self):
        await self.session.generate_reply(instructions=self.instructions)

def build_rag_prompt(user_q: str) -> str:
    emb = embed_text_or_image(user_q, content_type="text")
    top = retrieve_similar_chunks(emb, top_k=3)
    ctx = "\n\n".join(chunk_text for *_, chunk_text in top)
    return f"Context:\n{ctx}\n\nQuestion: {user_q}\nAnswer:"

async def _on_conversation_async(ev, session: AgentSession):
    role = ev.item.role.name.upper() if hasattr(ev.item.role, "name") else ""
    text = "".join(ev.item.content)

    if role == "USER":
        rag_body = build_rag_prompt(text)
        full_instructions = SYSTEM_RULES + "\n\n" + rag_body
        await session.generate_reply(instructions=full_instructions)

    elif role == "ASSISTANT":
        print("\n🗣  Agent:", text)

async def entrypoint(ctx: JobContext):
    sdk_client = _openai_sdk.AsyncClient(api_key=os.getenv("OPENAI_API_KEY"))

    session = AgentSession(
        stt=sarvam.STT(
            language=os.getenv("SARVAM_STT_LANG", "en-IN"),
            model=os.getenv("SARVAM_STT_MODEL", "saarika:v2.5"),
        ),
        llm=openai.LLM(client=sdk_client, model="gpt-4o", temperature=0.3),
        tts=sarvam.TTS(
            target_language_code=os.getenv("SARVAM_TTS_LANG", "en-IN"),
            model=os.getenv("SARVAM_TTS_MODEL", "bulbul:v2"),
            speaker=os.getenv("SARVAM_SPEAKER", "anushka"),
        ),
        vad=silero.VAD.load(),
    )

    await ctx.connect()

    # sync callback schedules our async handler
    session.on("conversation_item_added", lambda ev: asyncio.create_task(_on_conversation_async(ev, session)))

    assistant = Assistant(
        instructions=SESSION_INSTRUCTION,
        tools=[],
    )

    await session.start(
        agent=assistant,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVCTelephony(),
            video_enabled=False,
        ),
    )

if __name__ == "__main__":
    agents.cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name=os.getenv("AGENT_NAME", "rag-voice-agent"),
        )
    )