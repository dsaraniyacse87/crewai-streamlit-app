import os
import streamlit as st
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from openai import OpenAI
import base64
from io import BytesIO
from PIL import Image

# Expect OPENAI_API_KEY to be set (or AZURE_* if using AZURE; adjust model class accrordingly)
llm = ChatOpenAI(
    model="gpt-4o-mini", 
    temperature=0.2,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    # For Azure, use these parameters instead:)
)

# ------------------ OpenAI Client for Image Generation ------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ------------------ Agent Definition ------------------

# 1) Router agent - decides which specialist should handle the query
router_agent = Agent(
    role="Router", 
    goal=(
        "Look at the user's query and decide which specialist agent is best suited to handle it. " \
        "Return exactly one of these lables: 'python', 'general'." 
        "Use 'python' for questions about Python code, debugging, packages,"
        "environment, or stack traces. Use 'general' for all other questions."
    ),
    backstory=(
    "You are a routing agent that only decides which specialist is best suited"
    "for the question. You do not answer the question yourself. " \
    ),
    llm=llm,
    verbose=True,
)

# 2) Specialist: Python / technical helper
python_agent = Agent(
    role="PythonHelper", 
    goal="Help the user with Python, debugging, environments, and technical issues.",
    backstory="You are a strong Python developer and DevOps helper.",
    llm=llm,
    verbose=True,
)

#3) Specialist: General answer helper
general_agent = Agent(
    role="GeneralHelper", 
    goal="Answer the user's questions clearly and concisely in natural language that.",
    backstory="You are a helpful, concise assistant for all non-Python questions.",
    llm=llm,
    verbose=True,
)

#4) Social media agent - turns an answer into a social media post
social_media_agent = Agent(
    role="SocialMediaCreator",
    goal=(
        "Turn technical answers into engaging, concise social media posts. "
    ),
    backstory=(
        "You are a social media copywriter. You take an explanation and"
        "turn it into a short, engaging post suitable for LinkedIn or X (Twitter)"
    ),
    llm=llm,
    verbose=True,
)

# Image agent - creates an image prompt based on the answer
image_agent = Agent(
    role="ImagePromptDesigner",
    goal=(
        "Design clear, vivid image prompts suitable for AI image generator "
        "based on the assistant answer and the original question."
    ),
    backstory=(
        "You are a visual storyteller. Given text, you describe a single image that"
        "would illustrate the core idea. You do NOT generate the image yourself," 
        "only a prompt that another system can use."
    ),
    llm=llm,
    verbose=True,
)   

# Image Creation agent - conceptually responsible for creating the image
# (implementation is done directly via OpenAI client int the app)
image_creation_agent = Agent(
    role="ImageCreator",
    goal=(
        "Given a finalized image prompt, generate a single illustrative image "
        "using an AI image model."
    ),
    backstory=(
        "You orchestrate image generation using a given prompt. "
        "In this app, the actual API call is made in Python using the OpenAI client."
    ),
    llm=llm,
    verbose=True,
)   

AGENT_MAP = {
    "python": python_agent, 
    "general": general_agent,
}

st.set_page_config(page_title="CrewAI Multi-Agent + Social Media", page_icon="🤖")
st.title("CrewAI Multi-Agent Router with Social Media Post (Streamlit + Docker)")

user_query = st.text_area("Ask a question:", height=120)

api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    masked = api_key[:4] + "..." + api_key[-4:]
    st.sidebar.write(f"OpenAI API Key: {masked}")
else:
    st.sidebar.write("OpenAI API Key not set")

st.sidebar.write("Tech Stack Used")
st.sidebar.write("OpenAI")
st.sidebar.write("CrewAI for Orchestration")
st.sidebar.write("Langchai")
st.sidebar.write("Streamlit for UI")

if st.button("Run Agent"):
    if not user_query.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Routng your query to bthe best agent..."):
            routing_task = Task(
                description=(
                    "Given this user question, output only one word:" \
                    "'python' or 'general' ." \
                    f"User question: {user_query}"
                ),
                agent = router_agent,
                expected_output="One of two words: 'python' or 'general', nothing else.",
            )
        
            routing_crew = Crew(
                agents=[router_agent], 
                tasks = [routing_task],
                verbose=True,
            )

            routing_result = routing_crew.kickoff()
            routing_label = routing_result

        st.write(f"Router Agent Decision: `{routing_label}`")
        
        # Fallback if router output is unexpected
        chosen_label = "python" if routing_label == "python" else "general"
        chosen_agent = AGENT_MAP.get(chosen_label)

        with st.spinner(f"Running the {chosen_label} agent to answer the question..."):
            answer_task = Task(
                description=(
                    f"Answer the user's question: {user_query}"
                ),
                agent = chosen_agent,
                expected_output="A short, clear natural language answer (2-6 sentences).",
            )
        
            answer_crew = Crew(
                agents=[chosen_agent], 
                tasks = [answer_task],
                verbose=True,
            )

            answer_result = answer_crew.kickoff()
        st.subheader("Answer:")
        st.write(answer_result)

        #------- Step3: Social Media Post Creation -------
        with st.spinner("Creating a social media post based on the answer..."):
            social_media_task = Task(
                description=(
                    "You are given an assistat answer and the original user question. " \
                    "Create a short social media post (max 200 characters) that " \
                    "summarizes the key idea in an engaging way. Do NOT include " \
                    "hastags unless they are very relevant. Do NOT include links. " \
                    f"\n\n User question: {user_query}\n\n Answer: {answer_result}"
                ),
                agent = social_media_agent,
                expected_output="A single social media post text, under 200 characters.",
            )
        
            social_media_crew = Crew(
                agents=[social_media_agent], 
                tasks = [social_media_task],
                verbose=True,
            )

            social_media_result = social_media_crew.kickoff()
            st.subheader("Social Media Post:")
            st.write(social_media_result)
        #------- Step4: Image Prompt Creation -------
        with st.spinner("Creating an image prompt based on the answer..."):
            image_prompt_task = Task(
                description=(
                    "You are given an original user question and the assistant answer. "
                    "Design ONE detaied, vivid image prompt suitable for an AI image" \
                    "generation model (e.g., DALL-E or Stable Diffusion) that would"
                    "visually illustrate the core idea. "
                    "Describe suject, style, perspective, and key elements clearly." \
                    "Do NOT wiret instructions about aspect ratio or camera model." \
                    "just a natuaral language description of the image. Ask the prompt to return the generated image file. \n\n"                
                    f"\n\n User question: {user_query}\n\n Answer: {answer_result}"
                ),
                agent = image_agent,
                expected_output="A single paragraph image prompt in natural language (1-3 sentences)",
            )

            image_prompt_crew = Crew(
                agents=[image_agent], 
                tasks = [image_prompt_task],
                verbose=True,
            )
            image_prompt_result = image_prompt_crew.kickoff()

        st.subheader("Generated Image Prompt:") 
        st.write(image_prompt_result)

        #------- Step5: Image Creation -------
        with st.spinner("Generating an image based on the prompt..."):
            try:
                image_prompt_text = str(image_prompt_result).strip()
                if not image_prompt_text:
                    raise ValueError("Image prompt is empty. Cannot generate image.")
                
                st.sidebar.write(f"Using image prompt: {image_prompt_text}")
                st.sidebar.write(image_prompt_text[:200] + "..." if len(image_prompt_text) > 200 else "")

                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OpenAI API key not set. Cannot generate image.")
                
                image_client = OpenAI(api_key=api_key)
                img_response = image_client.images.generate(
                    model="gpt-image-1",
                    prompt=image_prompt_text,
                    size="1024x1024",
                    n=1,
                )

                # safely extract URL
                if not hasattr(img_response, "data") or not img_response.data:
                    raise ValueError("Unexpected image response format: " + str(img_response))
                
                first_item = img_response.data[0]
                image_url = getattr(first_item, "url", None)
                if not image_url:
                    b64 = getattr(first_item, "b64_json", None)
                    if b64:
                        b64_data = first_item.b64_json
                        image_bytes = base64.b64decode(b64_data)
                        image = Image.open(BytesIO(image_bytes))
                        st.image(image, caption="Generated Image", use_container_width=True)
                    else:
                        raise ValueError("No URL or base64 data found in image response.")
            except Exception as e:
                st.error(f"Error generating image: {str(e)}")
