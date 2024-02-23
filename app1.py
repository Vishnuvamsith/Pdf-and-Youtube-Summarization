import os
from langchain.llms import OpenAI
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
prompt="please summarize this youtube video in points,u will be give the it's transcript.this is the transcript:"
def getscript(videourl):
    VI=videourl.split("=")[1]
    transcrpt_list=YouTubeTranscriptApi.get_transcript(VI)
    transcript=""
    for i in transcrpt_list:
        transcript+=" "+i["text"]
    return transcript
def generate(transcript,prompt):
    llm=OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"),temperature=0.7)
    return llm.invoke(prompt+transcript)
st.title("youtube video summarizer using langchain")
video_url=st.text_input("please enter the url")
if video_url:
    video_id=video_url.split("=")[1]
    st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg",use_column_width=True)
if st.button("click here for detailed notes"):
    transcript=getscript(video_url)
    if transcript:
        output=generate(prompt,transcript)
        st.markdown("#Detailed Notes")
        st.write(output)
