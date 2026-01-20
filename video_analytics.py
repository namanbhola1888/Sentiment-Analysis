import settings                              
import cv2                                  
from fer import FER                         
import matplotlib.pyplot as plt             
from moviepy.editor import VideoFileClip     
import datetime
import pandas as pd                          
import numpy as np
import seaborn as sns                       
import os                                    
import subprocess
import speech_recognition as sr
import ffmpeg
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import subprocess
import shlex
import speech_recognition as sr

def get_video():
    
    ''' getting video file from directory'''
    
    myvideo = settings.video_file
    return myvideo


def emotions_face_video(myvideo):
    
    ''' Functions to return dictonary of bounding
    boxes for faces, emotions and scores'''
    
    clip = VideoFileClip(myvideo)
    duration = clip.duration                         
    vidcap = cv2.VideoCapture(myvideo)                      
    i = 0                                                   
    d = []                                                  
    sec = 0                                                 
    frameRate = 1.0                                         
    while i < abs((duration/frameRate) + 1):                
            sec = sec + frameRate
            vidcap.set(cv2.CAP_PROP_POS_MSEC, sec*1000)     
            ret, image = vidcap.read() 
            if ret:                                        
                    cv2.imwrite("image.jpg", image)         
                    img = plt.imread("image.jpg")           
                    detector = FER()                        
                    d = d + detector.detect_emotions(img)   
            i = i + 1                                      
    return d



def emotion_face_video_dataframe(d):
    
    
    ''' Sentiment Analysis based on emotion detection 
      returns list of dictionaries for each image emotions and scores '''
    
    cols = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    rows = []
    for i in range(len(d)):
        rows.append(d[i]['emotions'])

    df = pd.DataFrame(rows, columns=cols)
    return df





FFMPEG_PATH = r"C:\\ffmpeg\\ffmpeg-n8.0-latest-win64-gpl-8.0\bin\\ffmpeg.exe"

def speech_to_text(myvideo):
    '''converts speech to text using recognizer from Google'''

    # video -> mp3
    command = f'"{FFMPEG_PATH}" -i "{myvideo}" Test4.mp3'
    subprocess.run(command, shell=True, check=True)

    # mp3 -> wav
    command = f'"{FFMPEG_PATH}" -i Test4.mp3 Test4.wav'
    subprocess.run(command, shell=True, check=True)

    r = sr.Recognizer()
    with sr.AudioFile('Test4.wav') as source:
        audio = r.record(source, duration=50)
        try:
            text_output = r.recognize_google(audio, language='en-IN')
        except Exception:
            print("Could not understand audio")
            text_output = ""

    return text_output

    

def sentiment_Analysis_text(text_output):
    
    ''' Sentiment Analysis on the text'''
    
    nltk.download('vader_lexicon')                                   
    senti = SentimentIntensityAnalyzer()                            
    senti_text = senti.polarity_scores(text_output)                  
                                                                  
    stopwords = ["a", "this", "is", "and", "i", "to", "for", "very",                 
                 "know", "all", "the", "here", "about", "people", "you", "that"]
    
    reduced = list(filter(lambda w: w not in stopwords, (text_output.lower()).split()))
    
    data =({
    "Words":["Paragraph"] + reduced,
    "Sentiment":[senti_text["compound"]] + [senti.polarity_scores(word)["compound"] 
                                            for word in reduced]
     }) 
    
    return senti_text, data


def video_sentiments(df_faces):
    '''Using Seaborn to show heatmap of emotions with their probablities for video '''
    
    fig, ax = plt.subplots(figsize=(10, 10)) 
    sns.heatmap(df_faces, annot=True, ax=ax)
    
    # Save the plot
    plt.savefig('video_emotions_heatmap.png', dpi=300, bbox_inches='tight')
    print("Saved video emotions heatmap as 'video_emotions_heatmap.png'")
    
    # Show the plot
    plt.show()
    
    plt.close(fig)
    return None

   
def text_sentiments(data):
    ''' returns heatmap of text sentiments for text'''
    
    grid_kws = {"height_ratios": (0.1, 0.007), "hspace": 2}
    f, (ax, cbar_ax) = plt.subplots(2, gridspec_kw=grid_kws)
    f.set_figwidth(20)
    f.set_figheight(3)
    
    # Create DataFrame for heatmap
    heatmap_data = pd.DataFrame(data).set_index("Words").T
    
    sns.heatmap(heatmap_data, center=0, ax=ax, 
                annot=True, cbar_ax=cbar_ax, 
                cbar_kws={"orientation": "horizontal"}, 
                cmap="PiYG")
    
    # Save the plot
    plt.savefig('text_sentiments_heatmap.png', dpi=300, bbox_inches='tight')
    print("Saved text sentiments heatmap as 'text_sentiments_heatmap.png'")
    
    # Show the plot
    plt.show()
    
    plt.close(f)  # Close the figure to free memory
    return None

if __name__ == "__main__":
    
    myvideo = get_video()                                         
    d = emotions_face_video(myvideo)                         
    df = emotion_face_video_dataframe(d)                          
    video_sentiments(df)                                         


    text_output = speech_to_text(myvideo)                         
    print(text_output)                                            
    senti_text, data = sentiment_Analysis_text(text_output)       
    print(senti_text)                                             
    text_sentiments(data)                                        
