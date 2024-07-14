import gradio as gr
import transformers
from transformers import pipeline

classifier = pipeline("text-classification", model="mazed/distilbert-amazon-shoe-review")

def predict(review):
    prediction = classifier(review)
    print(prediction)
    stars = prediction[0]['label']
    stars = (int)(stars.split('_')[1])+1
    score=100*prediction[0]['score']
    return "{} {:.0f}%".format("\U00002B50"*stars, score)

iface=gr.Interface(fn=predict, inputs='text', outputs='text')
iface.launch()
    