from flask import Flask , request
import random
from pymessenger.bot import Bot
#####
import cv2
import os
import numpy as np
from pickle import load
import json
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.models import Model , model_from_json
from tensorflow.python.keras import applications
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.utils import to_categorical
from PIL import ImageFont, ImageDraw, Image
import requests
from googletrans import Translator
##############
tokenize_path = "tokenizer_all_caps.pkl"
max_length = 51
#####
json_file = open("Model_Structure.json")
json_string = json.load(json_file)
json_file.close()

caption_model = model_from_json(json_string)
caption_model.load_weights("Image_Caption_Model_Weights.h5")

img_model = applications.InceptionV3(weights = "imagenet", include_top=True, input_shape = (299, 299, 3))
img_model = Model(inputs = img_model.input, outputs = img_model.layers[-2].output)
with open(tokenize_path, "rb") as fp:   # Unpickling
    tokenizer = load(fp)
############

app = Flask(__name__)
ACCESS_TOKEN = os.environ['ACCESS_TOKEN']
VERIFY_TOKEN = os.environ['VERIFY_TOKEN']

bot = Bot(ACCESS_TOKEN)

@app.route("/"  , methods = ["GET" , "POST"])

def receive_message():
    
    if request.method == "GET" :
        token_sent = request.args.get("hub.verify_token")
        
        return verify_fb_token(token_sent)
    else:
        output = request.get_json()
        
        for event in output["entry"]:
            
            messaging = event["messaging"]
            for message in messaging:
                
                if message.get("message"):
                    recipient_id = message["sender"]["id"]
                    
                    if message["message"].get("text"):
                        response_sent_text = "https://scontent.xx.fbcdn.net/v/t1.15752-9/48269153_208016223421095_5548286354095341568_n.png?_nc_cat=106&_nc_ad=z-m&_nc_cid=0&_nc_ht=scontent.xx&oh=bdb21f0a3bc84508c4dcb5e7e91feb2a&oe=5C9EFD8D"
                        send_photo(recipient_id , response_sent_text)
                    
                    if message["message"].get("attachments"):
                        response_sent_nontext = "Pending....."
                        send_message(recipient_id , response_sent_nontext)
                        
                        print(message["message"])
                        print(message["message"]["attachments"][0]["payload"])
                        
                        photo = download_photo(message["message"]["attachments"][0]["payload"]["url"])
                        print("Success Download")
                        
                        feature = extract_features(photo , img_model)
                        text = generate_desc(caption_model , tokenizer , feature , max_length)
                        translator = Translator()
                        translated = translator.translate(text , dest = "zh-tw")
                        print(translated)
                        print(translated.text)
                        print("Success Generation")
                        send_message(recipient_id , text)
                        send_message(recipient_id , translated.text)
    return "Message Processed"




def download_photo(url):
    imgResp = requests.get(url)  # 640 * 480
    imgNp = np.array(bytearray(imgResp.content),dtype=np.uint8)
    photo=cv2.imdecode(imgNp,-1)
    return photo

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def extract_features(image, model):
    # resize
    image = cv2.resize(image, (299,299))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the InceptionV3 model
    image = applications.inception_v3.preprocess_input(image)
    # get features
    feature = model.predict(image, verbose=0)
    return feature


def next_word_prob(self,model,in_text, photo):
    #self,model, tokenizer, in_text, max_length
    sequence = tokenizer.texts_to_sequences([in_text])[0]   # integer encode input sequence
    sequence = pad_sequences([sequence], maxlen=max_length)   # pad input
    yhat = model.predict([photo,sequence], verbose=0)   # predict next word
    return yhat

def prob_select(self, yhat, seqe, k):
    # self, yhat, tokenizer, seqe, k
    all_candidates = list()
    seq, score = seqe
    max_list = yhat.argsort()[-10:][::-1] #####
    for j in max_list:
        candidate = [seq + [self.word_for_id(j)], score * yhat[j]]
        all_candidates.append(candidate)
    return all_candidates

def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'startseq' # start the generation process
    text = ""
    for i in range(max_length):   # iterate over the whole length of the sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]   # integer encode input sequence
        sequence = pad_sequences([sequence], maxlen=max_length)   # pad input
        yhat = model.predict([photo,sequence], verbose=0)   # predict next word
        yhat = np.argmax(yhat)   # convert probability to integer
        word = word_for_id(yhat, tokenizer)   # map integer to word
        # stop if we cannot map the word
        if (word == None) or (word == "endseq"):
            break
        # append as input for generating the next word
        in_text += ' ' + word
        text += " "+ word
    return text
def generate_desc_beam_search(self, model, photo, k):
    # self, model, tokenizer, photo, vocab_size, max_length, k
    in_text = 'startseq' # start the generation process
    seqe = [[list(), 1.0]]
    
    for m in range(max_length):   # iterate over the whole length of the sequence
        if m == 0:
            yhat = self.next_word_prob(model, in_text, photo)
            seqe = self.prob_select(yhat[0], seqe[0], k)
            seqe = seqe[:k] #####
            for i in range(len(seqe)):
                seqe[i][0] = [in_text] + seqe[i][0]
        else:
            seqe_tmp = []
            for i in range(k):
                list_string = seqe[i][0]
                if list_string[-1] == "endseq":
                    yhat = np.array([[0]*vocab_size])
                    yhat[0][3] = 1
                else:
                    yhat = self.next_word_prob(model," ".join(list_string), photo)
                ps = self.prob_select(yhat[0], seqe[i], k)
                seqe_tmp.extend(ps)
            ordered =  sorted(seqe_tmp,  key=lambda tup:tup[1], reverse=True)
            seqe = ordered[:k]
    for i in range(k):
        text = ""
        for j in range(1, max_length):
            if seqe[i][0][j] != 'endseq':
                text += " " +seqe[i][0][j]
            else:
                break
        seqe[i][0] = [text]
    return seqe
def  model_predict(recipient_id , input):
    
    #####Prediction
    
    
    #####Output String
    output = "output"
    return output

def send_prediction(recipient_id , output):
    
    bot.send_text_message(recipient_id , output)





def verify_fb_token(token_sent):
    
    if token_sent == VERIFY_TOKEN:
        return request.args.get("hub.challenge")
    return "Invalid verification token"

def send_photo(recipient_id , response):
    bot.send_image_url(recipient_id , response)
    
    return "Success"

def send_message(recipient_id , response):
    
    bot.send_text_message(recipient_id , response)
    return "success"

def get_message():
    
    sample_responses = ["You are Stunning" , "Great" , "We're proud of you!" ]
    
    return random.choice(sample_responses)



if __name__ == "__main__":
    app.run()
