import os
import cv2
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering
import speech_recognition as sr
import pyttsx3
import google.generativeai as genai
import requests
from imageai.Detection import ObjectDetection

# Initialize speech recognition and text-to-speech engines
r = sr.Recognizer()
engine = pyttsx3.init()

# Configure the Google Generative AI model
genai.configure(api_key='AIzaSyC4VD7VA7gGt9jYUBs8GN1r2qRoW7vNHao')

# Function to speak the response
def speak(text):
    try:
        voices = engine.getProperty('voices')
        if len(voices) > 1:
            engine.setProperty('voice', voices[1].id)  # Set the voice to female
        else:
            print("Female voice not available, using default voice.")
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"An error occurred in the speak function: {e}")

# Define function to process speech input
def get_audio():
    with sr.Microphone() as source:
        print("Listening...")
        r.adjust_for_ambient_noise(source, duration=1)
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            print(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand what you said.")
            return None
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
            return None

def capture_and_save_image(filename="webcam_capture.jpg"):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error opening webcam!")
        return

    print("Press 'q' to capture image.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame!")
            break

        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to capture
            cv2.imwrite(filename, frame)
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Image captured and saved as {filename}")

def analyze_image(image_path, question):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

    image = Image.open(image_path).convert('RGB')
    inputs = processor(image, question, return_tensors="pt")
    out = model.generate(**inputs)
    answer = processor.decode(out[0], skip_special_tokens=True)
    return answer

def speech_to_text(timeout=5, phrase_time_limit=10):
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        print("Adjusting for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source, duration=1)  # Reduced duration for quicker calibration
        print("Listening for your question...")
        try:
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        except sr.WaitTimeoutError:
            print("Listening timed out while waiting for phrase to start")
            return None

    try:
        question = recognizer.recognize_google(audio)
        print(f"You asked: {question}")
        return question
    except sr.UnknownValueError:
        print("Sorry, I did not understand that.")
        return None
    except sr.RequestError:
        print("Could not request results; check your network connection.")
        return None

def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Function to get weather information for a specific location (using OpenWeatherMap API)
def get_weather(city):
    api_key = '8d9e0918b17270bd2dbdef6c255be66d'
    url = f'http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric'
    response = requests.get(url).json()
    if 'main' in response:
        temp = response['main']['temp']
        description = response['weather'][0]['description']
        return f"The current weather in {city} is {description} with a temperature of {temp:.1f}°C."
    else:
        return "Sorry, I couldn't retrieve weather information for that location."

# Function to extract the city name from user input
def extract_city_name(user_input):
    words = user_input.lower().split()
    for i, word in enumerate(words):
        if word == "weather" and i + 2 < len(words) and words[i + 1] == "in":
            return words[i + 2].capitalize()  # Extract the next word after "in" as the city name
    return None  # Return None if city extraction fails

# Function to get a response from Google Generative AI using Gemini API
def get_google_response(prompt):
    try:
        model = genai.GenerativeModel(model_name="gemini-1.0-pro-latest")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"An error occurred while getting response from Google Generative AI: {e}")
        return "I'm sorry, I can't process that right now."

# Function to generate a response based on the captured image using ImageAI
def generate_response_from_image(image_name):
    detector = ObjectDetection()
    model_path = os.path.join(os.getcwd(), "yolo-tiny.h5")  # Use the TinyYOLOv3 model
    if not os.path.exists(model_path):
        print(f"Model file not found at path: {model_path}")
        return "Model file not found, please ensure it is placed in the correct directory."

    detector.setModelTypeAsTinyYOLOv3()  # Set the model type to TinyYOLOv3
    detector.setModelPath(model_path)
    detector.loadModel()

    output_image_path = os.path.join(os.getcwd(), "detected_image.jpg")
    detections = detector.detectObjectsFromImage(input_image=image_name, output_image_path=output_image_path)
    descriptions = [d["name"] for d in detections]
    if descriptions:
        return f"I see the following in the image: {', '.join(descriptions)}."
    else:
        return "I couldn't recognize anything in the image."

# Start Buji's conversation
greeting = "Hello bhairava! I'm Buji. How can I help you?"
print(greeting)
speak(greeting)

while True:
    user_input = get_audio()
    if user_input:
        if "capture" in user_input.lower():
            filename = "webcam_capture.jpg"
            capture_and_save_image(filename)
            print("Please ask a question about the image (or say 'exit' to quit):")
            question = speech_to_text()
            if question is None:
                continue
            if question.lower() == 'exit':
                break
            answer = analyze_image(filename, question)
            print(f"Answer: {answer}")
            text_to_speech(f"The answer is: {answer}")
        elif "weather" in user_input.lower():
            # Extract the city name from the user input
            city = extract_city_name(user_input)
            if city:
                weather_response = get_weather(city)
                print(f"Buji: {weather_response}")
                speak(weather_response)  # Speak the weather response
            else:
                print("Could not determine the city.")
                speak("Could not determine the city.")
        elif "exit" in user_input.lower():
            farewell = "Goodbye! Have a great day."
            print(f"Buji: {farewell}")
            speak(farewell)
            break
        else:
            # Get a response from Google Generative AI
            response = get_google_response(user_input)
            print(f"Buji: {response}")
            speak(response)
