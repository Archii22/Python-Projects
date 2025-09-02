### 🧠 Smart Chatbot (Tkinter + GPT + Voice AI)

An intelligent desktop chatbot built with Python Tkinter GUI, combining rule-based responses, OpenAI GPT API, and voice interaction (speech-to-text + text-to-speech).
It features a modern UI with dark/light themes, avatars, chat history, sound effects, and export options.

### 🚀 Features

✅ Rule-based + AI responses – simple greetings, time/date, plus GPT-powered answers
✅ Interactive GUI – built using Tkinter with dark/light themes
✅ Voice support

### 🎤 Voice input (speech → text) using Google Speech Recognition

### 🔊 Text-to-speech (TTS) replies using pyttsx3
✅ Chat history – saves logs in both .txt and .json
✅ Export chat – save conversations to a file
✅ Clear chat – reset history anytime
✅ Avatars – user & bot icons in chat
✅ Sound effects – send/receive message sounds (using pygame)
✅ Toggle features – mute/unmute TTS, enable/disable theme

### 🛠️ Tech Stack
Frontend/GUI: Tkinter (Python standard library)
AI/Backend: OpenAI GPT (gpt-3.5-turbo model)
Speech Recognition: speech_recognition + Google API
Text-to-Speech: pyttsx3
Audio Engine: pygame
Environment Management: python-dotenv
Image Handling: Pillow (PIL)

### 📸 Screenshots / Demo
#### Dark Theme :
![Dark Theme](Project%20Chatbot/Demo%20Images/demo%20chatbot%20black%20theme.png)


(Replace with your own screenshots of the running app)

### ⚙️ Setup Instructions
1️⃣ Clone the repository
git clone https://github.com/your-username/smart-chatbot.git
cd smart-chatbot

2️⃣ Create & activate virtual environment
python -m venv venv
#### On Windows:
venv\Scripts\activate
#### On Mac/Linux:
source venv/bin/activate

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Add your OpenAI API key
Create a .env file in the project root:
OPENAI_API_KEY=your_api_key_here

5️⃣ Run the chatbot
python chatbot.py

### 📂 Project Structure
smart-chatbot/
│── chatbot.py              # Main app
│── requirements.txt        # Dependencies
│── .env                    # OpenAI API key
│── user_avatar.png         # User icon
│── bot_avatar.png          # Bot icon
│── send.wav                # Send message sound
│── receive.wav             # Receive message sound
│── chat_log.txt            # Saved text logs
│── chat_log.json           # Saved JSON logs
│── assets/                 # Screenshots

### 👩‍💻 Author
Developed by Archita Ramchandani ✨
Feel free to ⭐ star the repo if you find it useful!
