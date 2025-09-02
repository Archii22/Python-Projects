import tkinter as tk
from tkinter import scrolledtext
from tkinter import filedialog, messagebox
import speech_recognition as sr
# import pyttsx3  # optional if you want a persistent engine
import pygame
import threading
import time
import json
import openai
import os
from dotenv import load_dotenv
from PIL import Image, ImageTk
from datetime import datetime

# -------------------- Setup --------------------
# Initialize mixer (handle systems without audio device)
try:
    pygame.mixer.init()
except Exception as e:
    print("[WARN] pygame mixer init failed:", e)

# Load OpenAI key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# -------------------- GPT + Rule-based Bot --------------------
def ask_gpt(question: str) -> str:
    q = question.lower()

    # Rule-based replies
    if "hello" in q or "hi" in q:
        return "Hey there! How can I assist you today?"
    elif "bye" in q:
        return "Goodbye! Have a good day ahead!"
    elif "your name" in q:
        return "I'm a smart Chatbot created by Archita."
    elif "time" in q:
        return f"The current time is {datetime.now().strftime('%I:%M %p')}"
    elif "day" in q:
        return f"Today is {datetime.now().strftime('%A, %B %d, %Y')}"

    # Fallback to GPT (make sure your openai package supports ChatCompletion)
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question},
            ],
            temperature=0.7,
            max_tokens=150,
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error: {str(e)}"


# -------------------- GUI Class --------------------
class ChatbotGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Smart Chatbot")
        self.root.geometry("500x600")
        self.root.configure(bg="#1e1e1e")

        # Feature toggles
        self.enable_sounds = True     # sound effects on/off
        self.enable_voices = True     # TTS on/off

        # Themes
        self.themes = {
            "dark": {
                "bg": "#121212", "fg": "white",
                "entry_bg": "#2e2e2e", "button_bg": "#0078D7",
                "chat_bg": "#1e1e1e", "insert_bg": "white",
            },
            "light": {
                "bg": "#f5f5f5", "fg": "black",
                "entry_bg": "#ffffff", "button_bg": "#0078D7",
                "chat_bg": "#ffffff", "insert_bg": "black",
            },
        }
        self.current_theme = "dark"

        # --- Chat display ---
        self.chat_display = scrolledtext.ScrolledText(
            root, wrap=tk.WORD, bg="#1e1e1e", fg="white",
            font=("Segoe UI", 11), borderwidth=0, highlightthickness=0,
        )
        self.chat_display.place(x=15, y=15, width=470, height=460)
        self.chat_display.config(state='disabled')

        # --- Separator ---
        self.separator = tk.Frame(root, bg="#444444", height=1)
        self.separator.place(x=10, y=490, width=480)

        # --- Entry field ---
        self.entry_field = tk.Entry(
            root, bg="#2e2e2e", fg="white", insertbackground="white",
            font=("Segoe UI", 12), relief=tk.FLAT,
        )
        self.entry_field.place(x=15, y=490, width=370, height=45)
        self.entry_field.bind("<Return>", self.send_message)

        # --- Send button ---
        self.send_button = tk.Button(
            root, text="Send ➤", command=self.send_message,
            bg="#0078D7", fg="white", font=("Segoe UI", 10, "bold"),
            activebackground="#005a9e", relief=tk.FLAT,
        )
        self.send_button.place(x=395, y=490, width=90, height=45)

        # --- Action row buttons (bottom) ---
        # Export Chat
        self.export_button = tk.Button(
            root, text="Export Chat", command=self.export_chat,
            bg="#28a745", fg="white", font=("Segoe UI", 9, "bold"), relief=tk.FLAT,
        )
        self.export_button.place(x=15, y=550, width=110, height=30)

        # Clear Chat
        self.clear_button = tk.Button(
            root, text="Clear Chat", command=self.clear_chat,
            bg="#d9534f", fg="white", font=("Segoe UI", 9, "bold"), relief=tk.FLAT,
        )
        self.clear_button.place(x=135, y=550, width=100, height=30)

        # Toggle Voice (TTS on/off)
        self.voice_toggle_button = tk.Button(
            root, text="Mute Voice", command=self.toggle_voice,
            bg="#6c757d", fg="white", font=("Segoe UI", 9, "bold"), relief=tk.FLAT,
        )
        self.voice_toggle_button.place(x=245, y=550, width=100, height=30)

        # Voice Input (speech → text)
        self.voice_input_button = tk.Button(
            root, text="🎤 Speak", command=self.voice_input,
            bg="#5bc0de", fg="white", font=("Segoe UI", 9, "bold"), relief=tk.FLAT,
        )
        self.voice_input_button.place(x=355, y=550, width=100, height=30)

        # Toggle Theme
        self.toggle_button = tk.Button(
            root, text="Toggle Theme", command=self.toggle_theme,
            bg="#444", fg="white", font=("Segoe UI", 9), relief=tk.FLAT,
        )
        self.toggle_button.place(x=370, y=520, width=115, height=25)

        # --- Load avatars ---
        try:
            self.user_avatar_img = Image.open("user_avatar.png").resize((30, 30))
            self.user_avatar = ImageTk.PhotoImage(self.user_avatar_img)
            self.bot_avatar_img = Image.open("bot_avatar.png").resize((30, 30))
            self.bot_avatar = ImageTk.PhotoImage(self.bot_avatar_img)
        except Exception:
            self.user_avatar = None
            self.bot_avatar = None

        # Apply theme once widgets exist
        self.apply_theme()

        # Load chat history AFTER widgets are ready
        self.load_chat_log()

    # -------------------- Logging --------------------
    def log_message(self, sender: str, message: str):
        """Append chat message to text and JSON logs."""
        log_txt = "chat_log.txt"
        log_json = "chat_log.json"

        # Text log
        try:
            with open(log_txt, "a", encoding="utf-8") as f:
                f.write(f"{sender}: {message}\n")
        except Exception as e:
            print("[WARN] Writing TXT log failed:", e)

        # JSON log
        log_entry = {
            "sender": sender,
            "message": message,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        try:
            data = []
            if os.path.exists(log_json) and os.path.getsize(log_json) > 0:
                with open(log_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
            data.append(log_entry)
            with open(log_json, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print("[WARN] Writing JSON log failed:", e)

    # -------------------- Load Chat History --------------------
    def load_chat_log(self):
        """Load previous chat messages from JSON if present and valid."""
        path = "chat_log.json"
        if not (os.path.exists(path) and os.path.getsize(path) > 0):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.chat_display.config(state="normal")
            for entry in data:
                sender = entry.get("sender", "User")
                message = entry.get("message", "")
                is_user = (sender.lower() == "user")
                self._insert_message(message, is_user=is_user, temp=False)
            self.chat_display.config(state="disabled")
            self.chat_display.yview(tk.END)
        except Exception as e:
            print("❌ Error loading logs:", e)

    # -------------------- Sound --------------------
    def play_sound(self, sound_file: str):
        if not os.path.exists(sound_file):
            return
        def _play():
            try:
                snd = pygame.mixer.Sound(sound_file)
                snd.play()
                time.sleep(snd.get_length())
            except Exception as e:
                print("[WARN] play_sound failed:", e)
        threading.Thread(target=_play, daemon=True).start()

    # -------------------- Clear / Export --------------------
    def clear_chat(self):
        """Clear chat window and reset logs."""
        self.chat_display.config(state="normal")
        self.chat_display.delete(1.0, tk.END)
        self.chat_display.config(state="disabled")
        for p in ("chat_log.txt", "chat_log.json"):
            try:
                if os.path.exists(p):
                    open(p, "w", encoding="utf-8").close()
            except Exception as e:
                print("[WARN] Clearing log failed for", p, e)
        self.display_message("Chat cleared.", is_user=False)

    def export_chat(self):
        """Export the current chat to a text file chosen by the user."""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text file", "*.txt"), ("All files", "*.*")],
            title="Save Chat As",
        )
        if not file_path:
            return
        try:
            self.chat_display.config(state="normal")
            chat_content = self.chat_display.get(1.0, tk.END)
            self.chat_display.config(state="disabled")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(chat_content)
            messagebox.showinfo("Export Chat", f"Chat successfully saved to:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Export Chat", f"Failed to save chat:\n{str(e)}")

    # -------------------- TTS (speak) --------------------
    def speak(self, text: str):
        if not self.enable_voices:
            return
        def _speak():
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty("rate", 160)
                engine.setProperty("volume", 1.0)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception as e:
                print("[WARN] TTS speak failed:", e)
        threading.Thread(target=_speak, daemon=True).start()

    def toggle_voice(self):
        self.enable_voices = not self.enable_voices
        if self.enable_voices:
            self.voice_toggle_button.config(text="Mute Voice", bg="#6c757d")
        else:
            self.voice_toggle_button.config(text="Enable Voice", bg="#28a745")

    # -------------------- Voice Input --------------------
    def voice_input(self):
        recognizer = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                self.display_message("Listening...", is_user=False)
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=8)
                text = recognizer.recognize_google(audio)
                self.entry_field.delete(0, tk.END)
                self.entry_field.insert(0, text)
                self.send_message()  # auto-send after recognition
        except sr.UnknownValueError:
            self.display_message("Sorry, I couldn’t understand that.", is_user=False)
        except sr.RequestError:
            self.display_message("Speech recognition service error.", is_user=False)
        except Exception as e:
            self.display_message(f"Voice input error: {e}", is_user=False)

    # -------------------- Chat Flow --------------------
    def send_message(self, event=None):
        user_message = self.entry_field.get().strip()
        if not user_message:
            return
        if self.enable_sounds:
            self.play_sound("send.wav")
        self.entry_field.delete(0, tk.END)
        self.display_message(user_message, is_user=True)
        self.log_message("User", user_message)
        threading.Thread(target=self.delayed_response, args=(user_message,), daemon=True).start()

    def delayed_response(self, user_message: str):
        self.display_message("Bot is typing...", is_user=False, temp=True)
        time.sleep(1.2)
        bot_reply = ask_gpt(user_message)
        # Remove temp line
        self.chat_display.config(state="normal")
        try:
            self.chat_display.delete("end-3l", "end-1l")
        except Exception:
            pass
        self.chat_display.config(state="disabled")

        self.display_message(bot_reply, is_user=False)
        self.log_message("Bot", bot_reply)
        self.speak(bot_reply)
        if self.enable_sounds:
            self.play_sound("receive.wav")

    def _insert_message(self, message: str, is_user: bool = True, temp: bool = False):
        avatar = self.user_avatar if is_user else self.bot_avatar
        name = "You" if is_user else "Bot"
        if avatar and not temp:
            self.chat_display.image_create(tk.END, image=avatar)
        prefix = f" {name}: " if not temp else ""
        self.chat_display.insert(tk.END, f"{prefix}{message}\n\n")

    def display_message(self, message: str, is_user: bool = True, temp: bool = False):
        self.chat_display.config(state="normal")
        self._insert_message(message, is_user=is_user, temp=temp)
        self.chat_display.yview(tk.END)
        self.chat_display.config(state="disabled")

    # -------------------- Theming --------------------
    def apply_theme(self):
        theme = self.themes[self.current_theme]
        self.root.configure(bg=theme["bg"])
        self.chat_display.configure(bg=theme["chat_bg"], fg=theme["fg"])
        self.entry_field.configure(bg=theme["entry_bg"], fg=theme["fg"], insertbackground=theme["insert_bg"])
        self.send_button.configure(bg=theme["button_bg"], fg="white")
        self.toggle_button.configure(
            bg="#444" if self.current_theme == "dark" else "#ccc",
            fg="white" if self.current_theme == "dark" else "black",
        )

    def toggle_theme(self):
        self.current_theme = "light" if self.current_theme == "dark" else "dark"
        self.apply_theme()


# -------------------- Run App --------------------
if __name__ == "__main__":
    try:
        print("Starting chatbot...")
        root = tk.Tk()
        app = ChatbotGUI(root)
        print("Chatbot window created, launching mainloop...")
        root.mainloop()
    except Exception:
        import traceback
        print("❌ Error occurred:")
        traceback.print_exc()
        input("Press Enter to exit...")
