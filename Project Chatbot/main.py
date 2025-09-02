import tkinter as tk
from tkinter import scrolledtext
import threading
import time
import openai
import os
from dotenv import load_dotenv
from PIL import Image , ImageTk
from datetime import datetime 

# load the openai field from the env file:
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# GPT Bain Function:
def ask_gpt(question):
    # 1. basic rule baed response
    question_lower = question.lower()

    if "hello" in question_lower or "hi" in question_lower:
        return "Hey there! How can I assist you today?"
    elif "bye" in question_lower or "hi" in question_lower:
        return "Goodbye! Have agood day ahead!"
    elif "your name" in question_lower:
        return "I'm a smart Chatbot created by Archita."
    elif "time" in question_lower:
        from datetime import datetime
        return f"the current time is {datetime.now().strftime('%I:%M %p')}"
    elif "day" in question_lower:
        from datetime import datetime
        return f"Today is {datetime.now().strftime('%A, %B %d, %Y')}"
    

    # 2. Fallback to GPT if no rule matched
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question}
            ],
            temperature=0.7,
            max_tokens=150,
            n=1,
            stop=None
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error: {str(e)}"

# 3. GUI layout
root = tk.Tk()
root.title("Smart Chatbot") 

# gui class:
class ChatbotGUI:
    def __init__(self,root):
        self.root = root
        self.root.title("Smart Chatbot")
        self.root.geometry("500x600")
        self.root.configure(bg = "#1e1e1e")

        # light and dark theme:
        self.themes = {
            "dark": {
                "bg" : "#121212",
                "fg" : "white",
                "entry_bg" : "#2e2e2e",
                "button_bg" : "#0078D7",
                "chat_bg" : "#1e1e1e",
                "insert_bg" : "white"
            },

            "light" : {
                "bg" : "#f5f5f5",
                "fg" : "black",
                "entry_bg" : "#ffffff",
                "button_bg" : "#0078D7",
                "chat_bg" : "#ffffff",
                "insert_bg" : "black"
            }
        }

        self.current_theme = "dark" 

# chat display area (Scrolled text box)
        self.chat_display = scrolledtext.ScrolledText(
            root,wrap=tk.WORD,
            bg="#1e1e1e",  # dark modern background
            fg="white",    # text color 
            font=("Segoe UI", 11), borderwidth=0, highlightthickness=0
        )
        self.chat_display.place(x=15, y=15, width=470, height=460)
        self.chat_display.config(
            font=("Segoe UI", 11),
            spacing3=6,   # extra space after each line
            borderwidth=0,
            relief="flat",
            highlightbackground="#444444",  # Border color
            highlightthickness=1
        )
        self.chat_display.config(state='disabled')

# seperater line :
        self.seperator = tk.Frame(root, bg="#444444", height=1)
        self.seperator.place(x=10, y=490, width=480)

# shadow frame for entry:
        self.entry_shadow = tk.Frame(root,bg="#111111")
        self.entry_shadow.place(x=11, y=501, width=402, height=42)


# chat text input fields:
        self.entry_field = tk.Entry(
            root,
            bg="#2e2e2e", fg="white", insertbackground="white",
            font=("Segoe UI", 12), relief=tk.FLAT, highlightthickness=1,
            highlightbackground="#444"
        )
        self.entry_field.place(x=15, y=490, width =370, height=45)
        self.entry_field.config(
            font=("Segoe UI", 12),
            relief="flat",
            highlightthickness=1,
            highlightbackground="#555"
        )
        self.entry_field.bind("<Return>", self.send_message) #press enter to send

# send button:
        self.send_button = tk.Button(
            root,
            text="Send ➤",
            command=self.send_message,
            bg="#0078D7",
            fg="white",
            font=("Segoe UI", 10, "bold"),
            activebackground="#005a9e",
            relief=tk.FLAT
        )
        self.send_button.place(x=395, y=490, width=90, height=45)
        self.send_button.config(font=("Segoe UI", 10, "bold"),
                                    relief="flat",
                                    bd=0
                                )

# Toggle theme button
        self.toggle_button = tk.Button(
            root, text="Toggle Theme", command=self.toggle_theme,
            bg="#444", fg="white", font=("Segoe UI", 9), relief=tk.FLAT
        )
        self.toggle_button.place(x=370, y=550, width=120, height=30)

        # Load avatars
        self.user_avatar_img = Image.open("user_avatar.png").resize((30,30))
        self.user_avatar =ImageTk.PhotoImage(self.user_avatar_img)

        self.bot_avatar_img = Image.open("bot_avatar.png").resize((30,30))
        self.bot_avatar =ImageTk.PhotoImage(self.bot_avatar_img)


    def send_message(self, event=None):
            user_message = self.entry_field.get().strip()
            if not user_message:
                return
        
            self.entry_field.delete(0, tk.END) #clear entry field
            self.display_message(f" 🧑 You:{user_message}\n")
            # user message with avatar:
            self.display_message(user_message,is_user=True)



        # Placeholder bot response (we'll add real logic later)
            threading.Thread(target=self.delayed_response, args=(user_message,)).start()

    
    def delayed_response(self,user_message):
            self.display_message("Bot is typing...\n")
            time.sleep(1.5)
            bot_reply = ask_gpt(user_message)

            # Step 4: Remove "Bot is typing..." and show actual reply
            self.chat_display.config(state='normal')

            # delete last line bot is typing:
            self.chat_display.delete("end-2l", "end-1l")

            self.display_message(f"Bot:{bot_reply}\n") #Step 5: Show reply
            self.chat_display.config(state="disabled")
            # bot reply  with avatar:
            self.display_message(bot_reply, is_user=False)


        # Add emoji to the reply (you can customize this)
            enhanced_reply = f"🤖 Bot: {bot_reply} 😊\n"
            self.display_message(enhanced_reply)

# creating function for smooth scrolling:
    def smooth_scroll_to_end(self):
            current = self.chat_display.yview()[0]
            # target is the bottom
            target = 1.0
            # step size:
            step = 0.02

            def do_scroll():
                nonlocal current
                if current < target :
                    current = min(current+step,target)
                    self.chat_display.yview_moveto(current)
                    self.root.after(10, do_scroll) #10 ms delay for smooth animation

            do_scroll()

    def display_message(self, message,is_user=True):
            timestamp = datetime.now().strftime("[%I:%M %p] ")
            self.chat_display.config(state = "normal")
            self.chat_display.insert(tk.END, message)
            self.chat_display.yview(tk.END) #auto scroll
            self.chat_display.config(state="disabled")

            avatar = self.user_avatar if is_user else self.bot_avatar
            name = "You" if is_user else "Bot"

            # insert avatar image:
            self.chat_display.image_create(tk.END, image=avatar)  # ✅

            self.chat_display.insert(tk.END, f" {name}: {message}\n\n")

            self.smooth_scroll_to_end()
            self.chat_display.config(state="disabled")


    def rule_based_bot(user_input):
        if "hello" in user_input.lower():
            return "Hi there!"
        elif "how are you" in user_input.lower():
            return "I'm just a bot,but I'm doing great."
        else:
            return "Sorry.I don't Understand."
        
# creating apply theme:
    def apply_theme (self):
        theme = self.themes[self.current_theme]

        self.root.configure(bg=theme["bg"])
        self.chat_display.configure(bg=theme["chat_bg"] , fg=theme["fg"])
        self.entry_field.configure(bg=theme["entry_bg"], fg=theme["fg"], insertbackground=theme["insert_bg"])
        self.send_button.configure(bg=theme["button_bg"], fg="white")
        self.toggle_button.configure(bg= "#444" if self.current_theme == "dark" else "#ccc", fg="white" if self.current_theme == "dark" else "black")


# creating toggle function:
    def toggle_theme(self):
        self.current_theme = "light" if self.current_theme == "dark" else "dark"
        self.apply_theme()

        self.apply_theme()


# Run the GUI app:
if __name__ == "__main__":
    root = tk.Tk()
    app = ChatbotGUI(root)
    root.mainloop()

