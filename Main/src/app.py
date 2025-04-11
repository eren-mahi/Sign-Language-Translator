import tkinter as tk
from PIL import Image, ImageTk
import cv2
from hand_tracking import detect_hand_gesture

class GestureRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.cap = cv2.VideoCapture(0)
        self.last_gesture = ''
        self.sentence = ""

        self.setup_ui()
        self.update_frame()

    def setup_ui(self):
        # Video display
        self.video_label = tk.Label(self.root)
        self.video_label.pack()

        # Sentence display
        self.sentence_label = tk.Label(self.root, text="Sentence: ", font=("Helvetica", 16))
        self.sentence_label.pack(pady=10)

        # Control buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack()

        self.add_btn = tk.Button(button_frame, text="Add Letter", command=self.add_letter, width=12)
        self.add_btn.grid(row=0, column=0, padx=5)

        self.space_btn = tk.Button(button_frame, text="Space", command=self.add_space, width=12)
        self.space_btn.grid(row=0, column=1, padx=5)

        self.del_btn = tk.Button(button_frame, text="Delete", command=self.delete_last, width=12)
        self.del_btn.grid(row=0, column=2, padx=5)

        self.clear_btn = tk.Button(button_frame, text="Clear", command=self.clear_sentence, width=12)
        self.clear_btn.grid(row=0, column=3, padx=5)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame, gesture = detect_hand_gesture(frame)
            self.last_gesture = gesture if gesture else self.last_gesture

            # Convert frame to display
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.root.after(1, self.update_frame)

    def add_letter(self):
        if self.last_gesture:
            self.sentence += self.last_gesture
            self.update_sentence_display()

    def add_space(self):
        self.sentence += ' '
        self.update_sentence_display()

    def delete_last(self):
        self.sentence = self.sentence[:-1]
        self.update_sentence_display()

    def clear_sentence(self):
        self.sentence = ""
        self.update_sentence_display()

    def update_sentence_display(self):
        self.sentence_label.config(text=f"Sentence: {self.sentence}")

    def on_close(self):
        self.cap.release()
        self.root.destroy()

def main():
    root = tk.Tk()
    root.title("Sign Language Translator")
    app = GestureRecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()

if __name__ == "__main__":
    main()
