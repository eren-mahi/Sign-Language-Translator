import tkinter as tk
import cv2
from PIL import Image, ImageTk
from hand_tracking import HandTracking

# Initialize the HandTracking object
hand_tracking = HandTracking()

def update_frame():
    # Get the full frame and gesture
    frame, gesture = hand_tracking.process_frame()

    if frame is not None:
        # Convert full frame to RGB and update video canvas
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        photo = ImageTk.PhotoImage(image=image)
        video_canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        video_canvas.image = photo

        # Update gesture text
        text_var.set(f"Gesture: {gesture}" if gesture else "Gesture: Unknown")

    # Get landmark-only frame (black background, 200x200)
    landmark_frame = hand_tracking.get_landmark_only_frame()
    if landmark_frame is not None:
        landmark_rgb = cv2.cvtColor(landmark_frame, cv2.COLOR_BGR2RGB)
        landmark_image = Image.fromarray(landmark_rgb)
        landmark_photo = ImageTk.PhotoImage(image=landmark_image)
        landmark_canvas.create_image(0, 0, image=landmark_photo, anchor=tk.NW)
        landmark_canvas.image = landmark_photo

    window.after(10, update_frame)

# Create the Tkinter window
window = tk.Tk()
window.title("Hand Gesture Recognition")

# Video feed canvas
video_canvas = tk.Canvas(window, width=640, height=480)
video_canvas.pack()

# Gesture label
text_var = tk.StringVar()
label = tk.Label(window, textvariable=text_var, font=("Helvetica", 16))
label.pack()

# Landmark-only canvas
landmark_canvas = tk.Canvas(window, width=200, height=200, bg='black')
landmark_canvas.pack(pady=10)

# Graceful shutdown
def on_closing():
    hand_tracking.release_resources()
    window.destroy()

window.protocol("WM_DELETE_WINDOW", on_closing)

# Start updating
update_frame()
window.mainloop()
