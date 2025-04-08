import tkinter as tk
import tkinter.font as font
from in_out import in_out
from motion import noise
from rect_noise import rect_noise
from record import record
from PIL import Image, ImageTk
import PIL
from find_motion import find_motion
from identify import maincall
from tamper import main

window = tk.Tk()
window.title("AI Surveillance System")
window.iconphoto(False, tk.PhotoImage(file='icons/mn.png'))
window.geometry('1080x700')


frame1 = tk.Frame(window)

label_title = tk.Label(frame1, text="AI powered Smart Surveillance System\nwith Real-time Anomaly Detection")
label_font = font.Font(size=24, weight='bold',family='Helvetica')
label_title['font'] = label_font
# label_title.grid(pady=(10,10), column=2)
label_title.grid(row=0, column = 0, columnspan=3, pady=10) 


icon = Image.open('icons/spy.png')
icon = icon.resize((120,120), PIL.Image.LANCZOS)
icon = ImageTk.PhotoImage(icon)
label_icon = tk.Label(frame1, image=icon)
# label_icon.grid(row=1, pady=(5,10), column=2)
label_icon.grid(row=1, column = 1, pady=5)

def load_icon(path):
    img = Image.open(path).resize((40, 40), PIL.Image.LANCZOS)
    return ImageTk.PhotoImage(img)


btn_images = {
    "monitor": load_icon('icons/lamp.png'),
    "rectangle": load_icon('icons/tamper_detect.png'),
    "noise": load_icon('icons/security-camera.png'),
    "record": load_icon('icons/recording.png'),
    "in_out": load_icon('icons/exit.png'),
    "identify": load_icon('icons/incognito.png'),
}

# Common button style
btn_font = font.Font(size=14, weight="bold")

# Buttons Grid
buttons = [
    ("Monitor", find_motion, btn_images["monitor"]),
    ("Tamper", main, btn_images["rectangle"]),
    ("Identify", maincall, btn_images["identify"]),
    ("Noise", noise, btn_images["noise"]),
    ("Record", record, btn_images["record"]),
    ("In Out", in_out, btn_images["in_out"]),
]

for i, (text, command, image) in enumerate(buttons):
    btn = tk.Button(frame1, text=text, image=image, compound="left", height=70, width=170, fg="black", command=command)
    btn['font'] = btn_font
    btn.grid(row=2 + i // 3, column=i % 3, padx=10, pady=10)


frame1.pack()
window.mainloop()


