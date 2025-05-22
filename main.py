import tkinter as tk
import tkinter.font as font
from in_out import in_out
from motion import noise
from rect_noise import rect_noise
from record import record
from find_motion import find_motion
from identify import maincall

window = tk.Tk()
window.title("AI Surveillance System")
window.geometry('1080x700')


frame1 = tk.Frame(window)

label_title = tk.Label(frame1, text="AI powered Smart Surveillance System\nwith Real-time Anomaly Detection")
label_font = font.Font(size=24, weight='bold',family='Helvetica')
label_title['font'] = label_font
label_title.grid(row=0, column=0, columnspan=3, pady=10) 


# Common button style
btn_font = font.Font(size=14, weight="bold")

# Buttons Grid
buttons = [
    ("Monitor", find_motion),
    ("Tamper", rect_noise),
    ("Identify", maincall),
    ("Noise", noise),
    ("Record", record),
    ("In Out", in_out),
]

for i, (text, command) in enumerate(buttons):
    btn = tk.Button(frame1, text=text, height=2, width=20, fg="black", command=command)
    btn['font'] = btn_font
    btn.grid(row=2 + i // 3, column=i % 3, padx=10, pady=10)


frame1.pack()
window.mainloop()


