import tkinter as tk
from PIL import Image, ImageTk

root = tk.Tk()
root.title("Emotion GUI")


def process_image():
    return


def next_image(image_number):
    global image_view
    global button_next

    image_view.grid_forget()
    image_view = tk.Label(image=image_list[image_number-1])
    button_next = tk.Button(frame_buttons, text="Next image", height=1, width=20, command=lambda: next_image(image_number+1))

    if image_number == 7:
        button_next = tk.Button(frame_buttons, text="Next image", height=1, width=20, state=tk.DISABLED)

    image_view.grid(row=3, column=1)
    button_next.grid(row=1, column=0)


def statistics():
    return


# Load images
img1 = ImageTk.PhotoImage(Image.open("ValidationImages/angry1.jpg"))
img2 = ImageTk.PhotoImage(Image.open("ValidationImages/disgust1.jpg"))
img3 = ImageTk.PhotoImage(Image.open("ValidationImages/fear1.jpg"))
img4 = ImageTk.PhotoImage(Image.open("ValidationImages/happy1.jpg"))
img5 = ImageTk.PhotoImage(Image.open("ValidationImages/neutral1.jpg"))
img6 = ImageTk.PhotoImage(Image.open("ValidationImages/sad1.jpg"))
img7 = ImageTk.PhotoImage(Image.open("ValidationImages/surprise1.jpg"))

image_list = [img1, img2, img3, img4, img5, img6, img7]

label_title = tk.Label(root, text="Emotion recognition and imitation")
# This labels will become frames with gt, result, and imitation
label_gt = tk.Label(root, text="Ground truth")
label_recognized = tk.Label(root, text="Recognized")
label_imitated = tk.Label(root, text="Imitated")

label_title.grid(row=0, column=0, columnspan=6)
label_gt.grid(row=1, column=1)
label_recognized.grid(row=1, column=3)
label_imitated.grid(row=1, column=5)

# images and icon boxes
# Frame ground truth
image_current = ImageTk.PhotoImage(Image.open("ValidationImages/happy1.jpg"))
image_view = tk.Label(image=image_current)
image_view.grid(row=3, column=1)

# Frame gt end

# Frame recognized

# Frame rec end

# Frame imitated

# Frame imi end

# Frame buttons
frame_buttons = tk.LabelFrame(root, text="Buttons", padx=5, pady=5)
frame_buttons.grid(row=5, column=1)

button_process = tk.Button(frame_buttons, text="Process image", height=1, width=20)
button_next = tk.Button(frame_buttons, text="Next image", command=lambda: next_image(2), height=1, width=20)
button_statistics = tk.Button(frame_buttons, text="Statistics", height=1, width=20)

button_process.grid(row=0, column=0)
button_next.grid(row=1, column=0)
button_statistics.grid(row=2, column=0)
# Frame but end

button_quit = tk.Button(root, text="Quit", command=root.destroy)
button_quit.grid(row=11, column=5)

root.mainloop()
