import tkinter as tk
import json

output_stream = []

class Marquee(tk.Canvas):
    def __init__(self, parent, text, margin=2, borderwidth=1, relief='flat', fps=100):
        tk.Canvas.__init__(self, parent, borderwidth=borderwidth, relief=relief)
        self.fps = fps

        # start by drawing the text off screen, then asking the canvas
        # how much space we need. Use that to compute the initial size
        # of the canvas. 
        output_stream = self.input_stream()
        #output_stream = text
        output_stream = self.create_text(0, -1000, text=output_stream, anchor="w", tags=("text",))
        (x0, y0, x1, y1) = self.bbox("text")
        width = (x1 - x0) + (2*margin) + (2*borderwidth)
        height = (y1 - y0) + (2*margin) + (2*borderwidth)
        self.configure(width=width, height=height)

        # start the animation
        self.animate()

    def animate(self):
        (x0, y0, x1, y1) = self.bbox("text")
        if x1 < 0 or y0 < 0:
            # everything is off the screen; reset the X
            # to be just past the right margin
            x0 = self.winfo_width()
            y0 = int(self.winfo_height()/2)
            self.coords("text", x0, y0)
        else:
            self.move("text", -1, 0)

        # do again in a few milliseconds
        self.after_id = self.after(int(1000/self.fps), self.animate)

    def input_stream(self):
        with open('marquee_content.json' , 'r') as reader:
            jf = json.loads(reader.read())
        
        for index in jf:
            output_stream.append(index['text_content'])
        return output_stream

        
root = tk.Tk()
marquee = Marquee(root, text="Hello, world", borderwidth=20, relief="sunken")
marquee.pack(side="top", fill="x", pady=20)
marquee.input_stream()

root.mainloop()