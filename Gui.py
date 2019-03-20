from tkinter import *
import blink_detect as bd
import blink_detect_svm as bd_svm 

def Begin1():
	bd.begin()

def Begin2():
	bd_svm.begin()

root = Tk()
root.title("Blink detect!")
root.geometry('200x100')  
Button(root, text = 'Blink detecting without svm', command = Begin1).pack()
Button(root, text = 'Blink detecting with svm', command = Begin2).pack()
Button(root, text = 'Quit', command = root.quit, width=10, height = 1).pack()
root.mainloop()
