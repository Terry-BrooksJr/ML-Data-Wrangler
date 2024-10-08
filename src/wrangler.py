#! ~/.venv/bin/python
from classes import DataWrangler
import tkinter as tk
import tkinter.ttk as ttk
import sys 
from utility import DelayedKeyboardInterrupt
from tkinter import filedialog as fd

window = tk.Tk()
instruction_frame = tk.Frame(master=window, relief=tk.RIDGE)
data_entry_frame = tk.Frame(master=window)
button_frame = tk.Frame(master=window)

DATA = DataWrangler()

def select_ticket_file() -> None:
    filetypes = (
        ('JSON files', '*.json'),
    )

    file = fd.askopenfile(
        title='Path to Ticket Payload File',
        initialdir='.',
        filetypes=filetypes)
    DATA.ticket_file = file

def select_comments_dir():
    directory = fd.askdirectory()
    DATA.comments_dir=directory
    
instructions = tk.Label(text="Welcome to the Data Wrangler. \n \n You will need to identify two file locations. \n The Tickets File Path represents the path to the ticket payload from the ZenDesk Tickets API.\n The Second is the path to the comments directory, which is the location of the directory that contains the comments for each ticket.", foreground="black", master=instruction_frame)
warning = tk.Label(text="NOTE: The Ticket and the individual comment files must be in JSON Format.", foreground="Red", master=instruction_frame)
process_button = ttk.Button(text="Process Data")
comments_dir_location = ttk.Button( width=50, master=data_entry_frame)
ticket_file_location_label = tk.Button(text="Select Path to Ticket File", master=data_entry_frame, command=select_ticket_file)
comments_dir_location_label = tk.Button(text="Select Path to Comments Directory", master=data_entry_frame, 
command=select_comments_dir)

instruction_frame.pack(fill=tk.BOTH, expand=True)
data_entry_frame.pack(fill=tk.BOTH, expand=True)
button_frame.pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    print("Initializing Wrangler...")
    window.mainloop()

