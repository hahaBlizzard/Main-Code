import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import Detection_Model


def browse_file(file_entry):
    # Function to browse and select a file
    window.update()  # Update the main window to fix focus issues
    file_path = filedialog.askopenfilename(parent=window, title="Select File", filetypes=(("All Files", "*.*"),))
    file_entry.delete(0, tk.END)
    file_entry.insert(0, file_path)

def upload_files():
    # Function to handle the "Process" button click event
    file_paths = [entry.get() for entry in file_entries]
    upload_button.config(state=tk.DISABLED) # Disable the "Process" button
    class_data, image_dimensions, image_original_size = Detection_Model.receive_file(file_paths)  # Get the class_data after processing
    display_class_selection(class_data, image_dimensions, image_original_size)  # Display the class_data for user selection

def display_class_selection(class_data,image_dimensions, image_original_size):
    # Function to display the class selection interface
    class_names = list(class_data.keys())
    class_selection_window = tk.Toplevel(window)
    class_selection_window.title("Select Class")
    class_selection_window.geometry("400x400")  # Set the size of the class selection window
    selected_classes = []

    def update_selected_classes():
        # Function to handle the "Submit" button click event
        process_selected_classes(selected_classes,class_data, image_dimensions, image_original_size)
        class_selection_window.destroy()
        inform_user()
        
    submit_button = tk.Button(class_selection_window, text="Submit", command=update_selected_classes)
    submit_button.pack(pady=10)

    def toggle_class(name):
        # Function to handle the check button toggle event
        if name in selected_classes:
            selected_classes.remove(name)
        else:
            selected_classes.append(name)

    for potential_class_name in class_names:
        check_button = tk.Checkbutton(class_selection_window, text=potential_class_name, command=lambda name=potential_class_name: toggle_class(name))
        check_button.pack(anchor=tk.W)

def process_selected_classes(selected_classes,class_data, image_dimensions, image_original_size):
    # Function to process the selected classes
    keys_list_set = set(selected_classes)
    keys_to_remove = []
    for key in class_data.keys():
        if key not in keys_list_set:
            keys_to_remove.append(key)
    for key in keys_to_remove:
        del class_data[key]
        
    Detection_Model.cropping(class_data, image_dimensions, image_original_size)  # Perform further processing in Detection_Model.py

def inform_user():
    # Function to inform the user about the completion of the model running
    messagebox.showinfo("Process Completed", "Model running is completed!")  # Display a message box to inform the user
    window.destroy()  # Close the Point Cloud Cropping interface

# Create the main window
window = tk.Tk()
window.geometry("750x400")
window.title("Point Cloud Cropping")

# Create file entry fields
file_entries = []
file_labels = ["Point Cloud", "Orthophoto", "TFW File", "XYZ File"]

for i in range(4):
    file_frame = tk.Frame(window)
    file_frame.pack()
    label = tk.Label(file_frame, text=f"{file_labels[i]}: ")
    label.pack(side=tk.LEFT)
    entry = tk.Entry(file_frame, width=50)
    entry.pack(side=tk.LEFT)
    browse_button = tk.Button(file_frame, text="Browse", command=lambda entry=entry: browse_file(entry))
    browse_button.pack(side=tk.LEFT)
    file_entries.append(entry)

# Create upload button
upload_button = tk.Button(window, text="Process", command=upload_files)
upload_button.pack()

# Run the GUI event loop
window.mainloop()