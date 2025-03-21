import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import threading
import time

class PlotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Linear Regression Visualization")

        self.create_widgets()

    def create_widgets(self):
        # Labels and Entries for x and y
        # tk.Label(self.root, text="X values (comma-separated):").pack()
        # self.x_entry = tk.Entry(self.root, width=50)
        # self.x_entry.pack()

        # tk.Label(self.root, text="Y values (comma-separated):").pack()
        # self.y_entry = tk.Entry(self.root, width=50)
        # self.y_entry.pack()

        # Create a frame for controls
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.LEFT, fill=tk.X)

        tk.Label(control_frame, text="Number of Data points (at least 2)").pack()
        self.size_entry = tk.Entry(control_frame, width=25)
        self.size_entry.pack(padx=5, pady=5)

        tk.Label(control_frame, text="Number of Epochs (at least 1)").pack()
        self.epochs_entry = tk.Entry(control_frame, width=25)
        self.epochs_entry.pack(padx=5, pady=5)

        # SGD Learning rate
        tk.Label(control_frame, text="Learning Rate for SGD (0.1 or 0.01 recommended)").pack()
        self.learning_rate_entry_sgd = tk.Entry(control_frame, width=25)
        self.learning_rate_entry_sgd.pack(padx=5, pady=5)

        # BGD Learning rate
        tk.Label(control_frame, text="Learning Rate for BGD (0.1 recommended)").pack()
        self.learning_rate_entry_bgd = tk.Entry(control_frame, width=25)
        self.learning_rate_entry_bgd.pack(padx=5, pady=5)

        # Put convergence difference
        tk.Label(control_frame, text="Set a convergence difference (optional) (e.g. 0.1)").pack()
        self.convergence_diff_entry = tk.Entry(control_frame, width=25)
        self.convergence_diff_entry.pack(padx=5, pady=5)

        # Generate Data Button
        plot_button = tk.Button(control_frame, text="Generate data", command=lambda: self.generate_data(size=self.size_entry.get()))
        plot_button.pack(padx=5, pady=5)

        # Train Models
        plot_button = tk.Button(control_frame, text="Train Models", command=lambda: self.train_model())
        plot_button.pack(padx=5, pady=5)


        # Create the SGD plots and a new frame
        # Create a container frame for the canvases
        canvas_frame = tk.Frame(self.root)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.figure_sgd, self.axes_sgd = plt.subplots(2, 1, figsize=(8, 10))
        self.figure_sgd.tight_layout(w_pad=2, h_pad=2)
        
        self.figure_sgd.subplots_adjust(top=0.95)

        # Create the BGD plots
        self.figure_bgd, self.axes_bgd = plt.subplots(2, 1, figsize=(8, 10))
        self.figure_bgd.tight_layout(w_pad=2, h_pad=2)
        
        self.figure_bgd.subplots_adjust(top=0.95)


        # Assign subplots individually
        self.ax_sgd_rep = self.axes_sgd[0]
        self.ax_sgd_loss = self.axes_sgd[1]

        self.ax_bgd_rep = self.axes_bgd[0]
        self.ax_bgd_loss = self.axes_bgd[1]

        # Set titles
        self.ax_sgd_rep.set_title("SGD Representation")
        self.ax_bgd_rep.set_title("BGD Representation")
        self.ax_sgd_loss.set_title("SGD Loss")
        self.ax_bgd_loss.set_title("BGD Loss")

        # Plot placeholders (You can replace these with actual data)
        self.ax_sgd_rep.plot([], [])  
        self.ax_bgd_rep.plot([], [])  
        self.ax_sgd_loss.plot([], [])  
        self.ax_bgd_loss.plot([], [])  

        # Embed figures into Tkinter
        self.canvas_sgd = FigureCanvasTkAgg(self.figure_sgd, canvas_frame)
        self.canvas_sgd.get_tk_widget().pack(side=tk.LEFT, pady=5)

        self.canvas_bgd = FigureCanvasTkAgg(self.figure_bgd, canvas_frame)
        self.canvas_bgd.get_tk_widget().pack(side=tk.RIGHT, pady=5)

        # Labels for displaying vales
        self.loss_mse_label = tk.Label(control_frame, text="")
        self.loss_mse_label.pack(pady=5)

        # self.loss_rmse_label = tk.Label(control_frame, text="")
        # self.loss_rmse_label.pack(pady=5)

        # self.loss_mae_label = tk.Label(control_frame, text="")
        # self.loss_mae_label.pack(pady=5)

        self.weights_label = tk.Label(control_frame, text="")
        self.weights_label.pack(pady=5)

        self.biases_labels = tk.Label(control_frame, text="")
        self.biases_labels.pack(pady=5)


        # Current epochs labels
        self.epochs_sgd_label = tk.Label(control_frame, text="")
        self.epochs_sgd_label.pack(pady=5)

        self.epochs_bgd_label = tk.Label(control_frame, text="")
        self.epochs_bgd_label.pack(pady=5)

        # Loss labels
        self.loss_bgd_label = tk.Label(control_frame, text="")
        self.loss_bgd_label.pack(pady=5)

        self.loss_sgd_label = tk.Label(control_frame, text="")
        self.loss_sgd_label.pack(pady=5)

        # Time labels
        self.time_sgd_label = tk.Label(control_frame, text="")
        self.time_sgd_label.pack(pady=5)

        self.time_bgd_label = tk.Label(control_frame, text="")
        self.time_bgd_label.pack(pady=5)

        # Convergence labels
        self.convergence_sgd_label = tk.Label(control_frame, text="")
        self.convergence_sgd_label.pack(pady=5)

        self.convergence_bgd_label = tk.Label(control_frame, text="")
        self.convergence_bgd_label.pack(pady=5)

    def generate_data(self, size):
        try:
            size = int(size)
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numeric values.")
            return
        if size < 2:
            messagebox.showerror("Invalid Input", "Please enter an integer greater than or equal to two.")
            return
        self.x = np.random.randn(size)
        self.size = size # Store the value for later computations

        # Add some noise
        self.y = []
        self.slope  = np.random.uniform(-20, 20)
        self.intercept = np.random.uniform(-20, 20)
        
        self.y = self.slope * self.x + self.intercept + np.random.randn(size) * 2

        # Clear previous plot
        self.ax_sgd_rep.clear()
        self.ax_bgd_rep.clear()

        # Set titles
        self.ax_sgd_rep.set_title("SGD Representation")
        self.ax_bgd_rep.set_title("BGD Representation")
        self.ax_sgd_loss.set_title("SGD Loss")
        self.ax_bgd_loss.set_title("BGD Loss")

        # Label axis
        self.ax_sgd_loss.set_xlabel("Epochs")
        self.ax_bgd_loss.set_xlabel("Epochs")
        self.ax_sgd_loss.set_ylabel("Loss (MAE)")
        self.ax_bgd_loss.set_ylabel("Loss (MAE)")

        # Plot data
        self.ax_sgd_rep.scatter(self.x, self.y)
        self.ax_bgd_rep.scatter(self.x, self.y)

        # Refresh canvas
        self.canvas_sgd.draw()
        self.canvas_bgd.draw()

    def check_convergence(self, diff, arr):
        if (arr[-2] - arr[-1]) < diff:
            return True
        return False

    def update_sgd_ui(self):
        # Clear the plots
        self.ax_sgd_rep.clear()
        self.ax_sgd_loss.clear()
        
        # Set titles
        self.ax_sgd_rep.set_title("SGD Representation")
        self.ax_sgd_loss.set_title("SGD Loss")

        # Label axis
        self.ax_sgd_loss.set_xlabel("Epochs")
        self.ax_sgd_loss.set_ylabel("Loss (MAE)")

        # Represent the data
        self.ax_sgd_rep.scatter(self.x, self.y)
        
        # Generating x and y values for the line functions (the model)
        x_vals_sgd = np.linspace(min(self.x), max(self.x), 100)
        y_vals_sgd = self.w_sgd * x_vals_sgd + self.b_sgd

        self.ax_sgd_rep.plot(x_vals_sgd, y_vals_sgd, color='red')

        if self.epoch_sgd > 0:
            self.ax_sgd_loss.plot(range(1, len(self.loss_history_sgd)+1), self.loss_history_sgd, color='blue')
            self.loss_sgd_label.config(text=f"The loss for SGD is: {self.loss_history_sgd[-1]:.5}")
            if self.epoch_sgd > 1 and not self.convergence_sgd:
                if self.check_convergence(diff=self.convergence_diff, arr=self.loss_history_sgd):
                    self.convergence_sgd = True
                    self.convergence_sgd_label.config(text=f"SGD converged at epoch {self.epoch_sgd}")
        else:
            self.loss_sgd_label.config(text=f"The loss for SGD is: WAIT FOR ONE EPOCH")


        self.canvas_sgd.draw()
        self.epochs_sgd_label.config(text=f"The current SGD epochs is: {self.epoch_sgd+1}")

        self.root.update_idletasks()
        self.root.update()

    def update_bgd_ui(self):
        # Clear the plots
        self.ax_bgd_rep.clear()
        self.ax_bgd_loss.clear()
        
        # Set titles
        self.ax_bgd_rep.set_title("BGD Representation")
        self.ax_bgd_loss.set_title("BGD Loss")

        # Label axis
        self.ax_bgd_loss.set_xlabel("Epochs")
        self.ax_bgd_loss.set_ylabel("Loss (MAE)")

        # Represent the data
        self.ax_bgd_rep.scatter(self.x, self.y)
        
        # Generating x and y values for the line functions (the model)
        x_vals_bgd = np.linspace(min(self.x), max(self.x), 100)
        y_vals_bgd = self.w_bgd * x_vals_bgd + self.b_bgd

        self.ax_bgd_rep.plot(x_vals_bgd, y_vals_bgd, color='red')

        if self.epoch_bgd > 0:
            self.ax_bgd_loss.plot(range(1, len(self.loss_history_bgd)+1), self.loss_history_bgd, color='blue')
            self.loss_bgd_label.config(text=f"The loss for BGD is: {self.loss_history_bgd[-1]:.5}")
            if self.epoch_bgd > 1 and not self.convergence_bgd:
                if self.check_convergence(diff=self.convergence_diff, arr=self.loss_history_bgd):
                    self.convergence_bgd = True
                    self.convergence_bgd_label.config(text=f"BGD converged at epoch {self.epoch_bgd}")
        else:
            self.loss_bgd_label.config(text=f"The loss for BGD is: WAIT FOR ONE EPOCH")
            
        self.canvas_bgd.draw()
        self.epochs_bgd_label.config(text=f"The current BGD epochs is: {self.epoch_bgd+1}")
        self.root.update_idletasks()
        self.root.update()

    def train_model(self):
        # Get input data
        try:
            self.epochs = int(self.epochs_entry.get())
            self.lr_sgd = float(self.learning_rate_entry_sgd.get())
            self.lr_bgd = float(self.learning_rate_entry_bgd.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numeric values.")
            return
        if self.epochs < 1:
            messagebox.showerror("Invalid Input", "Please enter an integer greater than or equal to one.")
            return

        if self.convergence_diff_entry.get() == "":
            self.convergence_diff = 0.01
        else:
            try:
                self.convergence_diff = float(self.convergence_diff_entry.get())
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter a numeric convergence or none")
                return
            
            if not self.convergence_diff > 0:
                messagebox.showerror("Invalid Input", "Please enter a numeric convergence greater than 0")
                return

        if not hasattr(self, 'x') or not hasattr(self, 'y'):
            messagebox.showerror("Error", "Please generate data first.")
        else:
            # Set the model's parameters
            self.w_sgd = 0
            self.w_bgd = 0
            self.b_sgd = 0
            self.b_bgd = 0
            
            # Start the loss history
            self.loss_history_sgd = []
            self.loss_history_bgd = []

            # Clear the plots
            self.ax_sgd_rep.clear()
            self.ax_bgd_rep.clear()
            self.ax_sgd_loss.clear()
            self.ax_bgd_loss.clear()

            # Define the training threads
            def train_sgd():
                self.convergence_sgd_label.config(text="")
                self.time_sgd_label.config(text="")
                self.convergence_sgd = False
                init_time = time.time()
                for self.epoch_sgd in range(self.epochs):
                    for xi, yi in zip(self.x, self.y):
                        # Compute the gradients using MAE as the loss function
                        dl_dw = -2 * xi * (yi - (self.w_sgd * xi + self.b_sgd))
                        dl_db = -2 * (yi - (self.w_sgd * xi + self.b_sgd))

                        # Compute the new weights
                        self.w_sgd -= self.lr_sgd * dl_dw
                        self.b_sgd -= self.lr_sgd * dl_db
                    self.loss_history_sgd.append(np.mean((self.y - (self.w_bgd * self.x + self.b_sgd )) ** 2)) # Compute and append the loss (MAE)
                    self.update_sgd_ui()
                self.time_sgd_label.config(text=f"SGD: {(time.time() - init_time):.5}s")
                if not self.convergence_sgd:
                    self.convergence_sgd_label.config(text=f"SGD did not converge")

            def train_bgd():
                self.convergence_bgd_label.config(text="")
                self.time_bgd_label.config(text="")
                self.convergence_bgd = False
                init_time = time.time()
                for self.epoch_bgd in range(self.epochs):
                    dl_dw = 0
                    dl_db = 0
                    for xi, yi in zip(self.x, self.y):
                        dl_dw += -2 * xi * (yi - (self.w_bgd * xi + self.b_bgd))
                        dl_db += -2 * (yi - (self.w_bgd * xi + self.b_bgd))
                    
                    # Adjust the weight with the mean after seeing the whole dataset
                    self.w_bgd -= self.lr_bgd * dl_dw/self.size
                    self.b_bgd -= self.lr_bgd * dl_db/self.size    
                    self.loss_history_bgd.append(np.mean((self.y - (self.w_bgd * self.x + self.b_bgd )) ** 2))
                    self.update_bgd_ui()
                self.time_bgd_label.config(text=f"BGD: {(time.time() - init_time):.5}s")
                if not self.convergence_bgd:
                    self.convergence_bgd_label.config(text=f"BGD did not converge")


            # Define and execute the training threads
            sgd_thread = threading.Thread(target=train_sgd)
            bgd_thread = threading.Thread(target=train_bgd)

            sgd_thread.start()
            bgd_thread.start()


if __name__ == "__main__":
    root = tk.Tk()
    app = PlotApp(root)
    root.mainloop()
