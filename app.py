import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import threading

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
        tk.Label(control_frame, text="Learning Rate for SGD (0.001 recommended)").pack()
        self.learning_rate_entry_sgd = tk.Entry(control_frame, width=25)
        self.learning_rate_entry_sgd.pack(padx=5, pady=5)

        # BGD Learning rate
        tk.Label(control_frame, text="Learning Rate for BGD (0.01 recommended)").pack()
        self.learning_rate_entry_bgd = tk.Entry(control_frame, width=25)
        self.learning_rate_entry_bgd.pack(padx=5, pady=5)

        # Generate Data Button
        plot_button = tk.Button(control_frame, text="Generate data", command=lambda: self.generate_data(size=self.size_entry.get(), epochs=self.epochs_entry.get()))
        plot_button.pack(padx=5, pady=5)

        # Train Models
        plot_button = tk.Button(control_frame, text="Train Models", command=lambda: self.train_model())
        plot_button.pack(padx=5, pady=5)


        # Create a 2x2 grid of subplots
        self.figure, self.axes = plt.subplots(2, 2, figsize=(14, 9))
        self.figure.tight_layout(w_pad=2, h_pad=2)
        
        self.figure.subplots_adjust(top=0.95)

        # Assign subplots individually
        self.ax_sgd_rep = self.axes[0, 0]
        self.ax_bgd_rep = self.axes[0, 1]
        self.ax_sgd_loss = self.axes[1, 0]
        self.ax_bgd_loss = self.axes[1, 1]

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

        # Embed figure into Tkinter
        self.canvas = FigureCanvasTkAgg(self.figure, self.root)
        self.canvas.get_tk_widget().pack(pady=5)

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

        self.epochs_label = tk.Label(control_frame, text="")
        self.epochs_label.pack(pady=5)

        self.loss_bgd_label = tk.Label(control_frame, text="")
        self.loss_bgd_label.pack(pady=5)

        self.loss_sgd_label = tk.Label(control_frame, text="")
        self.loss_sgd_label.pack(pady=5)

    def generate_data(self, size, epochs):
        try:
            size = int(size)
            epochs = int(epochs)
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numeric values.")
            return
        if size < 2:
            messagebox.showerror("Invalid Input", "Please enter an integer greater than or equal to two.")
            return
        elif epochs < 1:
            messagebox.showerror("Invalid Input", "Please enter an integer greater than or equal to one.")
            return
        self.x = np.random.randn(size)

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
        self.canvas.draw()


    def train_sgd(self):
        self.dl_dw__sgd = 0
        self.dl_db__sgd = 0
        # Iterate though the x data and the y values at the same time
        for xi, yi in zip(self.x, self.y):
            # Compute the gradients
            self.dl_dw__sgd = -2*xi*(yi-(self.w_sgd* xi + self.b_sgd))
            self.dl_db__sgd = -2*(yi-(self.w_sgd * xi + self.b_sgd))

            self.w_sgd -= self.lr_sgd*self.dl_dw__sgd
            self.b_sgd -= self.lr_sgd*self.dl_db__sgd

        self.loss_history_sgd.append(np.mean( (self.y-(self.w_sgd * self.x + self.b_sgd))**2 ))

    def train_bgd(self):
        self.dl_dw__bgd = 0
        self.dl_db__bgd = 0
        # Iterate though the x data and the y values at the same time
        for xi, yi in zip(self.x, self.y):
            # Compute the gradients
            self.dl_dw__bgd += -2*xi*(yi-(self.w_bgd* xi + self.b_bgd))
            self.dl_db__bgd += -2*(yi-(self.w_bgd * xi + self.b_bgd))

        self.w_bgd -= self.lr_bgd*(self.dl_dw__bgd/len(self.x))
        self.b_bgd -= self.lr_bgd*(self.dl_db__bgd/len(self.x))

        self.loss_history_bgd.append(np.mean((self.y-(self.w_bgd * self.x + self.b_bgd))**2))
        

    def update_ui(self):
        # Clear all the plots
        self.ax_sgd_rep.clear()
        self.ax_sgd_loss.clear()
        self.ax_bgd_rep.clear()
        self.ax_bgd_loss.clear()
        
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

        # Represent the data in their respective plots
        self.ax_sgd_rep.scatter(self.x, self.y)
        self.ax_bgd_rep.scatter(self.x, self.y)
        
        # Generating x and y values for the line functions (the model)
        x_vals_sgd = np.linspace(min(self.x), max(self.x), 100)
        y_vals_sgd = self.w_sgd * x_vals_sgd + self.b_sgd

        x_vals_bgd = np.linspace(min(self.x), max(self.x), 100)
        y_vals_bgd = self.w_bgd * x_vals_bgd + self.b_bgd


        self.ax_sgd_rep.plot(x_vals_sgd, y_vals_sgd, color='red')
        self.ax_bgd_rep.plot(x_vals_bgd, y_vals_bgd, color='red')

        if self.epoch > 0:
            self.ax_sgd_loss.plot(range(1, len(self.loss_history_sgd)+1), self.loss_history_sgd, color='blue')
            self.ax_bgd_loss.plot(range(1, len(self.loss_history_bgd)+1), self.loss_history_bgd, color='blue')

        self.canvas.draw()
        # self.loss_rmse_label.config(text=f"The loss (RMSE) is: {0}")
        # self.loss_mse_label.config(text=f"The loss (MSE) is: {0}")
        # self.loss_mae_label.config(text=f"The loss (MAE) is: {0}")
        self.epochs_label.config(text=f"The current epochs is: {self.epoch+1}")
        self.loss_sgd_label.config(text=f"The loss for SGD is: {self.loss_history_sgd[-1]:.5}")
        self.loss_bgd_label.config(text=f"The loss for BGD is: {self.loss_history_bgd[-1]:.5}")
        # self.weights_label.config(text=f"The real slope was(with some noise): {self.slope} The predicted is: {0}")
        # self.biases_labels.config(text=f"The real bias was(with some noise): {self.intercept} The predicted is: {0}")
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

            for self.epoch in range(self.epochs):
                
                self.train_sgd()
                self.train_bgd()
                self.update_ui()
                


if __name__ == "__main__":
    root = tk.Tk()
    app = PlotApp(root)
    root.mainloop()
