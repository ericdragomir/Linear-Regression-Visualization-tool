import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np


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

        tk.Label(control_frame, text="Number of Epochs").pack()
        self.epochs_entry = tk.Entry(control_frame, width=25)
        self.epochs_entry.pack(padx=5, pady=5)

        tk.Label(control_frame, text="Learning Rate (0.001 recommended for SGD and 0.1 or 0.01 for BGD)").pack()
        self.learning_rate_entry = tk.Entry(control_frame, width=25)
        self.learning_rate_entry.pack(padx=5, pady=5)

        # Generate Data Button
        plot_button = tk.Button(control_frame, text="Generate data", command=lambda: self.generate_data(size=self.size_entry.get()))
        plot_button.pack(padx=5, pady=5)

        # Train Model with SGD Button
        plot_button = tk.Button(control_frame, text="Train (Stochastic Gradient Descent)", command=lambda: self.train_model(algo="SGD"))
        plot_button.pack(padx=5, pady=5)

        # Train Model with BGD Button
        plot_button = tk.Button(control_frame, text="Train (Batch Gradient Descent)", command=lambda: self.train_model(algo="BGD"))
        plot_button.pack(padx=5, pady=5)

        tk.Label(control_frame, text="*Even though we provide different error measurement (losses), we use MSE for computing the gradient").pack()

        # Matplotlib Figure
        self.figure, self.ax = plt.subplots(figsize=(12, 8))
        self.canvas = FigureCanvasTkAgg(self.figure, self.root)
        self.canvas.get_tk_widget().pack()

        # Labels for displaying vales
        self.loss_mse_label = tk.Label(self.root, text="")
        self.loss_mse_label.pack(pady=5)

        self.loss_rmse_label = tk.Label(self.root, text="")
        self.loss_rmse_label.pack(pady=5)

        self.loss_mae_label = tk.Label(self.root, text="")
        self.loss_mae_label.pack(pady=5)

        self.weights_label = tk.Label(self.root, text="")
        self.weights_label.pack(pady=5)

        self.biases_labels = tk.Label(self.root, text="")
        self.biases_labels.pack(pady=5)

        self.epochs_label = tk.Label(self.root, text="")
        self.epochs_label.pack(pady=5)

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

        # Add some noise
        self.y = []
        self.slope  = np.random.uniform(-20, 20)
        self.intercept = np.random.uniform(-20, 20)
        for xi in self.x:
            self.y.append(self.slope * xi + self.intercept + np.random.randn(1)*2) # Add some Gaussian noise

        # Clear previous plot
        self.ax.clear()

        # Plot data
        self.ax.scatter(self.x, self.y)

        # Refresh canvas
        self.canvas.draw()


    def train_sgd(self):
        self.dl_dw = 0
        self.dl_db = 0
        # Iterate though the x data and the y values at the same time
        for xi, yi in zip(self.x, self.y):
            # Compute the gradients
            self.dl_dw += -2*xi*(yi-(self.w* xi + self.b))
            self.dl_db += -2*(yi-(self.w * xi + self.b))

            self.w -= self.lr*self.dl_dw
            self.b -= self.lr*self.dl_db

    def train_bgd(self):
        self.dl_dw = 0
        self.dl_db = 0
        # Iterate though the x data and the y values at the same time
        for xi, yi in zip(self.x, self.y):
            # Compute the gradients
            self.dl_dw += -2*xi*(yi-(self.w* xi + self.b))
            self.dl_db += -2*(yi-(self.w * xi + self.b))

        self.w -= self.lr*(self.dl_dw/len(self.x))
        self.b -= self.lr*(self.dl_db/len(self.x))
        
                    
    def choose_training_algo(self, algo):
        match algo:
            case "SGD":
                self.train_sgd()
            case "BGD":
                self.train_bgd()
            case _:
                messagebox.showerror("Internal Error", "Please restart the program.")
                quit()
                return

    def update_ui(self):
        self.ax.clear()
        self.ax.scatter(self.x, self.y)

        # Generating x and y values for the line function (the model)
        x_vals = np.linspace(min(self.x), max(self.x), 100)
        y_vals = self.w * x_vals + self.b

        self.ax.plot(x_vals, y_vals, color='red')
        self.canvas.draw()
        self.loss_rmse_label.config(text=f"The loss (RMSE) is: {self.rmse_loss}")
        self.loss_mse_label.config(text=f"The loss (MSE) is: {self.mse_loss}")
        self.loss_mae_label.config(text=f"The loss (MAE) is: {self.mae_loss}")
        self.epochs_label.config(text=f"The current epochs is: {self.epoch+1}")
        self.weights_label.config(text=f"The real slope was(with some noise): {self.slope} The predicted is: {self.w.item()}")
        self.biases_labels.config(text=f"The real bias was(with some noise): {self.intercept} The predicted is: {self.b.item()}")
        self.root.update_idletasks()
        self.root.update()


    def train_model(self, algo):
        # Get input data
        try:
            self.epochs = int(self.epochs_entry.get())
            self.lr = float(self.learning_rate_entry.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numeric values.")
            return
        
        if not hasattr(self, 'x') or not hasattr(self, 'y'):
            messagebox.showerror("Error", "Please generate data first.")
        else:
            self.w = 0
            self.b = 0
            for self.epoch in range(self.epochs):

                self.choose_training_algo(algo)
                self.rmse_loss = np.sqrt(np.mean((self.y-(self.w * self.x + self.b))**2)) # Calculate the loss (RMSE)
                self.mse_loss = np.mean((self.y-(self.w * self.x + self.b))**2) # Calculate the loss (MSE)
                self.mae_loss = np.mean(np.abs((self.y-(self.w * self.x + self.b)))) # Calculate the loss (MAE)
                self.update_ui()
                


if __name__ == "__main__":
    root = tk.Tk()
    app = PlotApp(root)
    root.mainloop()
