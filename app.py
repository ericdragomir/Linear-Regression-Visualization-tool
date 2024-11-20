import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

from linear_regression_class import LinearRegression

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

        tk.Label(control_frame, text="Number of Epochs").pack()
        self.epochs_entry = tk.Entry(control_frame, width=25)
        self.epochs_entry.pack(padx=5, pady=5)

        tk.Label(control_frame, text="Learning Rate (0.001 recommended)").pack()
        self.learning_rate_entry = tk.Entry(control_frame, width=25)
        self.learning_rate_entry.pack(padx=5, pady=5)

        # Generate Data Button
        plot_button = tk.Button(control_frame, text="Generate data", command=self.generate_data)
        plot_button.pack(padx=5, pady=5)

        # Train Model Button
        plot_button = tk.Button(control_frame, text="Train", command=self.train_model)
        plot_button.pack(padx=5, pady=5)

        # Matplotlib Figure
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, self.root)
        self.canvas.get_tk_widget().pack()

        # Labels for displaying values
        self.loss_label = tk.Label(self.root, text="")
        self.loss_label.pack(pady=5)

        self.weights_label = tk.Label(self.root, text="")
        self.weights_label.pack(pady=5)

        self.biases_labels = tk.Label(self.root, text="")
        self.biases_labels.pack(pady=5)

    def generate_data(self):
        # if len(x) != len(y):
        #     messagebox.showerror("Input Error", "X and Y must have the same number of elements.")
        #     return
        
        self.x = np.random.randn(np.random.randint(15, 30), 1)
        # Add some noise
        self.y = []
        self.slope  = np.random.random()
        self.intercept = np.random.random()
        for xi in self.x:
            self.y.append(self.slope * xi + self.intercept + np.random.randn(1)*0.2) # Add some Gaussian noise

        # Clear previous plot
        self.ax.clear()

        # Plot data
        self.ax.scatter(self.x, self.y)

        # Refresh canvas
        self.canvas.draw()

    def train_model(self):
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
            model = LinearRegression(0, 0)
            loss, weight, bias = model.fit(self.x, self.y, self.lr, self.epochs)

            self.ax.clear()
            self.ax.scatter(self.x, self.y)

            # Generating x and y values for the line function (the model)
            x_vals = np.linspace(min(self.x), max(self.x), 100)
            y_vals = weight * x_vals + bias

            self.ax.plot(x_vals, y_vals, color='red')
            self.canvas.draw()

            self.loss_label.config(text=f"The loss is: {loss}")
            self.weights_label.config(text=f"The real slope was(with some noise): {self.slope} The predicted is: {weight}")
            self.biases_labels.config(text=f"The real bias was(with some noise): {self.intercept} The predicted is: {weight}")


if __name__ == "__main__":
    root = tk.Tk()
    app = PlotApp(root)
    root.mainloop()
