class LinearRegression:
    def __init__(self, w, b):
        self.w = w
        self.b = b

    def fit(self, x, y, learning_rate, epochs):
        
        for epoch in range(epochs):
            # Initialize the gradient at 0 in each new epoch
            dl_dw = 0
            dl_db = 0

            # Iterate though the x data and the y values at the same time
            for xi, yi in zip(x, y):
                # Compute the gradients
                dl_dw += -2*xi*(yi-(self.w*xi +self.b))
                dl_db += -2*(yi-(self.w*xi+self.b))

                # Sum the inverse of the gradient
                self.w = self.w - learning_rate*dl_dw
                self.b = self.b - learning_rate*dl_db

            loss = 1/len(x) * sum(y-(self.w*x+self.b))**2 # Calculate the loss (MSE)
        return loss, self.w, self.b