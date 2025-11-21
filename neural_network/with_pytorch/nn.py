import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

import matplotlib.pyplot as plt
import seaborn as sns

class BasicNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad=False)
        self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)

        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)

        self.final_bias = nn.Parameter(torch.tensor(-16.), requires_grad=False)
    
    def forward(self, input):
        input_to_top_relu = input * self.w00 + self.b00
        top_relu_output = F.relu(input_to_top_relu)
        scale_top_relu_output = top_relu_output * self.w01

        input_to_bottom_relu = input * self.w10 + self.b10
        bottom_relu_output = F.relu(input_to_bottom_relu)
        scale_bottom_relu_output = bottom_relu_output * self.w11

        input_to_final_relu = scale_top_relu_output + scale_bottom_relu_output + self.final_bias
        final_output = F.relu(input_to_final_relu)
        return final_output
    
input_doses = torch.linspace(0, 1, steps=11)
model = BasicNN()
output_responses = model(input_doses)

sns.set_style("darkgrid")
sns.lineplot(x=input_doses, y=output_responses, color='blue', marker='o')
plt.title("Dose-Response Curve using Basic Neural Network")
plt.xlabel("Dose")
plt.ylabel("Response")
plt.show()

class BasicNN_train(nn.Module):
    def __init__(self):
        super().__init__()
        self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad=False)
        self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)

        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)

        self.final_bias = nn.Parameter(torch.tensor(0.0), requires_grad=True)
    
    def forward(self, input):
        input_to_top_relu = input * self.w00 + self.b00
        top_relu_output = F.relu(input_to_top_relu)
        scale_top_relu_output = top_relu_output * self.w01

        input_to_bottom_relu = input * self.w10 + self.b10
        bottom_relu_output = F.relu(input_to_bottom_relu)
        scale_bottom_relu_output = bottom_relu_output * self.w11

        input_to_final_relu = scale_top_relu_output + scale_bottom_relu_output + self.final_bias
        final_output = F.relu(input_to_final_relu)
        return final_output
    
input_doses = torch.linspace(0, 1, steps=11)
model = BasicNN_train()
output_responses = model(input_doses)

sns.set_style("darkgrid")
sns.lineplot(x=input_doses, y=output_responses.detach(), color='blue', marker='o')
plt.title("Dose-Response Curve using Basic Neural Network")
plt.xlabel("Dose")
plt.ylabel("Response")
plt.show()

inputs = torch.tensor([0, 0.5, 1])
labels = torch.tensor([0, 1, 0])

optimizer = SGD(model.parameters(), lr=0.1)

for epoch in range(100):
    total_loss = 0
    for iteration in range(len(inputs)):
        input = inputs[iteration]
        label = labels[iteration]

        output = model(input)
        loss = (output - label)** 2
        loss.backward()
    
        total_loss += float(loss)
        
    
    if (total_loss < 0.0001):
        print(f"Training complete at epoch {epoch} with total loss {total_loss}")
        break  
    optimizer.step()
    optimizer.zero_grad()

print(f"Epoch {epoch}: Total Loss = {total_loss}: Final Bias = {model.final_bias.data}")

# Test data (unseen)
test_inputs = torch.tensor([0.25, 0.75])
test_labels = torch.tensor([0.5, 0.5])

# After training loop
with torch.no_grad():
    test_output = model(test_inputs)
    test_loss = ((test_output - test_labels) ** 2).mean()
    print(f"Test Loss: {test_loss}")

