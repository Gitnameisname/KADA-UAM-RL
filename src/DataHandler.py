import json
import os
import matplotlib.pyplot as plt

def saveFlightData(agent):
    with open('./src/results/flightdata.json', 'w') as f:
        json.dump(agent.flightData, f, indent=4)

def plotFlightData(agent):
    # Calculate the number of subplots
    num_subplots = len(agent.flightData) - 1

    # Adjust the height of each plot
    fig_height = num_subplots * 5

    # Create a figure with subplots
    fig, axs = plt.subplots(num_subplots, 1, figsize=(10, fig_height))

    # Iterate over the flightData items
    for i, (key, value) in enumerate(agent.flightData.items()):
        # Skip the "time" key
        if key == "time":
            continue

        # Plot the data on the corresponding subplot
        axs[i-1].plot(agent.flightData["time"], value)

        # Set the title of the subplot to the current key
        axs[i-1].set_title(key)

    # Add a grid to the subplots
    for ax in axs:
        ax.grid(True)

    # Save the plots to the directory
    save_directory = './src/plots/'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    plt.savefig(save_directory + 'flightdata_plot.png')

    # Display the plots
    plt.show()
    
def saveActionData(agent):
    # Define the actionData dictionary
    actionData = {
        "action_0": list(map(str, agent.actionData["action_0"])),
        "action_1": list(map(str, agent.actionData["action_1"])),
        "action_2": list(map(str, agent.actionData["action_2"])),
        "action_3": list(map(str, agent.actionData["action_3"]))
    }

    # Save the actionData to a JSON file
    with open('./src/results/actionData.json', 'w') as file:
        json.dump(actionData, file)

def plotActionData(agent):
    # Convert the action data to lists
    action_0 = agent.actionData["action_0"]
    action_1 = agent.actionData["action_1"]
    action_2 = agent.actionData["action_2"]
    action_3 = agent.actionData["action_3"]

    # Create the time values
    time = [i * 0.05 for i in range(len(action_0))]

    # Create subplots for each action
    fig, axs = plt.subplots(4, 1, figsize=(8, 10))

    # Plot action 0
    axs[0].plot(time, action_0)
    axs[0].set_ylabel('Action 0')
    axs[0].grid(True)  # Add grid

    # Plot action 1
    axs[1].plot(time, action_1)
    axs[1].set_ylabel('Action 1')
    axs[1].grid(True)  # Add grid

    # Plot action 2
    axs[2].plot(time, action_2)
    axs[2].set_ylabel('Action 2')
    axs[2].grid(True)  # Add grid

    # Plot action 3
    axs[3].plot(time, action_3)
    axs[3].set_ylabel('Action 3')
    axs[3].grid(True)  # Add grid

    # Add x-axis label to the last subplot
    axs[3].set_xlabel('Time')

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Save the plots to the directory
    save_directory = './src/plots/'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    plt.savefig(save_directory + 'actiondata_plot.png')

    # Display the plots
    plt.show()
