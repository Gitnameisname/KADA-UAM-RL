import json
import os
import matplotlib.pyplot as plt

def saveFlightData(agent):
    with open('./src/results/flightdata.json', 'w') as f:
        json.dump(agent.flightData, f, indent=4)

def plotFlightDataSeparate(agent):
    # Iterate over the flightData items
    for key, value in agent.flightData.items():
        # Skip the "time" key
        if key == "time":
            continue

        # Replace '/' with 'per' in the key
        key = key.replace('/', 'per')

        # Create a new figure for each plot
        fig, ax = plt.subplots(figsize=(10, 5))

        # Plot the data
        ax.plot(agent.flightData["time"], value)

        # Set the title of the plot to the current key
        ax.set_title(key)

        # Add a grid to the plot
        ax.grid(True)

        # Set x label and y label
        ax.set_xlabel('Time(sec)')
        ax.set_ylabel('Value')

        # Save the plot to the directory
        save_directory = './src/plots/'
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        plt.savefig(save_directory + f'{key}_plot.png')

        # Display the plot
        plt.show()

def plotFlightDataCombined(json_file1, json_file2, title_1='data_1', title_2='data_2'):
    # Load the JSON files
    with open(json_file1, 'r') as file:
        flightData1 = json.load(file)

    with open(json_file2, 'r') as file:
        flightData2 = json.load(file)

    # Calculate the number of subplots
    num_subplots = len(flightData1.keys()) - 1

    # Adjust the height of each plot
    fig_height = num_subplots * 5

    # Create a figure with subplots
    fig, axs = plt.subplots(num_subplots, 1, figsize=(10, fig_height))

    # Iterate over the flightData items
    for i, (key, value) in enumerate(flightData1.items()):
        # Skip the "time" key
        if key == "time":
            continue

        # Replace '/' with 'per' in the key
        plotTitle = key.replace('/', 'per')

        # Plot the data on the corresponding subplot
        axs[i-1].plot(flightData1["time"], value, label=title_1)
        axs[i-1].plot(flightData2["time"], flightData2[key], label=title_2)

        # Set the title of the subplot to the current key
        axs[i-1].set_title(plotTitle)

        # Add grid to the subplots
        axs[i-1].grid(True)

        # Add legend to the subplots
        axs[i-1].legend()

    # Add x-axis label to the last subplot
    axs[5].set_xlabel('Time(sec)')

    # Save the plots to the directory
    save_directory = './src/plots/'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    plt.savefig(save_directory + 'flightdata_combined.png')

    # Display the plots
    plt.show()

def plotActionDataCombined(json_file1, json_file2, title_1='data_1', title_2='data_2'):
    # Load the JSON files
    with open(json_file1, 'r') as file:
        actionData1 = json.load(file)

    with open(json_file2, 'r') as file:
        actionData2 = json.load(file)

    # Convert the action data to lists
    action_0_1 = list(map(float, actionData1["action_0"]))
    action_1_1 = list(map(float, actionData1["action_1"]))
    action_2_1 = list(map(float, actionData1["action_2"]))
    action_3_1 = list(map(float, actionData1["action_3"]))

    action_0_2 = list(map(float, actionData2["action_0"]))
    action_1_2 = list(map(float, actionData2["action_1"]))
    action_2_2 = list(map(float, actionData2["action_2"]))
    action_3_2 = list(map(float, actionData2["action_3"]))

    # Create the time values
    time_1 = [i * 0.05 for i in range(len(action_0_1))]
    time_2 = [i * 0.05 for i in range(len(action_0_2))]

    # Create subplots for each action
    fig, axs = plt.subplots(4, 1, figsize=(8, 10))

    # Plot action 0
    axs[0].plot(time_1, action_0_1, label=title_1)
    axs[0].plot(time_2, action_0_2, label=title_2)
    axs[0].set_ylabel('Action 0')
    axs[0].grid(True)  # Add grid

    # Plot action 1
    axs[1].plot(time_1, action_1_1, label=title_1)
    axs[1].plot(time_2, action_1_2, label=title_2)
    axs[1].set_ylabel('Action 1')
    axs[1].grid(True)  # Add grid

    # Plot action 2
    axs[2].plot(time_1, action_2_1, label=title_1)
    axs[2].plot(time_2, action_2_2, label=title_2)
    axs[2].set_ylabel('Action 2')
    axs[2].grid(True)  # Add grid

    # Plot action 3
    axs[3].plot(time_1, action_3_1, label=title_1)
    axs[3].plot(time_2, action_3_2, label=title_2)
    axs[3].set_ylabel('Action 3')
    axs[3].grid(True)  # Add grid

    # Add legend to the plots
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    axs[3].legend()

    # Add x-axis label to the last subplot
    axs[3].set_xlabel('Time')

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Save the plots to the directory
    save_directory = './src/plots/'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    plt.savefig(save_directory + 'actiondata_combined.png')

    # Display the plots
    plt.show()


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
