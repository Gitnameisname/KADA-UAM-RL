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
    