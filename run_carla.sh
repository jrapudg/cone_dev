#!/bin/bash

# Path to the bash script you want to run
script_path="../carla/CARLA_0.9.14/CarlaUE4.sh"

# Make sure the script is executable
chmod +x $script_path

# Infinite loop to keep executing the script
while true; do
    # Run the script
    $script_path

    # Check the exit status of the script
    if [ $? -ne 0 ]; then
        echo "Script crashed with exit code $?. Restarting..."
        sleep 1  # Optional: sleep for 1 second before restarting
    else
        echo "Script exited successfully."
        break  # Exit the loop if the script didn't crash
    fi
done
