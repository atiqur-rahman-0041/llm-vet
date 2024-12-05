# Dataset Preprocess

## How to Run the Code

This README provides instructions to set up the environment and run the dataset preprocess script. Follow the steps carefully to ensure a smooth setup.

---

1. **Python Installation**  
   Ensure Python 3.x is installed on your system. You can download it from [Python's official website](https://www.python.org/downloads/).

2. **Install `pip`**  
   If not already installed, install `pip` by running:  
   ```bash
   sudo apt install python3-pip  # For Ubuntu/Debian

3. **Create and Activate a Virtual Environment**  
   Use the following command to create a virtual environment for the project:
   ```bash
   python3 -m venv env
   sourve env/bin/activate

4. To preprocess a dataset, run the script with the input JSON file as an argument:
   ```bash
   python3 create_dataset.py <input_json_file>


## Output Files
### This will create four files:
1. <input_json_file_name>_converted.json : Added and Removed context are together for each hunk
2. <input_json_file_name>_converted_v2.json : Augmented samples by reverse transition of <input_json_file_name>_converted.json
3. <input_json_file_name>_reversed_converted.json : Added and Removed context are seperated for each hunk
4. <input_json_file_name>_reversed_converted_v2.json : Augmented samples by reverse transition of <input_json_file_name>_reversed_converted.json

