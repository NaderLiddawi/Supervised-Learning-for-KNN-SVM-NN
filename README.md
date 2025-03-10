---------------------------------------------------------
Assignment 1 – Code and Data Reproducibility Instructions
---------------------------------------------------------

Overview:
-----------
This repository contains the implementation for KNN, SVM & NN algorithms on two datasets. The primary script, `main.py`, orchestrates data preprocessing, model training (using KNN, SVM, and a Neural Network), hyperparameter optimization (via RandomizedSearchCV and Optuna), and evaluation. Two CSV data files—`marketing_campaign.csv` and `spotify-2023.csv`—are required inputs and must be placed in the same directory as `main.py`. Although the instructions below are geared toward ensuring reproducibility on a standard Linux machine, they are equally applicable on Windows with minor adjustments (e.g., virtual environment activation commands).

Directory Structure:
-----------------------
The repository is organized as follows:
   - **README.txt:** This file, which provides comprehensive instructions for running the code.
   - **main.py:** The main Python script containing the complete implementation.
   - **marketing_campaign.csv:** CSV file containing the marketing campaign dataset (expected to be tab-delimited and encoded in 'latin1').
   - **spotify-2023.csv:** CSV file containing the Spotify dataset.
   - **results/** (directory): Created at runtime to store output files such as plots and CSV summaries from hyperparameter searches.
   - **requirements.txt** (optional): A file listing all required Python packages 

Dependencies and Setup:
--------------------------
The implementation requires **Python 3.8** or higher, but preferably **Python 3.11**. The following libraries are used throughout the code:

   - **numpy:** For numerical computations and array operations.
   - **pandas:** For data manipulation and CSV file handling.
   - **matplotlib:** For plotting graphs and visualizations.
   - **scikit-learn:** For machine learning functionalities (e.g., model selection, metrics, preprocessing).
   - **imbalanced-learn (imblearn):** For handling imbalanced datasets (using techniques such as SMOTE and SMOTEENN).
   - **optuna:** For automated hyperparameter optimization.
   - **tabulate:** For formatting output tables in the console.
   - **scipy:** For statistical functions and distributions (e.g., `randint`, `loguniform`).

To install these dependencies, open a command prompt (Windows) or terminal (Linux) and run:

   pip install numpy pandas matplotlib scikit-learn imbalanced-learn optuna tabulate scipy

Since a `requirements.txt` file is included in the repository, you may also install all required packages with:

   pip install -r requirements.txt

For isolation and to avoid dependency conflicts, it is recommended to create a Python virtual environment. On Windows, you can set one up using:

   python -m venv env
   env\Scripts\activate

On Linux, the activation command would be:

   python3 -m venv env
   source env/bin/activate

Data Files:
------------
The execution of `main.py` depends on two CSV files that must be available in the same directory:

   - **marketing_campaign.csv:** Contains the marketing campaign dataset. This file is read using a tab delimiter (`\t`) and assumes a 'latin1' encoding.
   - **spotify-2023.csv:** Contains the Spotify dataset. The file is expected to have a format that supports numerical conversion of fields like `streams`.

Please ensure that both files are present in the repository’s root directory alongside `main.py`.

Running the Code:
---------------------
To execute the code and reproduce the results, follow these detailed steps:

   1. **Clone the Repository:**
      - Open a command prompt (on Windows) or terminal (on Linux).
      - Run:
            git clone https://github.com/NaderLiddawi/Supervised-Learning-for-KNN-SVM-NN.git

   2. **Set Up the Python Environment:**
      - Create a virtual environment:
            python -m venv env
      - Activate the virtual environment:
            env\Scripts\activate   (on Windows)
         or
            source env/bin/activate   (on Linux)

   3. **Install Dependencies:**
      - Install all required packages:
            pip install -r requirements.txt
         (or install packages individually as listed above).

   4. **Verify Data Files:**
      - Confirm that `marketing_campaign.csv` and `spotify-2023.csv` are located in the same directory as `main.py`.

   5. **Run the Script:**
      - Execute the main Python script:
            python main.py
      - The script will preprocess the datasets, perform model training and hyperparameter tuning, and output results (including evaluation metrics and generated plots) to both the console and the `results` directory.

Data Availability and Reproducibility:
-------------------------------------------
The code and datasets have been prepared to ensure full reproducibility. All preprocessing steps (including label encoding, scaling, and handling class imbalance), training routines, hyperparameter searches, and evaluation metrics are documented both within the code and in this README file. All modifications are tracked in the commit history.


Research Report:
-------------------------------------------
Please find the research report that analyzes the results produced by the code in the same GitHub directory as the code.


Additional Notes:
--------------------
   - The code includes inline comments and detailed documentation to guide evaluators through each computational step.
   - The output from hyperparameter searches, learning curves, and validation curves is saved in the `results` directory.
   - If any issues or discrepancies are encountered during execution, please review the commit history and inline documentation or contact the repository maintainer for further clarification.
   - This repository has been designed with reproducibility in mind (RANDOM_SEED=42); every effort has been made to ensure that following these instructions will yield the same results as those reported in the assignment.

-----------------
End of README.txt
-----------------
