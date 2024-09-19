# Local_LLM_RAG

Different RAG (Retrieval-Augmented Generation) systems.

## Table of Contents

- [Installation Steps](#installation-steps)
- [Run Steps](#run-steps)
- [Contributing](#contributing)
- [License](#license)

## Installation Steps

1. **Install Conda** (if not already installed):
   - **For macOS:**
     ```bash
     brew install conda
     ```
   - **For Linux:**
     ```bash
     wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
     sudo bash Miniconda3-latest-Linux-x86_64.sh -b -f -p /usr/local
     conda install -q -y --prefix /usr/local -c conda-forge conda
     conda init bash
     source ~/.bashrc
     ```
2. **Initiate Conda**:
   ```bash
   conda create -n Local_LLM_RAG python=3.10 -y
   conda activate Local_LLM_RAG
   ```
3. **Install the Requirements**:
   ```bash
   cd Local_LLM_RAG
   pip install -r requirements.txt
   ``` 
## Run Steps
To run the project, navigate to the Local_LLM_RAG directory and execute your Python script:
    
    cd Local_LLM_RAG
    python <File_name.py>

Replace <File_name.py> with the name of your Python script.

## Contributing

We welcome contributions to this project! Please feel free to submit a pull request or raise an issue for any suggestions or improvements.

Feel free to modify any part of this as needed! Let me know if you need anything else.


## License

This project is licensed under the Apache License 2.0. See the LICENSE file for details.
