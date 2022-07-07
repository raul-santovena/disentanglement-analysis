# Disentanglement Analysis Visualization
A bokeh server example to visualize the disentanglement performance of a beta-VAE

![Tool example](https://github.com/raul-santovena/disentanglement-analysis/blob/main/example.gif)

## Instructions
### Set up the environment
First of all, it is advisable to create a virtual environment.

Using conda:

    conda create -n disentanglement-analysis python=3.7
    conda activate disentanglement-analysis
  
Now, you can install all the required libraries running the command below:

    conda install -c conda-forge numpy pandas bokeh

### Run the bokeh server:
To build and run the server you can run the following command from Anaconda Powershell, or use a code editor like VS Code (this command must be run from the root folder of the repository):

    bokeh serve --show main.py 
    
You can also run the server from the parent directory using the command below:

    bokeh serve --show disentanglement-analysis  
