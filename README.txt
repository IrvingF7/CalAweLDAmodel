This file will guide you through the process of preparing the necessary enviroment for the LDA model I provide. Just like the codes I 
provide, this instruction is not perfect, so I will keep updating when necessary.

This is the first time I am writing a instuction for remote users, so I am still rather inexperienced on this. Please email me if you
find anything confusing in this file.
My email:
irvingf7@berkeley.edu
===============================================================================================================================================

1. Have Python installed on the computer
Unlike most software, which you can update them from version 2 to version 3 easily, Python 2 and Python 3 are not the same software. 
For daily usage they are almost identical but they are listed independently by the official foundation and have some technical difference
that set them apart. They are compatible on the same computer, and also independent of each other. 
For this project both 2 and 3 will do the trick, but 3 is recommended.
(Among all the packages, there is one crucial package that seems like only working under Python 3, though it claims it is compatible with Python 2.
I still can't find a way to fix this, so I recommand using Python 3)
And I recommend only install Python 3, because if you install both Python 2 and 3, some commands we need to use may change.

Please refer to this link for Python installation:
https://www.python.org/downloads/
===============================================================================================================================================

2. Get a terminal that can run shell.
Windows 10 has a built-in tool called Windows Powershell, which can be the best choice of this task. 
Git Bash can also do the trick, please refer to this website for downloading: 
https://gitforwindows.org/

Windows also has a built-in software called cmd.exe, which has similar features but the commands for this one are of different style 
than commands on Linux, so I cannot say I am familir with them. I do not recommend using this one as I may not be able to provide solid support on this.
========================================================================================================================================================

3. Upgrade pip
pip is a Python software we will use later on to install packages. We need to update it just in case.
(1) Open Windows Powershell(Or Git Bash)

(2) If you have only Python 3 installed, then enter this:
python3 -m pip install --upgrade pip
If this does not work, and the prompt showed is something like "python 3 not recognized", then use:
python -m pip install --upgrade pip

(3) If you have only Python 2 installed, then enter this:
python -m pip install --upgrade pip

(4) If you have both Python 2 and Python 3 installed, please do the following:
¢ÙIf you want to use Python 2 through out this project, then use:
py -2 -m pip install --upgrade pip
¢ÚIf you want to use Python 3 through out this project, then use:
py -3 -m pip install --upgrade pip

(5) Wait for a little while.
========================================================================================================================================================


4. Downloading Jupyter Notebook
Jupyter Notebook is not required to run the codes, but as an interactive software, it can significantly boost the experiencing of using
the codes and viewing the data. I highly recommand using this.

(1) Open Windows Powershell(Or Git Bash)

(2) If you have only Python 3 installed, then enter this:
python3 -m pip install jupyter
If this does not work, and the prompt showed is something like "python 3 not recognized", then use:
python -m pip install --upgrade pip

(3) If you have only Python 2 installed, then enter this:
python -m pip install jupyter

(4) If you have both Python 2 and Python 3 installed, please do the following:
¢ÙIf you want to use Python 2 through out this project, then use:
py -2 -m pip install jupyter
¢ÚIf you want to use Python 3 through out this project, then use:
py -3 -m pip install jupyter

(5) Wait for a little while.
========================================================================================================================================================


5. Downloading the required packages
To render a complete result, all the packages need to be installed.

(1) Open Windows Powershell(Or Git Bash)

(2) Navigate to the folder that contains LDAmodel.ipynb, this file, and a file named requirements.txt.
requirements.txt contains the name and version of all the packages we need so far. Please do not edit the file.
About how to navigate between folders, please refer to this site:
https://stackoverflow.com/questions/8961334/how-to-change-folder-with-git-bash

(3) If you have only Python 3 installed, then enter this:
pip3 install -r requirements.txt
If this does not work, and the prompt showed is something like "pip3 not recognized", then use:
pip install -r requirements.txt

(4) If you have only Python 2 installed, then enter this:
pip install -r requirements.txt

(5) If you have both Python 2 and Python 3 installed, please do the following:
¢ÙIf you want to use Python 2 through out this project, then use:
py -2 pip install -r requirements.txt
¢ÚIf you want to use Python 3 through out this project, then use:
py -3 pip install -r requirements.txt

(6) Wait for the pip to install all the packages.
========================================================================================================================================================


6. Downloading the stopword library.
This is to download a library of stopword for English. We are using the package named Spacy.
We need admin account to finish this step.

(1) Right click the icon of Windows Powershell, left click "run as administrator"

(2)Enter:
py -3 -m spacy download en
(Or just like above, substitue "py -3" to "python" or "python3" if a specific situation applies to you)
========================================================================================================================================================



7. Run the code.
It is all set and done.

(1) Navigate to the folder contain the LDAmodel.ipynb

(2) Enter:
jupyter notebook LDAmodel_new.ipynb

(3) A new page in the broswer will appear, and you can run the code.

(4) For running your own CSV/JSON file:
¢Ù: Go to https://www.csvjson.com/csv2json if it is a CSV. Convert it to a JSON.
¢Ú: Download or copy paste the JSON file to the corresponding folder in the input folder.

(5) The parameters that you can play with:
dataName: the name of the data file you can analyze. For now it is just the language of the data.
num_of_representatives: the number of representative narrative you want to show for each topic
num_of_topics: the number of the topics you want to generate from the data



