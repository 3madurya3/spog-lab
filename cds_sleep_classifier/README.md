To get custom length or 60 second epochs fed into the CDS classifier, run the files in this order:

custom_epoch.py - takes in .its files and outputs csv files in the format the child directed speech classifier needs to receive. does 60 second epochs, but you can edit the code and change the "60"s to how many seconds you want the epochs to be.

pipeline.py - takes in the csv files generated previously and inputs them into online classifier using Selenium Webdriver. outputs csv files

name_files.py - renames the csv files generated from pipeline.py to the correct name (Child ID in the name)

Make sure to change the directory names to your local directories you want to keep them in.
You will also have to install pandas and Selenium Webdriver.

With the custom_epoch.py script, we generate the epochs from the .its file by looking through its entirety and splitting and joining segments of different lengths (the epochs were NOT generated originally by LENA).

-------------
To get 5 minute epochs, run these scripts in order:

mine.py - mines for 5 minute epochs and generates csv files
pipeline.py
name_files.py

Here, we already have information about 5 minute epochs at the bottom of each LENA-generated .its file, so we just parse that information directly. 

Happy classifying!
