Run the files in this order:

custom_epoch.py - takes in .its files and outputs csv files in the format the child directed speech classifier needs to receive. does 60 second epochs, but you can edit the code and change the "60"s to how many seconds you want the epochs to be.

pipeline.py - takes in the csv files generated previously and inputs them into online classifier using Selenium Webdriver. outputs csv files

name_files.py - renames the csv files generated from pipeline.py to the correct name (Child ID in the name)

Make sure to change the directory names to your local directories you want to keep them in.
You will also have to install pandas and Selenium Webdriver.
