**Project Structure**

1. excel_tables -> contains the cleaned data files in the .csv format
2. static folder -> contains the styles and the javascript files **(style.css and script.js)**
3. templates folder -> it contains the HTML part **(index1.html)**
4. app.py -> the backend function and all the renderings part
5. chrom.py -> this is the file which is used to convert the csv file into chrome embeds
6. count.py -> after running the chrom.pt to check how many files count is there 
7. requirements.txt -> contains the modules and libraries to be installed for the project workings

**Running Procedures**

1. First create an virtual environment (python -m venv venv)
2. activate the environment (venv/scripts/activate)
3. install the requirements.txt file for additional install the (re.txt file also)
4. run the chrom.py file and the chrom DB is created automatically
5. run the count.py file to check
6. run the app.py file and in browser enter **localhost** instead of **0.0.0.0**
