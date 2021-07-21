import zipfile
import os
import pyreadr
import pandas


def unzip(file_loc, extract_loc=None):
    """
    Unzips data files and saves them in the processed directory
    """
    try:
        with zipfile.ZipFile(
            file_loc, "r"
        ) as file:  # opening the zip file using 'zipfile.ZipFile' class
            print("Ok")
            # ZipFile.infolist() returns a list containing all the members of an archive file
            print(file.infolist())

            # ZipFile.namelist() returns a list containing all the members with names of an archive file
            print(file.namelist())

            # ZipFile.getinfo(path = filepath) returns the information about a member of Zip file.
            # It raises a KeyError if it doesn't contain the mentioned file
            print(file.getinfo(file.namelist()[-1]))

            # If extraction directory not given, extracted to 'data/processed/file_name'
            if extract_loc == None:
                base = os.path.dirname(file_loc)
                folder_name = os.path.basename(base)
                extract_loc = "data/processed/" + folder_name

            # ZipFile.extractall(path = filepath, pwd = password) extracts all
            # the files to current directory
            file.extractall(path=extract_loc)
            # after executing check the directory to see extracted files

    except zipfile.BadZipFile:  # if the zip file has any errors then it prints the
        # error message which you wrote under the 'except' block
        print("Error: Zip file is corrupted")

    except zipfile.LargeZipFile:
        print("Error: File size if too large")  # if the file size is too large to
        # open it prints the error you have written
    except FileNotFoundError:
        print("Error: File not found")


def RData_to_csv(file_loc, extract_loc=None):
    """
    Converts RData file to csv format. If extract_loc left empty, extracts to data/processed
    directory.
    """
    try:
        # If extraction directory not given, extracted to 'data/processed/file_name'
        if extract_loc == None:
            base = os.path.dirname(file_loc)
            folder_name = os.path.basename(base)
            extract_loc = "data/processed/" + folder_name

        # If extraction directionary does not already exist, create directory
        if not os.path.exists(extract_loc):
            os.makedirs(extract_loc, exist_ok=True)

        # Open RData file
        data = pyreadr.read_r(file_loc)

        # For dataframes in RData file, save each dataframe as df_name.csv
        for key in data.keys():
            df = data[key]
            csv_loc = extract_loc + "/" + key + ".csv"
            df.to_csv(csv_loc)

    except FileNotFoundError:
        print("Error: Extraction directory not found")
    except pyreadr.custom_errors.PyreadrError:
        print("Error: Rdata file not found")
