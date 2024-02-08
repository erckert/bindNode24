import requests
import shutil
import urllib.request, json
import time
import os
import re
from tqdm import tqdm
from pathlib import Path
import urllib
import urllib
import re
import os
import json
from config import FileManager, FileSetter
from selenium import webdriver
from selenium.webdriver.common.by import By
import selenium

"""Utility script to download Alphafold-DB files. XPATH ids might change anytime and break code."""

def extract_pdb_link(data):
    matches = re.finditer(re.compile('href=".*pdb'), data)
    results = []
    for match in matches:
        results.append(data[match.start() + 6 : match.end()])
    return results[0]


def download_pdb(url, header):
    pdbdata = requests.get(url)
    file = os.path.join(download_dir, f"{header}.pdb")
    with open(file, "w") as output:
        output.write(pdbdata.text)


# set chromodriver.exe path
fasta = FileManager.read_fasta(FileSetter.fasta_file())
driver = webdriver.Firefox(executable_path="/home/frank/Downloads/geckodriver")
download_dir = "/home/frank/bind/bindPredict/PDB/"
# implicit wait
driver.implicitly_wait(0.5)
failed = []
for header in fasta.keys():
    try:
        driver.get(f"https://alphafold.ebi.ac.uk/entry/{header}")
        # cookie accept
        pdbp = driver.find_element(
            By.XPATH,
            "/html/body/div[1]/app-root/section/app-entry/div[1]/div/app-summary-text/div/div[1]/div[2]/a[1]",
        )
        download_pdb(extract_pdb_link(pdbp.get_attribute("outerHTML")), header)
    except:
        failed.append(header)


# close browser
print("Download failed for:")

driver.quit()
