import os
import requests
from bs4 import BeautifulSoup
import json
import time
import base64
from pathlib import Path
import uuid
import threading
import concurrent.futures
import numpy as np
import re

# Replace with your GitHub username and personal access token
USERNAME = 'kst179'
TOKEN = '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'

# The language you want to search for
LANGUAGES = [
    "1CEnterprise",
    "ABAP",
    "ActionScript",
    "Ada",
    "Groovy",
    "APEX",
    "AppleScript",
    "ASP",
    "Assembly",
    "AutoHotKey",
    "AWK",
    "Basic",
    "Batch",
    "Bison",
    "C",
    "Clojure",
    "CMake",
    "COBOL",
    "CoffeeScript",
    "Lisp",
    "C++",
    "Crystal",
    "C#",
    "CSS",
    "CSV",
    "D",
    "Dart",
    "Delphi",
    "Docker",
    "Elixir",
    "Elm",
    "Erlang",
    "FIFT",
    "Forth",
    "Fortran",
    "F#",
    "GAMS",
    "Go",
    "Gradle",
    "GraphQL",
    "Hack",
    "Haskell",
    "HTML",
    "Icon",
    "IDL",
    "INI",
    "Java",
    "JavaScript",
    "JSON",
    "Julia",
    "Keyman",
    "Kotlin",
    "LaTeX",
    "Lisp",
    "Logo",
    "Lua",
    "Makefile",
    "Markdown",
    "MATLAB",
    "NGINX",
    "Nim",
    "Objective-C",
    "OCaml",
    "OpenEdgeABL",
    "Pascal",
    "Perl",
    "PHP",
    "SQL",
    "PowerShell",
    "Prolog",
    "Protobuf",
    "Python",
    "QML",
    "R",
    "Raku",
    "Regex",
    "Ruby",
    "Rust",
    "SAS",
    "Scala",
    "Scheme",
    "Shell",
    "Smalltalk",
    "Solidity",
    "SQL",
    "Swift",
    "Tcl",
    "Textile",
    "TL",
    "TypeScript",
    "UnrealScript",
    "Vala",
    "VBScript",
    "Verilog",
    "VisualBasic",
    "Wolfram",
    "XML",
    "YAML",
]

# The directory where you want to save downloaded files
DOWNLOAD_DIR = Path('github_data')  


# Function to authenticate with GitHub using a personal access token
def authenticate_github():
    session = requests.Session()
    session.auth = (USERNAME, TOKEN)
    return session

def download_file(session, item, output_dir):
    url = item["url"]
    sha = item["sha"]
    name = item["name"]
    
    output_file = output_dir / f"{sha}_{name}"
    output_file = Path(re.sub(r"[\%\&\{\}\<\>\(\)*? $!'\":@|`=]", "_", output_file.as_posix()))
    # print(output_file)
    if output_file.exists():
        return
    
    num_retries = 10
    
    while True:
        response = session.get(url)
        
        print(f"GET: {response.status_code} {url}")
        
        if response.ok:
            item = response.json()
            
            if "content" not in item:
                print("content not found")
                return
            
            content = base64.b64decode(item["content"])
            
            with open(output_file, "wb") as file:
                file.write(content)
            break
        
        elif response.status_code == 403:
            if num_retries == 0:
                return
            
            time.sleep(5)
            num_retries -= 1
        else:
            return


if __name__ == "__main__":
    DOWNLOAD_DIR.mkdir(exist_ok=True)
    
    session = authenticate_github()

    for language in LANGUAGES:
        print(language)
        file = DOWNLOAD_DIR / f"{language}.json"
        output = DOWNLOAD_DIR / language
        
        if not file.exists():
            continue
        
        with open(file, encoding="utf8") as file:
            code = json.load(file)
            
        if not code:
            continue
        
        output.mkdir(exist_ok=True)
            
        for item in code:
            download_file(session, item, output)

    print("Download completed.")