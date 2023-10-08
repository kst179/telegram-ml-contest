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

# Replace with your GitHub username and personal access token
USERNAME = 'kst179'
TOKEN = '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'

# The language you want to search for
LANGUAGES = [
    "1C Enterprise",
    "ABAP",
    "ActionScript",
    "Ada",
    "Groovy",
    # "APEX",
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

# Function to search for repositories in a specific language
def search_github_code(session, language, n_pages=10, page_start=1):
    items = []

    for page in range(page_start, page_start + n_pages):
        
        timeout = 60
        
        while True:
            url = f'https://api.github.com/search/code?q=language:{language.replace(" ", "+")}&page={page}&per_page=100'
            response = session.get(url)
            print(f"GET: {response.status_code} {url}")
            if response.status_code == 200:
                items.extend(response.json()["items"])
                break
            elif response.status_code == 403:
                time.sleep(timeout)
                timeout = 5
            else:
                print(response.text)
                return
        
    return items

def download_file(session, item, output_dir):
    while True:
        url = item["url"]
        response = session.get(url)
        
        print(f"GET: {response.status_code} {url}")
        
        # timeout = 60
        
        if response.ok:
            item = response.json()
            
            if "content" not in item:
                return
            
            content = base64.b64decode(item["content"])
            with open(output_dir / f"{uuid.uuid4()}_{item['name']}", "wb") as file:
                file.write(content)
            break
        
        elif response.status_code == 403:
            time.sleep(5)
        else:
            break

# Function to download files from a GitHub repository
def download_files_from_repository(session, repo_url):
    response = session.get(repo_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        file_links = soup.find_all('a', class_='js-navigation-open')

        for link in file_links:
            file_name = link['title']
            file_url = 'https://github.com' + link['href']
            download_url = file_url.replace('/blob/', '/raw/')
            response = session.get(download_url)

            if response.status_code == 200:
                # Create the download directory if it doesn't exist
                os.makedirs(DOWNLOAD_DIR, exist_ok=True)

                # Save the file to the download directory
                with open(os.path.join(DOWNLOAD_DIR, file_name), 'wb') as file:
                    file.write(response.content)
                print(f"Downloaded {file_name}")
            else:
                print(f"Error downloading {file_name}")
    else:
        print(f"Error accessing repository: {repo_url}")

if __name__ == "__main__":
    DOWNLOAD_DIR.mkdir(exist_ok=True)
    
    session = authenticate_github()
    
    n_pages = 10
    
    # for page_start in range(6, 1000, n_pages):
        # np.random.shuffle(LANGUAGES)
        
    # all_code = []
        
    for language in LANGUAGES:
        code = search_github_code(session, language, n_pages=n_pages)
        
        if not code:
            continue
            
        for item in code:
            item["language"] = language
            
        # if not code:
        #     continue
            
        # output = DOWNLOAD_DIR / language
        # output.mkdir(exist_ok=True)
        
        # with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        #     futures = []
        #     for item in code:
        #         futures.append(executor.submit(download_file, session, item, output))
                
            # for future in concurrent.futures.as_completed(futures):
            #     _ = future.result()
        # all_code.extend(code)

        with open(DOWNLOAD_DIR / f"{language}.json", "w", encoding="utf8") as file:
            json.dump(code, file, ensure_ascii=False)

    print("Download completed.")