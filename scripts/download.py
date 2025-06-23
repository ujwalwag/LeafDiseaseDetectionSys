import os
import requests
import json
import time

def download_github_folder_robust(repo_owner, repo_name, branch, folder_path, save_path="PlantVillage-Dataset-Downloaded"):

    base_api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/"
    base_raw_url = f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/{branch}/"

    full_api_path = os.path.join(base_api_url, folder_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Created main download directory: {save_path}")

    headers = {}


    def download_contents(api_url, current_save_dir):
        print(f"Fetching contents from: {api_url}")
        try:
            response = requests.get(api_url, headers=headers)
            response.raise_for_status() 
            contents = response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching API contents from {api_url}: {e}")
            print(f"Response content: {response.text}")
            return 

        for item in contents:
            item_name = item['name']
            item_type = item['type'] 
            download_url = item.get('download_url') 

            if item_type == 'dir':
                new_folder_path_in_repo = os.path.join(folder_path, item_name)
                new_save_dir = os.path.join(current_save_dir, item_name)
                if not os.path.exists(new_save_dir):
                    os.makedirs(new_save_dir)
                    print(f"Created subdirectory: {new_save_dir}")
                
          
                download_contents(item['url'], new_save_dir)
            elif item_type == 'file':
                if item_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    file_save_path = os.path.join(current_save_dir, item_name)

                    if os.path.exists(file_save_path):
   
                        print(f"Skipping existing file: {file_save_path}")
                        continue


                    effective_download_url = download_url if download_url else \
                                             os.path.join(base_raw_url, folder_path, item_name).replace('\\', '/')

                    print(f"Downloading: {item_name} to {file_save_path}")
                    try:
                        file_response = requests.get(effective_download_url, stream=True)
                        file_response.raise_for_status()
                        with open(file_save_path, 'wb') as f:
                            for chunk in file_response.iter_content(chunk_size=8192):
                                f.write(chunk)
            
                    except requests.exceptions.RequestException as e:
                        print(f"Error downloading {effective_download_url}: {e}")
                    time.sleep(0.01)

    download_contents(full_api_path, save_path)
    print("Download process completed.")

if __name__ == "__main__":

    repo_owner = "spMohanty"
    repo_name = "PlantVillage-Dataset"
    branch = "master" 
    folder_in_repo = "raw/color" 

    print("Starting download of PlantVillage-Dataset. This may take a while...")
    download_github_folder_robust(repo_owner, repo_name, branch, folder_in_repo)
    print("All files should now be downloaded!")