import os
import requests
import json


from matbench import __version__



def release_gh():
    payload = {
        "tag_name": "v" + __version__,
        "target_commitish": "main",
        "name": "v" + __version__,
        "body": "",
        "draft": False,
        "prerelease": False
    }
    response = requests.post(
        "https://api.github.com/repos/hackingmaterials/matbench/releases",
        data=json.dumps(payload),
        headers={
            "Authorization": "token " + os.environ["GITHUB_RELEASES_TOKEN"]})
    print(response.text)


if __name__ == "__main__":
    release_gh()
