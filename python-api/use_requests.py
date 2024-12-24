import requests

response = requests.get("https://www.google.com")
print(response.text)


# Create a dictionary of parameters for get()
query_params = {"artist": "Deep Purple", "include_track": True}

# Pass the dictionary to the get() function
response = requests.get("https://www.google.com", params=query_params)
print(response.text)


# Create a dictionary of parameters for post(), data argument
playlist_data = {"Name": "Rock Ballads"}

# Pass the dictionary to the post() function
response = requests.post("https://www.google.com", data=playlist_data)
# or use json argument so the data is sent as JSON
response = requests.post("https://www.google.com", json=playlist_data)
print(response.text)

# Use `requests.codes` lookup object
response = requests.get("https://www.google.com")
if response.status_code == requests.codes.ok:
    print("Success!")
elif response.status_code == requests.codes.not_found:
    print("Not found!")

# Find out content-type of the response
response = requests.get("https://www.google.com")
print(response.headers["content-type"])

# Find out what content-type the server can respond with
print(response.headers["accept"])

# Add an `accept` header to the request so server returns JSON formated data
headers = {"accept": "application/json"}
response = requests.get("https://www.google.com", headers=headers)
print(response.text)


# Handle content-type errors

headers = {"accept": "application/xml"}
response = requests.get("https://www.google.com", headers=headers)

if response.status_code == 406:
    print("The server cannot respond in the requested format")
    print("It can send data in the following formats:" + response.headers["accept"])
else:
    print(response.text)

# Authentication status codes
response = requests.get("https://www.google.com")
if response.status_code == 200:
    print("Success!")
elif response.status_code == 401:
    print("Authentication failed!")
else:
    print("Another error has occurred")

# Basic authentication
auth = ("user", "pass")
response = requests.get("https://www.google.com", auth=auth)
if response.status_code == 200:
    print("Success!")
elif response.status_code == 401:
    print("Authentication failed!")
else:
    print("Another error has occurred")

# API key authentication - use params argument
params = {"access_token": "my_token"}
response = requests.get("https://www.google.com", params=params)
if response.status_code == 200:
    print("Success!")
elif response.status_code == 401:
    print("Authentication failed!")
else:
    print("Another error has occurred")

# API key authentication - use headers argument
headers = {"Authorization": "Bearer my_token"}
response = requests.get("https://www.google.com", headers=headers)
if response.status_code == 200:
    print("Success!")
elif response.status_code == 401:
    print("Authentication failed!")
else:
    print("Another error has occurred")
