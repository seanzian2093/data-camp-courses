from urllib import urlopen

url = "http://www.google.com"
with urlopen(url) as response:
    # read the data
    data = response.read()
    # get the encoding
    encoding = response.headers.get_content_charset()
    # decode the data
    string = data.decode(encoding)
    print(string)
