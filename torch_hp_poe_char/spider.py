import requests
from lxml import html
from bs4 import BeautifulSoup
import re

main_url = "https://www.hplovecraft.com/writings/texts/"

response = requests.get(main_url)
tree = html.fromstring(response.content)

hrefs = tree.xpath(".//@href")
for href in hrefs:
    if "fiction" in href:
        url = main_url + href
        response = requests.get(url)
        tree = html.fromstring(response.content)
        text = tree.xpath(".//div[@align='justify']//text()")
        str = ""
        for line in text:
            str += line.replace("\r", "").replace("\n", "").replace(" ", "")
        with open("lovecraft.txt", "a", encoding='utf-8') as f:
            f.write(str + "\n")

