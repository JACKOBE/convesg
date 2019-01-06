import re
import requests
from bs4 import BeautifulSoup

url = 'http://www.dy2018.com/i/98477.html'

headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:38.0) Gecko/20100101 Firefox/38.0'}


def get_one_film_detail(url):
    #print("one_film doing:%s" % url)
    r = requests.get(url, headers=headers)
    # print(r.text.encode(r.encoding).decode('gbk'))
    bsObj = BeautifulSoup(r.content.decode('gbk','ignore'), "html.parser")
    td = bsObj.find('td', attrs={'style': 'WORD-WRAP: break-word'})
    if td is None:#没有找到下载标签的返回None，个别网页格式不同
        return None, None
    url_a = td.find('a')
    url_a = url_a.string
    title = bsObj.find('h1')
    title = title.string
    # title = re.findall(r'[^《》]+', title)[1] #此处处理一下的话就只返回影片名 本例结果为：猩球崛起3：终极之战
    return title, url_a

print (get_one_film_detail(url))

import re
import requests
from bs4 import BeautifulSoup

page_url = 'http://www.dy2018.com/2/index_22.html'

headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:38.0) Gecko/20100101 Firefox/38.0'}


def get_one_page_urls(page_url):
    #print("one_page doing：%s" % page_url)
    urls = []
    base_url = "http://www.dy2018.com"
    r = requests.get(page_url, headers=headers)
    bsObj = BeautifulSoup(r.content, "html.parser")
    url_all = bsObj.find_all('a', attrs={'class': "ulink", 'title': re.compile('.*')})
    for a_url in url_all:
        a_url = a_url.get('href')
        a_url = base_url + a_url
        urls.append(a_url)
    return urls

print (get_one_page_urls(page_url))
import eventlet
import re
import time
import requests
from bs4 import BeautifulSoup

result = []
headers={'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:38.0) Gecko/20100101 Firefox/38.0'}

def get_one_film_detail(urls):
    req = requests.session() #此处改用session，可以减少和服务器的 tcp链接次数。
    req.headers.update(headers)
    for url in urls:
        print("one_film doing:%s" % url)
        r = req.get(url)
        # print(r.text.encode(r.encoding).decode('gbk'))
        bsObj = BeautifulSoup(r.content.decode('gbk','ignore'), "html.parser")
        td = bsObj.find('td', attrs={'style': re.compile('.*')})
        if td is None:
            continue
        url_a = td.find('a')
        if url_a is None:
            continue
        url_a = url_a.string
        title = bsObj.find('h1')
        title = title.string
        # title = re.findall(r'[^《》]+', title)[1]
        f = open("download.txt", "a")
        f.write("%s:%s\n\n" % (title, url_a))



def get_one_page_urls(page_url):
    print("one_page doing：%s" % page_url)
    urls = []
    base_url = "http://www.dy2018.com"
    r = requests.get(page_url, headers=headers)
    bsObj = BeautifulSoup(r.content, "html.parser")
    url_all = bsObj.find_all('a', attrs={'class': "ulink", 'title': re.compile('.*')})
    for a_url in url_all:
        a_url = a_url.get('href')
        a_url = base_url + a_url
        urls.append(a_url)
    return urls
    # print(r.text.encode(r.encoding).decode('gbk'))


pool = eventlet.GreenPool()
start = time.time()
page_urls = ['http://www.dy2018.com/2/']
for i in range(2, 100):
    page_url = 'http://www.dy2018.com/2/index_%s.html' % i
    page_urls.append(page_url)

for urls in pool.imap(get_one_page_urls, page_urls):
        get_one_film_detail(urls)

end = time.time()
print('total time cost:')
print(end - start)
