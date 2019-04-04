# /usr/bin/env python3
# -*- coding:utf-8 -*-

import re
s = r'''
"footer": {
    "advertiser": "廣告主",
    "publisher": "管道",
    "ecommerce": "E-Commerce Plus",
    "about": "關於",
    "news": "新聞",
    "contactUs": "聯繫我們",
    "copyRight": "© 2018 廣州鈦動科技有限公司. 版權所有. 粵ICP備18007337號-1"
  }'''
compiler1 = re.compile(":.+\"(.+)\"")
news = compiler1.findall(s)
for new in news:
    print(new)