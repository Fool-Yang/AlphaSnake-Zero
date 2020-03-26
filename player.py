# -*- coding=gb2312 -*-
import os
import re
import time
class colors:
    c1='\033[0;37;41m'
    c2='\033[0;37;42m'
    c3='\033[0;37;43m'
    c4='\033[0;37;44m'
    c5='\033[0;37;45m'
    c6='\033[0;37;46m'
    c7='\033[0;37;47m'
    c8='\033[7;35;40m'
    c9='\033[0;37;48m'
    tail='\033[0m'

class Player:

    def show_page(self, page, color, total):
        time.sleep(0.2)
        os.system('cls')
        i=total[page]
        i=i.replace(' ','')
        i=i.replace('-1,',color.c1+'X '+color.tail)
        i=i.replace('-2,',color.c2+'X '+color.tail)
        i=i.replace('-3,',color.c3+'X '+color.tail)
        i=i.replace('-4,',color.c4+'X '+color.tail)
        i=i.replace('-5,',color.c5+'X '+color.tail)
        i=i.replace('-6,',color.c6+'X '+color.tail)
        i=i.replace('-7,',color.c7+'X '+color.tail)
        i=i.replace('-8,',color.c8+'X '+color.tail)
        i=i.replace('-9,',color.c9+'X '+color.tail)
        i=i.replace('-1]',color.c1+'X '+color.tail+']')
        i=i.replace('-2]',color.c2+'X '+color.tail+']')
        i=i.replace('-3]',color.c3+'X '+color.tail+']')
        i=i.replace('-4]',color.c4+'X '+color.tail+']')
        i=i.replace('-5]',color.c5+'X '+color.tail+']')
        i=i.replace('-6]',color.c6+'X '+color.tail+']')
        i=i.replace('-7]',color.c7+'X '+color.tail+']')
        i=i.replace('-8]',color.c8+'X '+color.tail+']')
        i=i.replace('1,',color.c1+'  '+color.tail)
        i=i.replace('2,',color.c2+'  '+color.tail)
        i=i.replace('3,',color.c3+'  '+color.tail)
        i=i.replace('4,',color.c4+'  '+color.tail)
        i=i.replace('5,',color.c5+'  '+color.tail)
        i=i.replace('6,',color.c6+'  '+color.tail)
        i=i.replace('7,',color.c7+'  '+color.tail)
        i=i.replace('8,',color.c8+'# '+color.tail)
        i=i.replace('9,',color.c9+'O '+color.tail)
        i=i.replace('0,','  ')
        i=i.replace('1]',color.c1+'  '+color.tail+']')
        i=i.replace('2]',color.c2+'  '+color.tail+']')
        i=i.replace('3]',color.c3+'  '+color.tail+']')
        i=i.replace('4]',color.c4+'  '+color.tail+']')
        i=i.replace('5]',color.c5+'  '+color.tail+']')
        i=i.replace('6]',color.c6+'  '+color.tail+']')
        i=i.replace('7]',color.c7+'  '+color.tail+']')
        i=i.replace('8]',color.c8+'# '+color.tail+']')
        i=i.replace('9]',color.c9+'O '+color.tail+']')
        i=i.replace('0]','  '+']')
        print(i+'\n')
        
    def main(self):
        file_name = "replay.txt"#input("Enter the replay file name:\n")
        f=open(file_name)
        total=f.read()
        total=total.split('\n\n')
        color=colors()
        page=0
        pasue=0
        while(1):
            try:
                self.show_page(page, color, total)
            except:
                break
            if(pasue==0):
                page+=1
            if(pasue==1):
                continue

if __name__ == '__main__':
    Player().main()