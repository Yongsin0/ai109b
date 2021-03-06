# HW2(維吉尼亞密碼原理及Python實現)
## (簡介)
   維吉尼亞密碼是使用一系列凱撒密碼組成密碼字母表的加密算
   法，屬於多表密碼的一種簡單形式。
## 算法描述
* 加密過程:明文字母p對應的列和秘鑰字母k對應的行的交叉點就 是加密字母后的密文字母
* 解密過程:在密鑰字母k對應的行找到對應密文字母，則該密 文  字母對應的列的字母就是明文字母

* ![維吉尼亞密碼的表格]
* picture
 <img src="../img/abc.jpg" width="300" height="200"  align=center /> 
 
## [程式碼](https://github.com/Yongsin0/ai109b/blob/main/homework/week4/ddd.py):

```
import string

"""
确保大小写正确转换，用了两个列表
"""
letter_list = string.ascii_uppercase
letter_list2 = string.ascii_lowercase


def get_real_key():
    """
    获取列需要加的秘钥
    """
    print("输入你的秘钥")
    key = input()      #得确保都是英文
    tmp = []
    flag = 0
    for i in key:
        if i.isalpha():
            pass
        else:
            flag = 1
    if flag == 0:
        for i in key:
            tmp.append(ord(i.upper()) - 65)
        return tmp
    else:
        print("请输入英文秘钥")


def get_info():
    """
    获取信息
    """
    print("input your message: ")
    message = input()
    return  message


def Encrypt(message,key):
    ciphertext = ""
    flag = 0
    key_list = key
    for plain in message:
        if flag % len(key_list) == 0:
            flag = 0
        if plain.isalpha(): #判断是否为英文
            if plain.isupper():
                ciphertext += letter_list[(ord(plain) - 65 + key_list[flag]) % 26] #行偏移加上列偏移
                flag += 1
            if plain.islower():
                ciphertext += letter_list2[(ord(plain) - 97 + key_list[flag]) % 26]
                flag += 1
        else:#不是英文不加密
            ciphertext += plain

    return ciphertext


def Decrypt(message,key):
    plaintext = ""
    flag = 0
    key_list = key
    for cipher in message:
        if flag % len(key_list) == 0:
            flag  = 0
        if cipher.isalpha():
            if cipher.isupper():
                plaintext += letter_list[(ord(cipher) - 65 - key_list[flag]) %26]
                flag += 1
            if cipher.islower():
                plaintext += letter_list2[(ord(cipher) - 97 - key_list[flag]) % 26]
                flag += 1
        else:
            plaintext += cipher
    return plaintext


if __name__ == '__main__':
    while True:
        print("请选择加密或解密模式")
        print("1. Encrypt")
        print("2. Decrypt")
        choice = input("Your choice")
        if choice == '1':
            message = get_info()
            key = get_real_key()
            print(Encrypt(message, key))
        if choice == '2':
            message = get_info()
            key = get_real_key()
            print(Decrypt(message, key))
```
* 這里以輸入”helloworld”為例，然後進行加解密，能夠正確完
成
### 結果

* picture
 <img src="../img/01.jpg" width="300" height="200"  align=center /> 

* picture
 <img src="../img/02.jpg" width="300" height="200"  align=center /> 