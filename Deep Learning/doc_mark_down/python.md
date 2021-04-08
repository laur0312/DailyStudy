#### 30 个 Python 的最佳实践、小贴士和技巧

##### 检查对象使用内存的状况

```python
import sys

mylist = range(0, 10000)
print(sys.getsizeof(mylist))
# 48
```

​		等等，为什么这个巨大的列表仅包含48个字节？

​		因为这里的 range 函数返回了一个类，只不过它的行为就像一个列表。在使用内存方面，range 远比实际的数字列表更加高效。

​		你可以试试看使用列表推导式创建一个范围相同的数字列表：

```python
import sys

myreallist = [x for x in range(0, 10000)]
print(sys.getsizeof(myreallist))
# 87632
```

##### 使用数据类

​		Python从版本3.7开始提供数据类。与常规类或其他方法（比如返回多个值或字典）相比，数据类有几个明显的优势：

- 数据类的代码量较少
- 你可以比较数据类，因为数据类提供了 \__eq__ 方法
- 调试的时候，你可以轻松地输出数据类，因为数据类还提供了 \__repr__ 方法
- 数据类需要类型提示，因此可以减少Bug的发生几率 

```python
from dataclasses import dataclass

@dataclass
class Card:
    rank: str
    suit: str

card = Card("Q", "hearts")

print(card == card)
# True

print(card.rank)
# 'Q'

print(card)
Card(rank='Q', suit='hearts')
```

##### 合并字典（Python 3.5以上的版本）

```python
dict1 = { 'a': 1, 'b': 2 }
dict2 = { 'b': 3, 'c': 4 }
merged = { **dict1, **dict2 }
print (merged)
# {'a': 1, 'b': 3, 'c': 4}
```

​		如果 key 重复，那么第一个字典中的 key 会被覆盖。

字符串的首字母大写

```python

mystring = "10 awesome python tricks"
print(mystring.title())
'10 Awesome Python Tricks'
```

##### map()

​		Python 有一个自带的函数叫做 map()，语法如下：

```python
map(function, something_iterable)
```

​		所以，你需要指定一个函数来执行，或者一些东西来执行。任何可迭代对象都可以。在如下示例中，我指定了一个列表：

```python
def upper(s):
    return s.upper()

mylist = list(map(upper, ['sentence', 'fragment']))
print(mylist)
# ['SENTENCE', 'FRAGMENT']

# Convert a string representation of
# a number into a list of ints.
list_of_ints = list(map(int, "1234567")))
print(list_of_ints)
# [1, 2, 3, 4, 5, 6, 7]
```

##### 查找出现频率最高的值

​		你可以通过如下方法查找出现频率最高的值：

```python
test = [1, 2, 3, 4, 2, 2, 3, 1, 4, 4, 4]
print(max(set(test), key = test.count))
# 4
```

​		max() 会返回列表的最大值。参数 key 会接受一个参数函数来自定义排序，在本例中为 test.count。该函数会应用于迭代对象的每一项。

​		test.count 是 list 的内置函数。它接受一个参数，而且还会计算该参数的出现次数。因此，test.count(1) 将返回2，而 test.count(4) 将返回4。

​		set(test) 将返回 test 中所有的唯一值，也就是 {1, 2, 3, 4}。

​		因此，这一行代码完成的操作是：首先获取 test 所有的唯一值，即{1, 2, 3, 4}；然后，max 会针对每一个值执行 list.count，并返回最大值。

#####  快速创建Web服务器

​		你可以快速启动一个Web服务，并提供当前目录的内容：

```python
python3 -m http.server
```


​		当你想与同事共享某个文件，或测试某个简单的HTML网站时，就可以考虑这个方法。

##### 统计元素的出现次数

​		你可以使用集合库中的 Counter 来获取列表中所有唯一元素的出现次数，Counter 会返回一个字典：

```python
from collections import Counter

mylist = [1, 1, 2, 3, 4, 5, 5, 5, 6, 6]
c = Counter(mylist)
print(c)
# Counter({1: 2, 2: 1, 3: 1, 4: 1, 5: 3, 6: 2})

# And it works on strings too:
print(Counter("aaaaabbbbbccccc"))
# Counter({'a': 5, 'b': 5, 'c': 5})
```

##### 日期的处理

​		python-dateutil 模块作为标准日期模块的补充，提供了非常强大的扩展，你可以通过如下命令安装： 

```
pip3 install python-dateutil 
```


​		你可以利用该库完成很多神奇的操作。在此我只举一个例子：模糊分析日志文件中的日期：

```python
from dateutil.parser import parse

logline = 'INFO 2020-01-01T00:00:01 Happy new year, human.'
timestamp = parse(log_line, fuzzy=True)
print(timestamp)
# 2020-01-01 00:00:01
```

​		你只需记住：当遇到常规 Python 日期时间功能无法解决的问题时，就可以考虑 python-dateutil ！

##### 通过chardet 来检测字符集

​		你可以使用 chardet 模块来检测文件的字符集。在分析大量随机文本时，这个模块十分实用。安装方法如下：

```
pip install chardet
```


​		安装完成后，你就可以使用命令行工具 chardetect 了，使用方法如下：

```python
chardetect somefile.txt
# somefile.txt: ascii with confidence 1.0
```

