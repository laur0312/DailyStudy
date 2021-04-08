# 单机存储系统

注解 | 含义 |
--- | --- |
@RestController | 告诉Spring以字符串的形式渲染结果，并直接返回给调用者 |
@RequestMapping | 提供路由信息
@EnableAutoConfiguration | 告诉Spring Boot根据添加的jar依赖猜测你想如何配置Spring
@SpringBootApplication | 等价于以默认属性使用@Configuration，@EnableAutoConfiguration和@ComponentScan


# 作为一个打包后的应用运行
运行一个打包的程序并开启远程调试支持是可能的，这允许你将调试器附加到打包的应用程序上：

```java
$ java -Xdebug -Xrunjdwp:server=y,transport=dt_socket,address=8000,suspend=n \
       -jar target/myproject-0.0.1-SNAPSHOT.jar
```

- 存储引擎  
哈希：对应键值存储系统，仅支持随机读取。内存存储了主键和内容的索引信息，磁盘存储了主键和内容的实际内容。  
B树：对应关系数据库存储系统，支持顺序扫描  
LSM树：