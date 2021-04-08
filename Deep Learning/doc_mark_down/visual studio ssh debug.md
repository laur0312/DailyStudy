---
typora-root-url: ..\img_ssh_debug
---

## Debug Alpha

- 修改CMakeLists.txt编译模式为Debug

  ```cmake
  #g++ configuration in linux
  #set(CMAKE_BUILD_TYPE Release)
  set(CMAKE_BUILD_TYPE Debug)
  ```

  <p align="center">
  	<img src="Alpha_CMakeList.jpg"/>  
  </p>

- 配置选项选择 Linux-Debug，编译

  <p align="center">
  	<img src="Alpha_Linux_Debug.jpg"/>  
  </p>

- 选择对应启动项，并完成参数配置

  <p align="center">
  	<img src="Alpha_run.jpg"/>  
  </p>



## Debug Alpha-acl

- 运行环境

  ```dockerfile
  version: '2.4'
  services:
    yanghua_alpha_api:
      image: harbor.do.proxima-ai.com/alpha/alpha-deploy:1.3
      restart: always
      env_file:
        - alpha.env
      environment:
        - LANG=C.UTF-8
        - ENV=RUN
      ports:
        - "30052:22"
        - "30053:8088"
      security_opt:
        - seccomp:unconfined
      container_name: yanghua-alpha-prod
      volumes:
        - /data:/data
        - /fileser:/fileser
        - /ssd:/ssd
        - /ssd2:/ssd2
        - /home:/home
        - /etc/localtime:/etc/localtime:ro
        - ../third/lib:/opt/third/lib
        - ../third/include:/opt/third/include
        - ../alpha:/opt/alpha
        - ../acl:/opt/acl
        - ../install:/opt/alpha/install
        - ./:/opt/alpha/run
        - ../test_case:/opt/alpha/test_case
        - ../bad_case:/opt/alpha/bad_case
        - /opt/ringV2/deploy/static/cornerstone:/data/ring/deploy/static/cornerstone
      network_mode:
        bridge
      shm_size: 48G
  ```

- 依赖项

  - Alpha

    - 【已默认实现】使用CPack打包（抽取头文件和库文件）

      1. 在根目录CMakeLists.txt的底部添加几行代码

        ```cmake
        include (InstallRequiredSystemLibraries)
        set (CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/License.txt")
        set (CPACK_PACKAGE_VERSION_MAJOR "${VERSION_INFO_MAJOR}")
        set (CPACK_PACKAGE_VERSION_MINOR "${VERSION_INFO_MAJOR}")
        include (CPack)
        ```

      2. 在各自项目中添加要打包的文件，以common项目中的CommonUtils模块为例，在 "Alpha\common\utils\CMakelist.txt" 末尾添加如下代码

         ```cmake
         install (TARGETS CommonUtils DESTINATION bin)
         install (FILES ${CMAKE_CURRENT_SOURCE_DIR}/inc/CommonUtils.h DESTINATION include/common)
         ```

    - 在docker镜像内任意目录拉取Alpha仓库，并执行 sh ./build.sh，在 build目录生成如下产物

      <p align="center">
      	<img src="Alpha_CPack.jpg"/>  
      </p>

    - 在build目录，执行如下命令生成打包文件：ALPHA-xxx-Linux.sh，ALPHA-xxx-Linux..tar.gz，ALPHA-xxx-Linux.tar.z

      ```sh
      cpack -C CPackConfig.cmake
      ```

    - 执行如下命令，解压生成ALPHA-xxx-Linux目录，提取头文件和库文件

      ```shell
      sh ./ALPHA-5.5.1-Linux.sh
      ```

      <p align="center">
      	<img src="Alpha_unCPack.jpg"/>  
      </p>

    - 将bin，config，include，lib，model文件夹下内容拷贝到/opt/alpha对应目录

  - Hemo

    - 在镜像内任意目录拉取Hemo仓库，并执行 sh ./build.sh（CMakeLists.txt会将产物拷贝至/opt/alpha对应目录）

- 修改CMakeLists.txt编译模式为Debug

  ```cmake
  #g++ configuration in linux
  #set(CMAKE_BUILD_TYPE Release)
  set(CMAKE_BUILD_TYPE Debug)
  ```

  <p align="center">
  	<img src="Alpha-acl_CMakeList.jpg"/>  
  </p>

- 配置选项选择 Linux-Debug，编译

  <p align="center">
  	<img src="Alpha-acl_Linux_Debug.jpg"/>  
  </p>

- 选择对应启动项，并完成参数配置

  <p align="center">
  	<img src="Alpha-acl_run.jpg"/>  
  </p>

