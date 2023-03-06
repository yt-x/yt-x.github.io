[TOC]



# OPEN CV基础

[OpenCV官方界面]: https://docs.opencv.org/3.3.1/index.html



## 图像简介

图像是由像素点组成的，像素点的矩阵就组成图像大小

计算机是01编码制，数字图像也是用01来记录信息，一般接触的都是八位数图像，0是最黑，255是最白

像素点：比如说`[500,500,3]` 就是`分别代表h，w，像素通道`

### H S V 颜色模型

HSV（Hue Saturation Value）颜色模型是面向用户的

HSV(Hue, Saturation, Value)是根据颜色的直观特性由A. R. Smith在1978年创建的一种颜色空间, 也称六角锥体模型(Hexcone Model)。

这个模型中颜色的参数分别是：色调（H），饱和度（S），明度（V）

#### 色调

用角度度量，取值范围为0°～360°，从红色开始按逆时针方向计算，红色为0°，绿色为120°,蓝色为240°。它们的补色是：黄色为60°，青色为180°,紫色为300°；

#### 饱和度

饱和度S表示颜色接近光谱色的程度。一种颜色，可以看成是某种光谱色与白色混合的结果。其中光谱色所占的比例愈大，颜色接近光谱色的程度就愈高，颜色的饱和度也就愈高。饱和度高，颜色则深而艳。光谱色的白光成分为0，饱和度达到最高。通常取值范围为0%～100%，值越大，颜色越饱和。

#### 明度

明度表示颜色明亮的程度，对于光源色，明度值与发光体的光亮度有关；对于物体色，此值和物体的透射比或反射比有关。通常取值范围为0%（黑）到100%（白）。

<img src="https://bkimg.cdn.bcebos.com/pic/8d5494eef01f3a29fb2420739925bc315d607c9b?x-bce-process=image/resize,m_lfit,w_220,h_220,limit_1/format,f_auto" alt="HSV颜色空间模型" style="zoom:150%;" />

## 图像分类

### 二值图像

![二值图像是指：每个像素点均为黑色或者白色的图像。二值图像一般用来描述字符图像，其优点是占用空间少，缺点是，当表示人物，风景的图像时，二值图像只能展示其边缘信息，图像内部的纹理特征表现不明显。这时候要使用纹理特征更为丰富的灰度图像](D:\typora\src\image-20210911131838722.png)

### 灰度图像与彩色图像

一幅完整的图像，是由红色、绿色、蓝色三个[通道](https://baike.baidu.com/item/通道)组成的。红色、绿色、蓝色三个通道的缩览图都是以[灰度](https://baike.baidu.com/item/灰度)显示的。用不同的灰度色阶来表示“ 红，绿，蓝”在图像中的比重。通道中的纯白，代表了该色光在此处为最高亮度，亮度级别是255。

通道是整个Photo shop显示图像的基础。色彩的变动，实际上就是间接在对通道灰度图进行调整。通道是Photo shop处理图像的核心部分，所有的色彩调整工具都是围绕在这个核心周围使用的。

在计算机领域中，这类图像通常显示为从最暗黑色到最亮的白色的灰度，尽管理论上这个采样可以任何颜色的不同深浅，甚至可以是不同亮度上的不同颜色。灰度图像与黑白图像不同，在计算机图像领域中黑白图像只有黑色与白色两种颜色；灰度图像在黑色与白色之间还有许多级的颜色深度。但是，在数字图像领域之外，“黑白图像”也表示“灰度图像”，例如灰度的照片通常叫做“黑白照片”。在一些关于数字图像的文章中单色图像等同于灰度图像，在另外一些文章中又等同于黑白图像。

我们可以通过下面几种方法，将其转换为灰度：

1.浮点算法：Gray=R * 0.3+G * 0.59+B * 0.11

2.整数方法：Gray=(R * 30+G * 59+B * 11)/100

3.移位方法：Gray =(R * 76+G * 151+B * 28)>>8;

4.[平均值法](https://baike.baidu.com/item/平均值法)：Gray=（R+G+B）/3;

5.仅取绿色：Gray=G；

通过上述任一种方法求得Gray后，将原来的RGB(R,G,B)中的R,G,B统一用Gray替换，形成新的颜色RGB(Gray,Gray,Gray)，用它替换原来的RGB(R,G,B)就是灰度图了。

![image-20210911131936378](D:\typora\src\image-20210911131936378.png)





## 数据读取--图像

`cv2.imread`

`cv2.IMREAD_COLOR`:彩色图像读取   可以使用1 代替

`cv2.IMREAD_GRAYSCALE` ：灰度图像 可以使用0代替

`cv2.IMREAD_UNCHANGED 包括alpha（透明度）通道的加载图像模式·`  可以使用-1代替

`cv2.waitKey()`是让程序暂停的意思，参数是等待时间（毫秒ms）的时间一到，会继续执行接下来的程序，**传入0的话表示一直等待等待期间也可以获取用户的按键输入**

```python
import numpy as np
import cv2
img=cv2.imread("NV.jpg",0)#加载灰度图像
cv2.imshow("image",img)
cv2.waitKey(0) 
"如果路径有错误，会返回None值，但并不会报错"
```

可以先用`cv2.namedWindow()`创建一个画面，之后再显示

参数 1 仍然是图片的，参数 2 默认是`cv2.WINDOW_AUTOSIZE`，表示图片大小图片，也可以设置为`cv2.WINDOW_NORMAL`，表示图片大小可调整。

```python
# 先定义窗口，后显示图片
cv2.namedWindow('lena2', cv2.WINDOW_NORMAL)
cv2.imshow('lena2', img)
cv2.waitKey(0)
```



## 显示图像 

`cv2.imshow(windows_name, image)`    用的是B G R通道

`imshow`函数作用是在窗口中显示图像，窗口自动适合于图像大小，我们也可以通过`imutils`模块调整显示图像的窗口的大小

windows_name： 窗口名称(字符串)
image： 图像对象，类型是`numpy`中的`ndarray`类型

在这之后需要调用 `cv2.waitKey()它的唯一参数是它应该等待用户输入多长时间（以毫秒给图像绘制留下时间，否则窗口会出现无响应的情况，并且图像无法显示出来`

也就是说`cv2,imshow`后面必须跟`waitKey()`否则无法显示

这里可以通过`imutils`模块改变图像显示大小，下面示例展示

```python
cv2.imshow('image',img) 
cv2.imshow('image',imutils.resize(img,800)) #利用imutils模块调整图片尺寸

```

除了 `opencv` 外也可以调用 `matplotlib`对图像进行展示 

```python
#matplotlib.pyplot  展示
plt.imshow(img[:,:,::-1])  # cv2是BGR  而plt是RGB需要换一下通道
plt.show()  #彩色图展示
```

![image-20210912141915286](D:\typora\PYTHON\OpenCV\image-20210912141915286.png)

```python
#灰度图展示
plt.imshow(img,cmap=plt.cm.gray)
plt.show

```

![](D:\typora\src\image-20210912140643353.png)

## 保存图像

`imwrite 函数保存图像`

`cb2.imwrite(image_filename,image)`

函数参数一： 保存的图像名称(字符串)
函数参数二： 图像对象，类型是`numpy`中的`ndarray`类型

```python
cv2.imwrite('img.jpg', img)   #将图像保存成jpg文件
cv2.imwrite('img2.png', img) #将图像保存成png文件
```

## 图像读取显示保存练习

1. 打开`lena.jpg`并显示，如果按下q，就保存图片为`'lena_save.bmp'`，否则就结束程序。

   ```python
   import cv2
   
   img = cv2.imread('lena.jpg')
   cv2.imshow('lena', img)
   
   k = cv2.waitKey(0)
   # ord()用来获取某个字符的编码
   
   if k == ord('q'):   #键输入q就会保存
       #cv2.imwrite('lena_save.bmp', img)
       print('已保存')
   
   ```

   

2. `Matplotlib` 是 Python 中常用的一个绘图库，

## 接口文档

- [map对象](https://docs.opencv.org/4.0.0/d3/d63/classcv_1_1Mat.html)
- [`cv2.imread()`](https://docs.opencv.org/4.0.0/d4/da8/group__imgcodecs.html#ga288b8b3da0892bd651fce07b3bbd3a56)
- [`cv2.imshow()`(https://docs.opencv.org/4.0.0/d7/dfc/group__highgui.html#ga453d42fe4cb60e5723281a89973ee563)
- [`cv2.imwrite()`](https://docs.opencv.org/4.0.0/d4/da8/group__imgcodecs.html#gabbc7ef1aa2edfaa87772f1202d67e0ce)
- [`cv.namedWindow()`](https://docs.opencv.org/4.0.0/d7/dfc/group__highgui.html#ga5afdf8410934fd099df85c75b2e0888b)

## 窗口销毁函数

当我们使用`imshow`函数展示图像时，最后需要在程序中对图像展示窗口进行销毁，否则程序将无法正常终止

`cv2.destroyWindow(windows_name)` 销毁单个特定窗口，参数： 将要销毁的窗口的名字
`cv2.destroyAllWindows() `销毁全部窗口，无参数

销毁窗口，不能图片窗口一出现我们就将窗口销毁，这样便没法观看窗口,应该采用以下方式

`cv2.waitKey(time_of_milliseconds)`

1.让窗口停留一段时间就销毁

2.接收指令，如接收指定的键盘敲击后结束窗口

`参数：time_of_milliseconds`  大于0，此时的参数表示时间，单位是毫秒，表示等待一定毫秒后自动销毁窗口

```python
#表示等待10秒后，将销毁所有图像
if cv2.waitKey(10000):
    cv2.destroyAllWindows()
#表示等待10秒，将销毁窗口名称为'image'的图像窗口
if cv2.waitKey(10000):
    cv2.destroyWindow('image')
```

参数小于等于0时： 此时窗口将等待一个键盘指令，接收到指令后就会进行窗口销毁，这个指令是可以自动定义的

```python
#当指定waitKey(0) == 27时当敲击键盘 Esc 时便销毁所有窗口
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()
#当接收到键盘敲击A时，便销毁名称为'origin image'的图像窗口
if cv2.waitKey(-1) == ord('A'):
    cv2.destroyWindow('origin image')
```

## 颜色空间装换

### *图像色彩空间（color space）变换函数 `cv2.cvtColor`

`cv2.cvtColor(input_image,flag)` 

参数一： input_image表示将要变换色彩的图像,`ndarray`对象

 参数二： 表示图像色彩空间变换的类型,常用有两种

` cv2.COLOR_BGR2GRAY:表示将图像从BGR空间转化成灰度图，最常用` 

`cv2.COLOR_BGR2HSV:表示将图像从RGB空间转换到HSV空间`

如果需要查看flag所有的类型，可以通过以下程序

```python
import cv2
flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
print(flags)
#共有274种
```

## 为何总是对灰度图进行处理？

**图像的颜色主要是由于图像受到外界光照影响随之产生的不同颜色信息，同一个背景物的图像在不同光源照射下产生的不同颜色效果的图像，因此在我们做图像特征提取和识别过程时，我们要的是图像的梯度信息，也就是图像的本质内容，而颜色信息会对我们对梯度信息提取造成一定的干扰，因此我们会在做图像特征提取和识别前将图像转化为灰度图，这样同时也降低了处理的数据量并且增强了处理效果。**

## 绘制图像

### 原理

一个长宽分别为w、h的 R G B彩色图像来说，它的每个像素值是由(B、G、R)的一个tuple组成，`opencv-python`中每个像素三个值的顺序是B、G、R，而对于灰度图像来说，每个像素对应的便只是一个整数，如果要把像素缩放到0、1，则灰度图像就是二值图像，0便是黑色，1便是白色。我们通过下面的例子来理解一下

处理的图像如下![NV](D:\typora\src\NV.jpg)

```python
import cv2
rgb_img = cv2.imread('E:/pycharm/opencv/new/NV.jpg')
print(rgb_img.shape)  # (676, 1202, 3)  # h有676个像素点，w有1202个像素点，3就是三通道，也就是说看到的是彩图
print(rgb_img[0, 0])  # [16 11 12]
print(rgb_img[0, 0, 0]) # 16

gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)  #灰度
print(gray_img.shape) #(676, 1202)
print(gray_img[0, 0])#12
```

从以上程序运行结果可以得知，彩色图像的高度height=676，宽度=1202 ，通道数为3     像素(0,0)的值为（16 11 12）   即 R=16 G=11  B=12 

但是对于灰度图像来说就是单通道的

因此(0, 0, 0)便是代表一个黑色像素，(255, 255, 255)便是代表一个白色像素。这么想，B=0, G=0,  R=0相当于关闭了颜色通道也就相当于无光照进入，所以图像整个是黑的，而(255, 255, 255)即B=255, G=255, R=255，  相当于打开了B、G、R所有通道光线全部进入，因此便是白色。上图的灰度就是12即[12 12 12]

### 创建一个简单的图

```python
import cv2
import numpy as np

white_img = np.ones((512,512,3), np.uint8)
white_img = 255*white_img
cv2.imshow('white_img', white_img)
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()
```

<img src="D:\typora\src\image-20210912192702186.png" alt="image-20210912192702186" style="zoom:33%;" />

**公共参数：**

**`img`：表示需要进行绘制的图像对象ndarray**
**`color`：表示绘制几何图形的颜色，采用`BGR`即上述说的(B、G、R)**
**`thickness`表示绘制几何图形中线的粗细，默认为1，对于圆、椭圆等封闭图像取-1时是填充图形内部**
**`lineType `表示绘制几何图形线的类型，默认8-connected线是光滑的，当取`cv2.LINE_AA`时线呈现锯齿状**

#### `cv2.line` 直线绘制函数



```python
cv2.line(image, starting, ending, color, thickness, lineType)
#starting,ending 分别表示线的起点像素坐标，终点像素坐标
```



#### `cv2.rectangle`   矩形

```python
cv2.rectangle(image, top-left, bottom-right, color, thickness, lineType)
#top-left , bottom-right 分别表示长方形左上角像素坐标、右下角像素坐标
```



#### `cv2.circle`  圆

```python
cv2.circle(image, center, radius, color, thickness, lineType)
# center 表示圆的圆心像素坐标
# radius 圆的半径长度
#当thickness=-1时，绘制的圆是实心圆，当thickness>=0时绘制的是空心圆
```



#### `cv2.ellipse `  椭圆

```python
cv2.circle(image, center, (major-axis-length, minor-axis-length), angle, startAngle, endAngle, color, thickness, lineType)
#当参数thickness = -1 时绘制的是实心椭圆，当thickness >= 0 时绘制的是空心椭圆
# center： 表示椭圆中心像素坐标
# major-axis-length： 表示椭圆的长轴长度
# minor-axis-length： 表示椭圆的短轴长度
# angle： 表示椭圆在逆时针方向旋转的角度
# startAngle： 表示椭圆从主轴向顺时针方向测量的椭圆弧的起始角度
# endAngle： 表示椭圆从主轴向顺时针方向测量的椭圆弧的终止时角度
```



#### `cv2.polylines` 多边形

```python
cv2.polylines(image, [point-set], flag, color, thickness, lineType)
# [point-set]： 表示多边形点的集合，如果多边形有m个点，则便是一个m12的数组，表示共m个点
# flag： 当flag = True 时，则多边形是封闭的，当flag = False 时，则多边形只是从第一个到最后一个点连线组成的图像，没有封闭
```

cv2.putText

#### 实例

```python
import cv2
import numpy as np

img = np.ones((512,512,3), np.uint8)
img = 255*img
img = cv2.line(img, (100,100), (400,400),(255, 0, 0), 5)
img = cv2.rectangle(img,(200, 20),(400,120),(0,255,0),3)
img = cv2.circle(img,(100,400), 50, (0,0,255), 2)
img = cv2.circle(img,(250,400), 50, (0,0,255), 0)
img = cv2.ellipse(img,(256,256),(100,50),0,0,180,(0, 255, 255), -1)
pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
img = cv2.polylines(img,[pts],True,(0, 0, 0), 2)

cv2.imshow('img', img)
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()
```

<img src="D:\typora\src\image-20210912194703373.png" alt="image-20210912194703373" style="zoom:50%;" />

## 对图像的简单像素操作

对于一个图像，每个像素点都有其对应的坐标`img[height,width,颜色通道]`而对于彩色图像每个像素点都是由[g,b,r]组成的

- `img[y,x]`获取/设置像素点值，`img.shape`：图片的形状（行数(height)、列数(width)、通道数），：`img.dtype`图像的数据类型。
- `img[y1:y2,x1:x2]`进行ROI截取，`cv2.split()/cv2.merge()`通道分割/合并。更推荐的获取单通道方式：`b = img[:, :, 0]`

#### 获取和修改像素点值

操作以后，计算机内存中img的像素点是改变了 但是因为并没有保存，因此原图是没有更改的

```python
import cv2
img = cv2.imread('lena.jpg')
# 1.获取像素的值 b g r
px = img[100, 90]
print(px)  # [103 98 197]

# 只获取蓝色blue通道的值
px_blue = img[100, 90, 0]
print(px_blue)  # 103

# 2.修改像素的值
img[100, 90] = [255, 255, 255]
print(img[100, 90])  # [255 255 255]

# 3.图片形状
print(img.shape)  # (263, 247, 3)
# 形状中包括行数、列数和通道数
height, width, channels = img.shape
# img是灰度图的话：height, width = img.shape

# 总像素数 h*W*通道数
print(img.size)  # 263*247*3=194883
# 数据类型
print(img.dtype)  # uint8   0-256


# 4.ROI截取   截取部分图像数据
face = img[100:200, 115:188]
cv2.imshow('face', face)
cv2.waitKey(0)


# 5.通道分割与合并
b, g, r = cv2.split(img)
img = cv2.merge((b, g, r))
# 更推荐的获取某一通道方式
b = img[:, :, 0]
cv2.imshow('b', b)
cv2.waitKey(0)

```



#### 对图像取反

`reverse_img = 255 - gray_img  `

```python
'''[[12 14 14 ... 18 21 19]
 [11 14 13 ... 23 20 17]
 [13 13 13 ... 21 23 20]
 ...
 [20 22 22 ... 31 30 28]
 [19 19 22 ... 28 29 29]
 [21 19 16 ... 33 30 30]]'''
#变成
'''
[[243 241 241 ... 237 234 236]
 [244 241 242 ... 232 235 238]
 [242 242 242 ... 234 232 235]
 ...
 [235 233 233 ... 224 225 227]
 [236 236 233 ... 227 226 226]
 [234 236 239 ... 222 225 225]]'''
```

<img src="D:\typora\src\image-20210920180815747.png" alt="image-20210920180815747" style="zoom:50%;" />

#### 对图像像素线性变换

```python
for i in range(gray_img.shape[0]):
    for j in range(gray_img.shape[1]):
        random_img[i, j] = gray_img[i, j]*1.2 #灰度图所有像素都成了1.2  这里的[i,j]就代表一个一个的像素点
```

<img src="D:\typora\src\image-20210920180922849.png" alt="image-20210920180922849" style="zoom:50%;" />

#### 截取部分图像数据 ROI

学了特征后，就可以自动截取

先了解下这个图像的坐标左上角是原点，x轴与常规相同，y轴朝下  

下例中cat就是对NV这个图片进行切片，h切`0:200`,w也切`0:200`

```python
rgb_img = cv2.imread('E:/pycharm/opencv/new/NV.jpg')
cat=rgb_img[0:200,0:200]
cv2.imshow('img',rgb_img)
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()
```

![截取到的](D:\typora\src\image-20210920184022634.png)

#### 颜色通道的分割与合并

彩色图的`BGR`三个通道是可以分开单独访问的，可以也。将单独的三个通道合并分类中翻译一副图像分别使用`cv2.split()`状语从句：`cv2.merge()`

这个效率比较低

```python
b, g, r = cv2.split(img)
img = cv2.merge((b, g, r))
```

用`num`索引的方法比较简单

```python
b = img[:, :, 0]
cv2.imshow('blue', b)
cv2.waitKey(0)
```

## 掩膜mask

物理的角度
在半导体制造中，许多芯片工艺步骤采用光刻技术，用于这些步骤的图形“底片”称为掩膜（也称作“掩模”），其作用是：在硅片上选定的区域中对一个不透明的图形模板遮盖，继而下面的腐蚀或扩散将只影响选定的区域以外的区域。
图像掩膜与其类似，用选定的图像、图形或物体，对处理的图像（全部或局部）进行遮挡，来控制图像处理的区域或处理过程。

 数字图像处理中,图像掩模主要用于：

①提取感兴趣区,用预先制作的感兴趣区掩模与待处理图像相乘,得到感兴趣区图像,感兴趣区内图像值保持不变,而区外图像值都为0。

②屏蔽作用,用掩模对图像上某些区域作屏蔽,使其不参加处理或不参加处理参数的计算,或仅对屏蔽区作处理或统计。

③结构特征提取,用相似性变量或图像匹配方法检测和提取图像中与掩模相似的结构特征。

④特殊形状图像的制作。用选定的图像、图形或物体,对待处理的图像(全部或局部)进行遮挡,来控制图像处理的区域或处理过程。用于覆盖的特定图像或物体称为掩模或模板。

![这里写图片描述](https://img-blog.csdnimg.cn/img_convert/e1eee55fde4d735986d0108c6004db3a.png)

## 图像基本运算

bitwise_and、bitwise_or、bitwise_xor、bitwise_not这四个按位操作函数。

bitwise_and是对二进制数据进行“与”操作，即对图像（灰度图像或彩色图像均可）每个像素值进行二进制“与”操作，1&1=1，1&0=0，0&1=0，0&0=0
bitwise_or是对二进制数据进行“或”操作，即对图像（灰度图像或彩色图像均可）每个像素值进行二进制“或”操作，1|1=1，1|0=0，0|1=0，0|0=0
bitwise_xor是对二进制数据进行“异或”操作，即对图像（灰度图像或彩色图像均可）每个像素值进行二进制“异或”操作，1 ^ 1=0,1 ^ 0=1,0 ^ 1=1,0^0=0
bitwise_not是对二进制数据进行“非”操作，即对图像（灰度图像或彩色图像均可）每个像素值进行二进制“非”操作，~ 1 =0，~0=1



## 颜色分离

对于一张彩色图像，若有需要对某种颜色进行分离出来，在 OpenCV 中需要使用到 cv2.inRange 和 cv2.bitwise_and 两个函数。

首先介绍一下两个函数：

1、cv2.inRange

cv2.inRange(src, lowerb, upperb)
用以确认元素值是否介于某个区域
inRange 函数需要设定三个参数，其中 src 指源图像；lowerb 指图像中低于 lowerb 的值，其所对应的图像值将为 0；upperb指图像中高于 upperb 的值，图像值变为 0 。换言之，源图像中仅有图像值介于 lowerb 和 upperb 之间的值才不为 0 ，且值将变成 255
2、cv2.bitwise_and

cv2.bitwise_and(src1, scr2, mask=)
用于对两个数组（图像也是数组）对位元素进行运算，即计算机中的“和”运算。以二进制为例，1&1输出 1 ，1&0、0&1、0&0则均输出 0 。
bitwise_and 函数需要设定三个参数，其中 src1 指第一个数组（源图像），src2 指第二个数组（源图像），mask= 用于指定具体的掩模（常以 0 和 1 元素为主，用以输出具体的元素），应设为 uint8 格式，即单通道的 8-bit 的数组。另外，mask 参数为可选参数，可省略。
言归正传，以某张船的照片为例，分离出图像的蓝色。

```python
# 导入模块，输出原图
import cv2
import matplotlib.pyplot as plt
ship_rgb = cv2.imread('ship.jpg')[:,:,::-1]
plt.imshow(ship_rgb)
plt.show()
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210302103907984.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80ODI0OTU2Mw==,size_16,color_FFFFFF,t_70#pic_center)

```python
# 将图像转为HSV格式进而得到mask，HSV分别代表色相(Hue)、饱和度(Saturation)、明度(Value)
ship_hsv = cv2.cvtColor(ship_rgb, cv2.COLOR_RGB2HSV)
# 设定参数lowerb、upperb
import numpy as np
lowerb = np.array([100,100,100])
upperb = np.array([140,255,255])
# 获取mask
mask = cv2.inRange(ship_hsv, lowerb, upperb)
# 利用mask进行颜色分离
ship_masked = cv2.bitwise_and(ship_bgr,ship_bgr,mask=mask)
# 转回RGB格式
ship_blue = cv2.cvtColor(ship_masked,cv2.COLOR_BGR2RGB)
```



## 拍摄与本地视频的读取与处理：

`cv2.ViedoCapture()`可以捕获摄像头，用数字来控制不同的设备，例如0,1

如果是视频文件直接指定路径就行

`cap = cv2.VideoCapture()`创建视频捕捉对象**cap**
其中参数可以可以写本地路径或者打开设备摄像头。
`ret, frame = cap.read()；`
1）ret,frame是获read()方法的两个返回值，其中ret是布尔值，如果读取帧是正确的则返回True，如果文件读取到结尾，它的返回值就False；
2）frame就是每一帧的图像，是个三维矩阵（按帧读取）。
这里需要注意的是由于`read`是一帧一帧读取，要么读取一张操作一张，要么将所有的帧全部存到list中统一处理

```python
import cv2

vc=cv2.VideoCapture('C:/Users/XYT/Desktop/机械创新设计大赛/Modern Forest Planting Machine   Amazing life  #1 00_00_00-00_01_36.mp4')   #视频文件指定路径  记得改斜杠
if vc.isOpened(): 
    open,frame=vc.read()  # read() 返回两个值，其中open是布尔类型，就是如果你读取到了视频中的这一帧，那么就返回True 反之。frame接收到的是这一帧图像
else:
    open=False
while open:  #利用循环一帧一帧的播放视频，每次获取一帧
    ret,frame=vc.read()
    if frame is None:
        break
    if ret == True:
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #将这一帧图像转换成灰度图
        cv2.imshow('gray_img',gray)
        if cv2.waitKey(10)&0xFF==27:
            break
vc.release()
cv2.destroyAllWindows()
```

#### 获取摄像头属性（视频捕捉属性）`cap.get(propId)`

通过`cap.get(propId)`采集摄像头的一些属性，比如设备属性的属性，可以参考从0~18的属性

也可以使用`cap.set(propId,value)`来修改属性值

```python
# 获取捕获的分辨率
# propId可以直接写数字，也可以用OpenCV的符号表示
width, height = capture.get(3), capture.get(4)
print(width, height)

# 以原分辨率的一倍来捕获
capture.set(cv2.CAP_PROP_FRAME_WIDTH, width * 2)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height * 2)

'''import cv2
cap=cv2.VideoCapture(0)
print(cap.get(3),cap.get(4))  # 640 489   分别代表h， 与 w
cap.set(3,cap.get(3)*2)
cap.set(4,cap.get(4)*2)
if cap.isOpened():
    ret,frame=cap.read()
else:
    print('摄像头未正常开启')
while ret :
    ret,frame=cap.read()
    if frame is None:
        break
    if ret == True:
        cv2.imshow('cap_mp4',frame)
        if cv2.waitKey(25) == ord('s'):
            break
cap.release()
cv2.destroyAllWindows()
'''

```



​	'[VideoCaptureProperties](https://docs.opencv.org/4.0.0/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d)' 

| `CAP_PROP_POS_MSEC Python：cv.CAP_PROP_POS_MSEC`           | 0    | 以毫秒为单位的视频文件的当前位置。                           |
| :--------------------------------------------------------- | ---- | ------------------------------------------------------------ |
| `CAP_PROP_POS_FRAMES Python：cv.CAP_PROP_POS_FRAMES`       | 1    | 下一个要解码/捕获的帧的基于 0 的索引。                       |
| `CAP_PROP_POS_AVI_RATIO Python：cv.CAP_PROP_POS_AVI_RATIO` | 2    | 视频文件的相对位置：0=影片开头，1=影片结尾。                 |
| `CAP_PROP_FRAME_WIDTH Python：cv.CAP_PROP_FRAME_WIDTH`     | 3    | 视频流中帧的宽度。                                           |
| `CAP_PROP_FRAME_HEIGHT Python：cv.CAP_PROP_FRAME_HEIGHT`   | 4    | 视频流中帧的高度。                                           |
| `CAP_PROP_FPS Python：cv.CAP_PROP_FPS`                     | 5    | 帧率。                                                       |
| `CAP_PROP_FOURCC Python：cv.CAP_PROP_FOURCC`               | 6    | 编解码器的 4 字符代码。见[VideoWriter::fourcc](https://docs.opencv.org/4.0.0/dd/d9e/classcv_1_1VideoWriter.html#afec93f94dc6c0b3e28f4dd153bc5a7f0)。 |
| `CAP_PROP_FRAME_COUNT Python：cv.CAP_PROP_FRAME_COUNT`     | 7    | 视频文件中的帧数。                                           |
| `CAP_PROP_FORMAT Python：cv.CAP_PROP_FORMAT`               | 8    | [VideoCapture::retrieve()](https://docs.opencv.org/4.0.0/d8/dfe/classcv_1_1VideoCapture.html#a9ac7f4b1cdfe624663478568486e6712)返回的 Mat 对象的格式。 |
| `CAP_PROP_MODE Python：cv.CAP_PROP_MODE`                   | 9    | 指示当前捕获模式的后端特定值。                               |
| `CAP_PROP_BRIGHTNESS Python：cv.CAP_PROP_BRIGHTNESS`       | 10   | 图像的亮度（仅适用于支持的相机）。                           |
| `CAP_PROP_CONTRAST Python：cv.CAP_PROP_CONTRAST`           | 11   | 图像对比度（仅适用于相机）。                                 |
| `CAP_PROP_SATURATION Python：cv.CAP_PROP_SATURATION`       | 12   | 图像的饱和度（仅适用于相机）。                               |
| `CAP_PROP_HUE Python：cv.CAP_PROP_HUE`                     | 13   | 图像的色调（仅适用于相机）。                                 |
| `CAP_PROP_GAIN Python：cv.CAP_PROP_GAIN`                   | 14   | 图像增益（仅适用于支持的相机）。                             |
| `CAP_PROP_EXPOSURE Python：cv.CAP_PROP_EXPOSURE`           | 15   | 曝光（仅适用于支持的相机）。                                 |
| `CAP_PROP_CONVERT_RGB Python：cv.CAP_PROP_CONVERT_RGB`     | 16   | 指示图像是否应转换为 RGB 的布尔标志。                        |

注：部分摄像头设置经验等参数时会被禁用，因为它们有固定的大小支持，一般可以在摄像头的资料页中找到。

#### 录制并保存视频 `VideoWriter`

之前我们用的是`cv2.imwrite()`保存图片，要保存视频，我们需要创建一个`VideoWriter`对象，需要给它保存四个参数：

- 输出的文件名，如'output.avi'
- 编码方式四[CC](https://baike.baidu.com/item/fourcc/6168470?fr=aladdin)码
- 帧率[FPS](https://baike.baidu.com/item/FPS/3227416)
- 要保存大小 (h,w)

`FourCC`是指定视频编码方式的四字节码，所有的编码可参考[Video Codecs](http://www.fourcc.org/codecs.php)。如`MJPG`编码可以这样写：

`cv2.VideoWriter_fourcc(*'MJPG')`

或`cv2.VideoWriter_fourcc('M','J','P','G')`

```python
import cv2
capture = cv2.VideoCapture(0)

# 定义编码方式并创建VideoWriter对象
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
outfile = cv2.VideoWriter('output.avi', fourcc, 55, (640, 480))

while(capture.isOpened()):
    ret, frame = capture.read()

    if ret:
        outfile.write(frame)  # 写入文件
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    else:
        break

```

#### 练习：实现一个可以选择播放播放的属性

`cv2.createTrackbar('R','image',0,255,call_back)`



创建一个轨迹栏并将其附加到指定的窗口

参数

滑动条名称

所在窗口名称

当前的值

最大值

回调函数名称，回调函数默认有一个表示当前值的参数

```python
import cv2
def track_back(x):
    '''
    ### 回调函数，x表示滑块的位置
    '''
    # 更改视频的帧位置 cv.CAP_PROP_POS_FRAME是下一个要捕获的帧的基于0的索引
    capture.set(cv2.CAP_PROP_POS_FRAMES, x)


cv2.namedWindow('window')

capture = cv2.VideoCapture('./demo_video.mp4')
# 获取视频总共多少帧
frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
# 创建滑动条
cv2.createTrackbar('process', 'window', 1, int(frames), track_back)

while(capture.isOpened()):
    ret, frame = capture.read()
    cv2.imshow('window', frame)
    if cv2.waitKey(30) == ord('q'):
        break
```

## 阈值分割

### 固定图像阈值处理 threshold

`ret,dst=cv2.threshold(src,thresh,maxval,type)`

scr: 输入需要处理的原图，只能输入单通道图像，一般是灰度图

dst:输出图

thresh:设置的阈值

`maxval`:对于`THRESH_BINARY`、`THRESH_BINARY_INV`阈值方法用的最大阈值，一般为255

type:阈值的方式，主要有5种，详情可见：[ThresholdTypes](https://docs.opencv.org/4.0.0/d7/d1b/group__imgproc__misc.html#gaa9e58d2860d4afa658ef70a9b1115576)

```python
#五种不同的阈值方法
ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)  #大于阈值127的部分区最大值255也就是白色，小于127部分取0 也就是黑色
ret, th2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)#与THRESH_BINARY 结果相反
ret, th3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC) #所有大于127的就取127在这里进行截断， 而小于的部分不进行改变
ret, th4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO) #大于127的部分保持不变，而其他部分都变为黑色
ret, th5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)  #反转
```

### 自适应阈值

## 卷积基础--图形边框

### 二维卷积：

![img](http://cos.codec.wang/cv2_understand_convolution.jpg)

卷积就是循环对**图像跟一个核逐个元素相乘再求和得到另外一副图像的操作**，比如结果图中第一个元素5是由原图中3×3的区域与3×3的核逐个元素相乘再相加：



算完之后，整个框再往右移一步继续计算，横向计算完后，再往下移一步继续计算……网上有一副很经典的动态图，方便我们理解卷积：

![img](http://cos.codec.wang/cv2_understand_cnn.gif)

### padding

不难发现，前面我们用3×3的核对一副6×6的图像进行卷积，得到的是4×4的图，图片缩小了！那怎么办呢？我们可以**把原图扩充一圈，再卷积，这个操作叫填充padding**。

> 事实上，原图为n×n，卷积核为f×f，最终结果图大小为**(n-f+1) × (n-f+1)。**

![img](http://cos.codec.wang/cv2_understand_padding.jpg)

那么扩展的这一层应该填充什么值呢？`OpenCV`中有好几种填充方式，都使用`cv2.copyMakeBorder()`函数实现，一起来看看。

#### 添加边框

`cv2.copyMakeBorder()`用来给图片添加边框，它有下面几个参数：

- src：要处理的原图
- top, bottom, left, right：上下左右要扩展的像素数( 相应方向上的边框宽度 )
- **`borderType`**：边框类型，这个就是需要关注的填充方式，详情请参考：[ BorderTypes ](https://docs.opencv.org/3.3.1/d2/de8/group__core__array.html#ga209f2f4869e304c82d07739337eae7c5)   下图中第二个镜面反射是默认的边框类型
- ![image-20211015211738420](D:\typora\src\image-20211015211738420.png)

其中默认方式和固定值方式最常用，我们详细说明一下：

#### 固定值填充

顾名思义，`cv2.BORDER_CONSTANT`这种方式就是边框都填充成一个固定的值，比如下面的程序都填充0：

```python
img = cv2.imread('6_by_6.bmp', 0)
print(img)

# 固定值边框，统一都填充0也称为zero padding
cons = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
print(cons)Copy to clipboardErrorCopied
```

![img](http://cos.codec.wang/cv2_zero_padding_output.jpg)

### 默认边框类型

默认边框`cv2.BORDER_DEFAULT`其实是取镜像对称的像素填充，比较拗口，一步步解释：

```python
default = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
print(default)
```

首先进行上下填充，填充成与原图像边界对称的值，如下图：

![img](http://cos.codec.wang/cv2_up_down_padding_first.jpg)

同理再进行左右两边的填充，最后把四个顶点补充上就好了：

![img](http://cos.codec.wang/cv2_right_left_padding_second2.jpg)

> 经验之谈：一般情况下默认方式更加合理，因为边界的像素值更加接近。具体应视场合而定。

### `OpenCV`进行卷积

`OpenCV`中用`cv2.filter2D()`实现卷积操作，比如我们的核是下面这样（3×3区域像素的和除以10）：

![image-20211015212412694](D:\typora\src\image-20211015212412694.png)

```python
img = cv2.imread('lena.jpg')
# 定义卷积核
kernel = np.ones((3, 3), np.float32) / 10
# 卷积操作，-1表示通道数与原图相同
dst = cv2.filter2D(img, -1, kernel)
```

![img](http://cos.codec.wang/cv2_convolution_kernel_3_3.jpg)

可以看到这个核对图像进行了模糊处理，这是卷积的众多功能之一。

## 图像平滑操作

### 滤波与模糊

关于滤波和模糊:

- 它们都属于卷积，不同滤波方法之间只是卷积核不同（对线性滤波而言）
- 低通滤波器是模糊，高通滤波器是锐化

**低通滤波器**就是允许低频信号通过，在图像中边缘和噪点都相当于高频部分，所以低通滤波器用于去除噪点、平滑和模糊图像。**高通滤波器**则反之，用来增强图像边缘，进行锐化处理。

> 常见噪声有[椒盐噪声](https://baike.baidu.com/item/椒盐噪声/3455958?fr=aladdin)(脉冲噪声)和[高斯噪声](https://baike.baidu.com/item/高斯噪声)，椒盐噪声可以理解为斑点，随机出现在图像中的黑点或白点；高斯噪声可以理解为拍摄图片时由于光照等原因造成的噪声。

### 均值滤波

均值滤波是一种最简单的滤波处理，它取的是卷积核区域内元素的均值，用`cv2.blur()`实现，如3×3的卷积核：

![image-20211016133058089](D:\typora\src\image-20211016133058089.png)

```python
img = cv2.imread('lena.jpg')
blur = cv2.blur(img, (3, 3))  # 均值模糊
```

> 所有的滤波函数都有一个可选参数borderType，这个参数就是***卷积基础--图形边框***中所说的边框填充方式。

### 方框滤波

方框滤波跟均值滤波很像，如3×3的滤波核如下：

![image-20211016133349723](D:\typora\src\image-20211016133349723.png)

用`cv2.boxFilter()`函数实现，当可选参数normalize为True的时候，方框滤波就是均值滤波，上式中的a就等于1/9；normalize为False的时候，a=1，相当于求区域内的像素和。

```python
# 前面的均值滤波也可以用方框滤波实现：normalize=True
blur = cv2.boxFilter(img, -1, (3, 3), normalize=True) #卷积操作 -1表示通道数与原图相同
```

### 高斯滤波

前面两种滤波方式，卷积核内的每个值都一样，也就是说图像区域中每个像素的权重也就一样。高斯滤波的卷积核权重并不相同：中间像素点权重最高，越远离中心的像素权重越小，还记得标准正态分布的曲线吗？

![img](http://cos.codec.wang/cv2_gaussian_kernel_function_theory.jpg)

显然这种处理元素间权值的方式更加合理一些。图像是2维的，所以我们需要使用[2维的高斯函数](https://en.wikipedia.org/wiki/Gaussian_filter)，比如OpenCV中默认的3×3的高斯卷积核（具体原理和卷积核生成方式请参考文末的[番外小篇](http://codec.wang/#/)）：

![image-20211016133709083](D:\typora\src\image-20211016133709083.png)

OpenCV中对应函数为`cv2.GaussianBlur(src,ksize,sigmaX)`：

```python
img = cv2.imread('gaussian_noise.bmp')
# 均值滤波vs高斯滤波
blur = cv2.blur(img, (5, 5))  # 均值滤波
gaussian = cv2.GaussianBlur(img, (5, 5), 1)  # 高斯滤波
```

参数3 σx值越大，模糊效果越明显。高斯滤波相比均值滤波效率要慢，但可以有效消除高斯噪声，能保留更多的图像细节，所以经常被称为最有用的滤波器。均值滤波与高斯滤波的对比结果如下（均值滤波丢失的细节更多）：

![img](http://cos.codec.wang/cv2_gaussian_vs_average.jpg)

#### 高斯滤波卷积核   OpenCV中7*7以下的卷积核是算好了的。

![image-20211016134947978](D:\typora\src\image-20211016134947978.png)

我们可以用[`cv2.getGaussianKernel(ksize,sigma)`](https://docs.opencv.org/3.3.1/d4/d86/group__imgproc__filter.html#gac05a120c1ae92a6060dd0db190a61afa)来生成一维卷积核：

- sigma<=0时，`sigma=0.3*((ksize-1)*0.5 - 1) + 0.8`
- sigma>0时，sigma=sigma

```python
print(cv2.getGaussianKernel(3, 0))
# 结果：[[0.25][0.5][0.25]]
```

![image-20211016135226745](D:\typora\src\image-20211016135226745.png)

### 中值滤波

中值又叫中位数，是所有数排序后取中间的值。中值滤波就是用区域内的中值来代替本像素值，所以那种孤立的斑点，如0或255很容易消除掉，**适用于去除椒盐噪声和斑点噪声**。中值是一种非线性操作，效率相比前面几种线性滤波要慢。

比如下面这张斑点噪声图，用中值滤波显然更好：

```python
img = cv2.imread('salt_noise.bmp', 0)
# 均值滤波vs中值滤波
blur = cv2.blur(img, (5, 5))  # 均值滤波
median = cv2.medianBlur(img, 5)  # 中值滤波
```

![img](http://cos.codec.wang/cv2_median_vs_average.jpg)

### 双边滤波

模糊操作基本都会损失掉图像细节信息，尤其前面介绍的线性滤波器，图像的边缘信息很难保留下来。然而，边缘（edge）信息是图像中很重要的一个特征，所以这才有了[双边滤波](https://baike.baidu.com/item/双边滤波)。用`cv2.bilateralFilter()`函数实现：

```python
img = cv2.imread('lena.jpg')
# 双边滤波vs高斯滤波
gau = cv2.GaussianBlur(img, (5, 5), 0)  # 高斯滤波
blur = cv2.bilateralFilter(img, 9, 75, 75)  # 双边滤波
```

![img](http://cos.codec.wang/cv2_bilateral_vs_gaussian.jpg)

**双边滤波明显保留了更多边缘信息，但是由于保存了过多的高频信息，对于彩色图像里的高频噪声，双边滤波器不能够干净的滤掉，只能够对于低频信息进行较好的滤波**。



### 小结

- 在不知道用什么滤波器好的时候，优先高斯滤波`cv2.GaussianBlur()`，然后均值滤波`cv2.blur()`。
- 斑点和椒盐噪声优先使用中值滤波`cv2.medianBlur()`。
- 要去除噪点的同时尽可能保留更多的边缘信息，使用双边滤波`cv2.bilateralFilter()`。
- 线性滤波方式：均值滤波、方框滤波、高斯滤波（速度相对快）。
- 非线性滤波方式：中值滤波、双边滤波（速度相对慢）。

### 接口

- [cv2.blur()](https://docs.opencv.org/4.0.0/d4/d86/group__imgproc__filter.html#ga8c45db9afe636703801b0b2e440fce37)
- [cv2.boxFilter()](https://docs.opencv.org/4.0.0/d4/d86/group__imgproc__filter.html#gad533230ebf2d42509547d514f7d3fbc3)
- [cv2.GaussianBlur()](https://docs.opencv.org/4.0.0/d4/d86/group__imgproc__filter.html#gaabe8c836e97159a9193fb0b11ac52cf1)
- [cv2.getGaussianKernel()](https://docs.opencv.org/4.0.0/d4/d86/group__imgproc__filter.html#gac05a120c1ae92a6060dd0db190a61afa)
- [cv2.medianBlur()](https://docs.opencv.org/4.0.0/d4/d86/group__imgproc__filter.html#ga564869aa33e58769b4469101aac458f9)
- [cv2.bilateralFilter()](https://docs.opencv.org/4.0.0/d4/d86/group__imgproc__filter.html#ga9d7064d478c95d60003cf839430737ed)

### 引用

- [图像平滑处理](http://www.opencv.org.cn/opencvdoc/2.3.2/html/doc/tutorials/imgproc/gausian_median_blur_bilateral_filter/gausian_median_blur_bilateral_filter.html)

## 图像梯度（理论基础）：

https://www.jianshu.com/p/2334bee37de5

低通滤波器是模糊，高通滤波器是锐化

### Sobel算子

https://www.bilibili.com/video/BV1YZ4y1s7Ra?from=search&amp;seid=18409194473369625551&amp;spm_id_from=333.337.0.0

水平方向的边界  也就是左右的；  就是假如这里是边界，那么P5的左右两边他P4和P6的值会相差很大，然后P5的值算出来就会很明显，如果P4和P6很接近那么计算的P5就会很接近0  没那么明显  然后就可以根据这个计算出图像的边界。

<img src="D:\typora\src\image-20211016140326016.png" alt="image-20211016140326016" style="zoom:50%;" />

同理。垂直方向就是吧卷积核改一下

<img src="D:\typora\src\image-20211016140853557.png" alt="image-20211016140853557" style="zoom:50%;" />

垂直方向和水平方向的梯度都计算出来了  那么图像的梯度就可以计算

<img src="D:\typora\src\image-20211016140939825.png" alt="image-20211016140939825" style="zoom:50%;" />

<img src="D:\typora\src\image-20211016141022498.png" alt="image-20211016141022498" style="zoom:50%;" />

```python
#代码表示
sobelx = cv2.Sobel(img, -1, 1, 0, ksize=3)  # 只计算x方向
sobely = cv2.Sobel(img, -1, 0, 1, ksize=3)  # 只计算y方向
```



### ![垂直和水平边界下降](http://cos.codec.wang/cv2_horizen_vertical_edge_detection.jpg)Laplacian（拉普拉斯）算子

拉普拉斯算子类似于**二阶**Sobel导数。

在OpenCV中通过调用sobel算子来计算拉普拉斯算子，使用公式：

![image-20211016142426256](D:\typora\src\image-20211016142426256.png)

使用卷积核：

![image-20211016142440656](D:\typora\src\image-20211016142440656.png)

就算出来就是上下左右全部减一次中间，然后再相加

<img src="D:\typora\src\image-20211017095425784.png" alt="image-20211017095425784"  />

然后判断的依据和sobel算子类似，值小那就是梯度值小，非边界，值大就是梯度大，是边界

OpenCV中直接使用`cv2.Laplacian()`函数：

```python
laplacian = cv2.Laplacian(img, -1)  # 使用Laplacian算子
```

![img](http://cos.codec.wang/cv2_laplacian.jpg)



也可如此理解，更加深入一点。

![image-20211016142242956](D:\typora\src\image-20211016142242956.png)

## 边缘检测：

cv2.Canny()

Canny边缘检测方法被誉为边缘检测的最优方法:

```python
#示例
import cv2
import numpy as np

img = cv2.imread('handwriting.jpg', 0)
edges = cv2.Canny(img, 30, 70)  # canny边缘检测

cv2.imshow('canny', np.hstack((img, edges)))
cv2.waitKey(0)
```

之前采用低通滤波模糊图片，也就是去噪点，而想要得到图像边缘就需要用到高通滤波，锐化图像

### Canny边界检测

1. 使用5×5高斯排除噪音：

   边缘操作本身属于锐化操作，对噪点比较敏感，所以需要进行平滑处理

2.   计算图像梯度方向

   使用sobel算子计算两个方向上的GX和GY，然后算出梯度方向：

   ![image-20211017100727722](D:\typora\src\image-20211017100727722.png)

3.  取局部极大值：

   梯度已经有了轮廓，但为了进一步筛选，可以从（0°/45°/90°/135°）这四个角度方向上取局部极大值

   比如A点在45°方向上大于B/C点，那就保留A然后将B/C设置为0.

   ![img](http://cos.codec.wang/cv2_understand_canny_direction.jpg)

4.  滞后阈值：Canny推荐的高低阈值比在2:1到3:1之间。

   经过前面三步，就只剩下0和可能的边缘梯度值了，为了最终确定下来，需要设定高低阈值：

   ​	像素点的值大于最高阈值,那肯定是边缘

   ​	像素值小于最低阈值那肯定不是边缘

   ​	像素值介于两者之间，如果与高于最高阈值的点连接，也算边缘

   AC是边缘   B不是。

   ![img](http://cos.codec.wang/cv2_understand_canny_max_min_val.jpg)

### 先阈值分割后检测

```python
_, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  #自适应阈值分割 python中如果某个值不用，可以用下划线代替
edges = cv2.Canny(thresh, 30, 70)  #边缘检测，上阈值70 ，下阈值30

cv2.imshow('canny', np.hstack((img, thresh, edges)))
cv2.waitKey(0)
```

![](http://cos.codec.wang/cv2_canny_edge_detection_threshold.jpg)

```
#滑动条调节阈值，理解高地阈值效果：
import cv2
import numpy as np
def track_back(x):
    pass

ori_img=cv2.imread('NV.jpg',0)
cv2.namedWindow('window')
cv2.createTrackbar('upper','window',100,255,track_back)
cv2.createTrackbar('low','window',200,255,track_back)

while(True):
    upper_= cv2.getTrackbarPos('upper','window')
    low_=cv2.getTrackbarPos('low','window')
    edges=cv2.Canny(ori_img,low_,upper_)
    cv2.imshow('window',edges)
    if cv2.waitKey(30)==ord('q'):  #键入 q 关闭窗口   记得一定不要设置为0  无限等待  不然循环无法进行下去
        break
```

## 腐蚀与膨胀

形态学操作  腐蚀  膨胀  开运算和闭运算

cv2.erode()  cv2.dilate()  cv2.morpho;ogyEx()

### 腐蚀

`cv2.erode(src, kernel, iteration)`

参数说明：

src表示图片 

kernel指腐蚀操作的内核，默认是一个简单的3X3矩阵，我们也可以利用`getStructuringElement（）`函数指明它的形状

iterations指的是腐蚀次数，省略是默认为1

```python
import cv2
import numpy as np

img = cv2.imread('j.bmp', 0)
kernel = np.ones((5, 5), np.uint8) #指定核大小
erosion = cv2.erode(img, kernel)  # 腐蚀
```

这个核也叫结构元素，因为形态学操作其实也是应用卷积来实现的结构元素可以是矩形/椭圆/十字形，可以用。`cv2.getStructuringElement()`来生成不同形状的结构元素

```python
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 矩形结构
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 椭圆结构
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))  # 十字形结构
```

![img](http://cos.codec.wang/cv2_morphological_struct_element.jpg)

###  膨胀

`dilate()`可以对输入图像用特定结构元素进行膨胀操作，该结构元素确定膨胀操作过程中的邻域的形状，各点像素值将被替换为对应邻域上的最大值

```python
dilation = cv2.dilate(img, kernel)  # 膨胀
```

### 开运算

`cv2.morphologyEx()`

先腐蚀后膨胀开运算 

作用：分开物体，分解区域

```python
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 定义结构元素  矩形结构

img = cv2.imread('j_noise_out.bmp', 0)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)  # 开运算
```

### 闭运算

先膨胀后腐蚀（先膨胀会使白色的部分扩张，以至于消除/"闭合"物体里面的小黑洞，所以叫闭运算）

```python
img = cv2.imread('j_noise_in.bmp', 0)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)  # 闭运算
```

### 其他操作

- 形态学梯度：膨胀图减去腐蚀图，`dilation - erosion`，这样会得到物体的轮廓：

```python
img = cv2.imread('school.bmp', 0)
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)Copy to clipboardErrorCopied
```

![img](http://cos.codec.wang/cv2_morphological_gradient.jpg)

- 顶帽：原图减去开运算后的图：`src - opening`

```python
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)Copy to clipboardErrorCopied
```

- 黑帽：闭运算后的图减去原图：`closing - src`

```python
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
```

## 轮廓

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200622220127452.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3h5czQzMDM4MV8x,size_16,color_FFFFFF,t_70)

**轮廓是连续的，边缘并不全都连续**（下图）。其实**边缘主要是作为图像的特征使用**，比如可以用边缘特征可以区分脸和手，而**轮廓主要用来分析物体的形态**，比如物体的周长和面积等，可以说边缘包括轮廓。

![边缘和轮廓的区别](http://cos.codec.wang/cv2_understand_contours.jpg)

**寻找轮廓一般用于二值化图像，所以通常会使用阈值分割或者Canny边缘检测得到二值图**

寻找轮廓是针对白色物体的，一定要保证物体是白色，背景是黑色，不然很多人在新专辑轮廓是会找到图片最外面的一个框

### 第一步：寻找轮廓

使用`cv2.findContours(image, mode, method[, contours[, hierarchy[, offset ]]])`

返回两个值：contours;hierarchy

参数：

image 表示寻找轮廓的图像

mode表示轮廓的检索模式，有四种

```python
cv2.RETR_EXTERNAL   #表示只检测外轮廓
cv2.RETR_LIST   #检测的轮廓不建立等级关系
cv2.RETR_CCOMP #建立两个等级的轮廓，上面的一层为外边界，里面的一层为内孔的边界信息。如果内孔内还有一个连通物体，这个物体的边界也在顶层。
cv2.RETR_TREE   #建立一个等级树结构的轮廓
```
method是轮廓的近似方法：

```python
cv2.CHAIN_APPROX_NONE#存储所有的轮廓点，相邻的两个点的像素位置差不超过1，即max（abs（x1-x2），abs（y2-y1））==1
cv2.CHAIN_APPROX_SIMPLE#压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需4个点来保存轮廓信息
cv2.CHAIN_APPROX_TC89_L1，CV_CHAIN_APPROX_TC89_KCOS使用teh-Chinl chain #近似算法
```

### 例子

```python
import cv2
 
img = cv2.imread('D:\\test\\contour.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
 
contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  #寻找轮廓
cv2.drawContours(img,contours,-1,(0,0,255),3)
 
cv2.imshow("img", img)
cv2.waitKey(0)
```

原图：<img src="https://img-blog.csdn.net/20131030153346984?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc3VubnkyMDM4/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast" alt="img" style="zoom:33%;" />检测图<img src="https://img-blog.csdn.net/20131030153441656?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc3VubnkyMDM4/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast" alt="img" style="zoom:33%;" />

**findcontours函数会“原地”修改输入的图像**  也就是说原图被改变了

**contour返回值**
	cv2.findContours()函数首先返回一个list，list中每个元素都是图像中的一个轮廓，用numpy中的ndarray表示。这个概念非常重要。在下面drawContours中会看见。通过

```python
print (type(contours))
print (type(contours[0]))
print (len(contours))
#可以验证上述信息。会看到本例中有两条轮廓，一个是五角星的，一个是矩形的。每个轮廓是一个ndarray，每个ndarray是轮廓上的点的集合。
由于我们知道返回的轮廓有两个，因此可通过
cv2.drawContours(img,contours,0,(0,0,255),3)
和
cv2.drawContours(img,contours,1,(0,255,0),3)
分别绘制两个轮廓，关于该参数可参见下面一节的内容。同时通过
print (len(contours[0]))
print (len(contours[1]))
输出两个轮廓中存储的点的个数，可以看到，第一个轮廓中只有4个元素，这是因为轮廓中并不是存储轮廓上所有的点，而是只存储可以用直线描述轮廓的点的个数，比如一个“正立”的矩形，只需4个顶点就能描述轮廓了。


```

**hierarchy返回值**

```python
此外，该函数还可返回一个可选的hiararchy结果，这是一个ndarray，其中的元素个数和轮廓个数相同，每个轮廓contours[i]对应4个hierarchy元素hierarchy[i][0] ~hierarchy[i][3]，分别表示后一个轮廓、前一个轮廓、父轮廓、内嵌轮廓的索引编号，如果没有对应项，则该值为负数。
通过
print (type(hierarchy))
print (hierarchy.ndim)  # 3
print (hierarchy[0].ndim)  # 2
print (hierarchy.shape) # (1, 2, 4)
```

### 绘制轮廓

<img src="D:\typora\src\image-20211018233258899.png" alt="image-20211018233258899" style="zoom:150%;" />

`cv2.drawContours(image, contours, contourIdx, color[, thickness[, lineType[, hierarchy[, maxLevel[, offset ]]]]])`

第一个参数是指明在哪幅图像上绘制轮廓；
第二个参数是轮廓本身，在Python中是一个list。
第三个参数指定绘制轮廓list中的哪条轮廓，如果是-1，则绘制其中的所有轮廓。

后面的参数很简单。其中thickness表明轮廓线的宽度，如果是-1（cv2.FILLED），则为填充模式。绘制参数将在以后独立详细介绍。

## 轮廓层级

很多情况下，图像中的形状之间是有关联的

![img](http://cos.codec.wang/cv2_understand_hierarchy.jpg)

图中总共有8条轮廓，2和2a分别表示外层和里层的轮廓，3和3a也是一样。从图中看得出来：

- 轮廓0/1/2是最外层的轮廓，我们可以说它们处于同一轮廓等级：0级
- 轮廓2a是轮廓2的子轮廓，反过来说2是2a的父轮廓，轮廓2a算一个等级：1级
- **同样3是2a的子轮廓，轮廓3处于一个等级：2级**
- 类似的，3a是3的子轮廓

这里面OpenCV关注的就是两个概念：**同一轮廓等级**和**轮廓间的子属关系**。

### OpenCV中轮廓等级的表示

如果我们打印出`cv2.findContours()`函数的返回值hierarchy，会发现它是一个包含4个值的数组：**[Next, Previous, First Child, Parent]**

- *Next：与当前轮廓处于同一层级的下一条轮廓*

举例来说，前面图中跟0处于同一层级的下一条轮廓是1，所以Next=1；同理，对轮廓1来说，Next=2；*那么没有与它同一层级的轮廓的下一条轮廓了时，此时Next=-1。*

- *Previous：与当前轮廓处于同一层级的上一条轮廓*

跟前面一样，对于轮廓1来说，Previous=0；对于轮廓2，Previous=1；对于轮廓1，没有上一条轮廓了，所以Previous=-1。

- *First Child：当前轮廓的第一条子轮廓*

比如对于轮廓2，第一条子轮廓就是轮廓2a，所以First Child=2a；对轮廓3a，First Child=4。

- *Parent：当前轮廓的父轮廓*

比如2a的父轮廓是2，Parent=2；轮廓2没有父轮廓，所以Parent=-1。

下面我们通过代码验证一下：

```python
import cv2

# 1.读入图片
img = cv2.imread('hierarchy.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 2.寻找轮廓
image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, 2)

# 3.绘制轮廓
print(len(contours),hierarchy)  # 8条
cv2.drawContours(img, contours, -1, (0, 0, 255), 2)  #画出所有的轮廓  红色线条
```

> 经验之谈：OpenCV中找到的轮廓序号跟前面讲的不同噢，如下图：

![img](http://cos.codec.wang/cv2_hierarchy_RETR_TREE.jpg)

现在既然我们了解了层级的概念，那么类似cv2.RETR_TREE的轮廓寻找方式又是啥意思呢？

### 轮廓寻找方式

OpenCV中有四种轮廓寻找方式[RetrievalModes](https://docs.opencv.org/3.3.1/d3/dc0/group__imgproc__shape.html#ga819779b9857cc2f8601e6526a3a5bc71)，下面分别来看下：

### 1. RETR_LIST

这是最简单的一种寻找方式，它不建立轮廓间的子属关系，也就是所有轮廓都属于同一层级。这样，hierarchy中的后两个值[First Child, Parent]都为-1。比如同样的图，我们使用cv2.RETR_LIST来寻找轮廓：

```python
_, _, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, 2)
print(hierarchy)
# 结果如下
[[[ 1 -1 -1 -1]
  [ 2  0 -1 -1]
  [ 3  1 -1 -1]
  [ 4  2 -1 -1]
  [ 5  3 -1 -1]
  [ 6  4 -1 -1]
  [ 7  5 -1 -1]
  [-1  6 -1 -1]]]
```

因为没有从属关系，所以轮廓0的下一条是1，1的下一条是2……

> 经验之谈：**如果你不需要轮廓层级信息的话，cv2.RETR_LIST更推荐使用，因为性能更好。**

### 2. RETR_TREE

cv2.RETR_TREE就是之前我们一直在使用的方式，它会完整建立轮廓的层级从属关系，前面已经详细说明过了。

### 3. RETR_EXTERNAL

这种方式只寻找最高层级的轮廓，也就是它只会找到前面我们所说的3条0级轮廓：

```python
_, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 2)
print(len(contours), hierarchy, sep='\n')
# 结果如下
3
[[[ 1 -1 -1 -1]
  [ 2  0 -1 -1]
  [-1  1 -1 -1]]]
```

![img](http://cos.codec.wang/cv2_hierarchy_RETR_EXTERNAL.jpg)

### 4. RETR_CCOMP

相比之下cv2.RETR_CCOMP比较难理解，但其实也很简单：它把所有的轮廓只分为2个层级，不是外层的就是里层的。结合代码和图片，我们来理解下：

```python
_, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, 2)
print(hierarchy)
# 结果如下
[[[ 1 -1 -1 -1]
  [ 2  0 -1 -1]
  [ 4  1  3 -1]
  [-1 -1 -1  2]
  [ 6  2  5 -1]
  [-1 -1 -1  4]
  [ 7  4 -1 -1]
  [-1  6 -1 -1]]]Copy to clipboardErrorCopied
```

![img](http://cos.codec.wang/cv2_hierarchy_RETR_CCOMP.jpg)

> 注意：使用这个参数找到的轮廓序号与之前不同。

图中括号里面1代表外层轮廓，2代表里层轮廓。比如说对于轮廓2，Next就是4，Previous是1，它有里层的轮廓3，所以First Child=3，但因为只有两个层级，它本身就是外层轮廓，所以Parent=-1。大家可以针对其他的轮廓自己验证一下。

## 轮廓特征

<img src="http://cos.codec.wang/cv2_min_rect_rect_bounding.jpg" alt="img" style="zoom:50%;" />

在计算轮廓特征前，先寻找轮廓

```python
import cv2
import numpy as np

img = cv2.imread('handwriting.jpg', 0)
_, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
image, contours, hierarchy = cv2.findContours(thresh, 3, 2)

# 以数字3的轮廓为例
cnt = contours[0]

```


![img](http://cos.codec.wang/cv2_31_handwriting_sample.jpg)

```python
import cv2
import numpy as np

img = cv2.imread('handwriting.jpg', 0)
_, thresh = cv2.threshLEold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
image, contours, hierarchy = cv2.findContours(thresh, 3, 2)

# 以数字3的轮廓为例
cnt = contours[0]
```

为了便于绘制，我们创建出两幅彩色图，并把轮廓画在第一幅图上：

```python
img_color1 = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
img_color2 = np.copy(img_color1)
cv2.drawContours(img_color1, [cnt], 0, (0, 0, 255), 2)
```

### 轮廓面积

```python
area = cv2.contourArea(cnt)  # 4386.5
```

注意轮廓特征计算的结果并不等同于像素点的个数，而是根据几何方法算出来的，所以有小数。

> 如果统计二值图中像素点个数，应尽量避免循环，**可以使用`cv2.countNonZero()`**，更加高效。

### 轮廓周长

```python
perimeter = cv2.arcLength(cnt, True)  # 585.7
```

参数2表示轮廓是否封闭，显然我们的轮廓是封闭的，所以是True。

### 图像矩

矩可以理解为图像的各类几何特征，详情请参考：[[Image Moments](http://en.wikipedia.org/wiki/Image_moment)]

```python
M = cv2.moments(cnt)
```

M中包含了很多轮廓的特征信息，比如M['m00']表示轮廓面积，与前面`cv2.contourArea()`计算结果是一样的。质心也可以用它来算：

```python
cx, cy = M['m10'] / M['m00'], M['m01'] / M['m00']  # (205, 281)
```

### 外接矩形

形状的外接矩形有两种，如下图，绿色的叫外接矩形，表示不考虑旋转并且能包含整个轮廓的矩形。蓝色的叫最小外接矩，考虑了旋转：

![img](http://cos.codec.wang/cv2_min_rect_rect_bounding.jpg)

```python
x, y, w, h = cv2.boundingRect(cnt)  # 外接矩形
cv2.rectangle(img_color1, (x, y), (x + w, y + h), (0, 255, 0), 2)Copy to clipboardErrorCopied
rect = cv2.minAreaRect(cnt)  # 最小外接矩形
box = np.int0(cv2.boxPoints(rect))  # 矩形的四个角点取整
cv2.drawContours(img_color1, [box], 0, (255, 0, 0), 2)
```

其中np.int0(x)是把x取整的操作，比如377.93就会变成377，也可以用x.astype(np.int)。

### 最小外接圆

外接圆跟外接矩形一样，找到一个能包围物体的最小圆：

```python
(x, y), radius = cv2.minEnclosingCircle(cnt)
(x, y, radius) = np.int0((x, y, radius))  # 圆心和半径取整
cv2.circle(img_color2, (x, y), radius, (0, 0, 255), 2)
```

![img](http://cos.codec.wang/cv2_min_enclosing_circle.jpg)

### 拟合椭圆

我们可以用得到的轮廓拟合出一个椭圆：

```python
ellipse = cv2.fitEllipse(cnt)
cv2.ellipse(img_color2, ellipse, (255, 255, 0), 2)
```

![img](http://cos.codec.wang/cv2_fitting_ellipse.jpg)

### 形状匹配

`cv2.matchShapes()`可以检测两个形状之间的相似度，返回**值越小，越相似**。先读入下面这张图片：

![img](http://cos.codec.wang/cv2_match_shape_shapes.jpg)

```python
img = cv2.imread('shapes.jpg', 0)
_, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
image, contours, hierarchy = cv2.findContours(thresh, 3, 2)
img_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)  # 用于绘制的彩色图
```

图中有3条轮廓，我们用A/B/C表示：

```python
cnt_a, cnt_b, cnt_c = contours[0], contours[1], contours[2]
print(cv2.matchShapes(cnt_b, cnt_b, 1, 0.0))  # 0.0
print(cv2.matchShapes(cnt_b, cnt_c, 1, 0.0))  # 2.17e-05
print(cv2.matchShapes(cnt_b, cnt_a, 1, 0.0))  # 0.418
```

可以看到BC相似程度比AB高很多，并且图形的旋转或缩放并没有影响。其中，参数3是匹配方法，详情可参考：[ShapeMatchModes](https://docs.opencv.org/4.0.0/d3/dc0/group__imgproc__shape.html#gaf2b97a230b51856d09a2d934b78c015f)，参数4是OpenCV的预留参数，暂时没有实现，可以不用理会。

形状匹配是通过图像的Hu矩来实现的(`cv2.HuMoments()`)，大家如果感兴趣，可以参考：[Hu-Moments](http://en.wikipedia.org/wiki/Image_moment#Rotation_invariant_moments)

## 直方图

![img](http://cos.codec.wang/cv2_understand_histogram.jpg)



- 计算并绘制直方图
- （自适应）直方图均衡化
- OpenCV函数：`cv2.calcHist()`, `cv2.equalizeHist()`

### 啥叫直方图

简单来说，直方图就是图像中每个像素值的个数统计，比如说一副灰度图中像素值为0的有多少个，1的有多少个……:

![img](http://cos.codec.wang/cv2_understand_histogram.jpg)

在计算直方图之前，有几个术语先来了解一下：

- dims: 要计算的通道数，对于灰度图dims=1，普通彩色图dims=3
- range: 要计算的像素值范围，一般为[0,256)
- bins: 子区段数目，如果我们统计0`~`255每个像素值，bins=256；如果划分区间，比如0`~`15, 16`~`31…240`~`255这样16个区间，bins=16

### 计算直方图

OpenCV和Numpy中都提供了计算直方图的函数，我们对比下它们的性能。

#### OpenCV中直方图计算

使用`cv2.calcHist(images, channels, mask, histSize, ranges)`计算，其中：

- 参数1：要计算的原图，以方括号的传入，如：[img]
- 参数2：类似前面提到的dims，**灰度图写[0]就行，彩色图B/G/R分别传入[0]/[1]/[2]**
- 参数3（mask）：要计算的区域，计算整幅图的话，写None
- 参数4：前面提到的bins
- 参数5：前面提到的range

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('hist.jpg', 0)
hist = cv2.calcHist([img], [0], None, [256], [0, 256])  # 性能：0.025288 s
```

### 计算部分图像直方图

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt
img=cv2.imread('hist.jpg',0)  # (1024,683)
mask=np.zeros(img.shape,dtype=np.uint8)
mask[0:200,0:200]=255
hist=cv2.calcHist([img],[0],mask,[256],[0,256])
plt.plot(hist)
plt.show()

```

<img src="D:\typora\src\image-20211026210142552.png" alt="image-20211026210142552" style="zoom:50%;" /><img src="D:\typora\src\image-20211026210157181.png" alt="image-20211026210157181" style="zoom:50%;" />

#### Numpy中直方图计算

也可用Numpy的函数计算，其中[ravel()](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ravel.html)函数将二维矩阵展平变成一维数组，之前有提到过：

```python
hist, bins = np.histogram(img.ravel(), 256, [0, 256])  # 性能：0.020628 s
```

> 经验之谈：Numpy中还有一种更高效的方式：（还记得怎么评估性能吗：代码性能优化]

```python
hist = np.bincount(img.ravel(), minlength=256)  # 性能：0.003163 s
```

计算出直方图之后，怎么把它画出来呢？

### 绘制直方图

其实Matplotlib自带了一个计算并绘制直方图的功能，不需要用到上面的函数：

```python
plt.hist(img.ravel(), 256, [0, 256])
plt.show()
```

当然，也可以用前面计算出来的结果绘制：

```python
plt.plot(hist)
plt.show()
```

![img](http://cos.codec.wang/cv2_calc_draw_histogram.jpg)

从直方图上可以看到图片的大部分区域集中在150偏白的附近，这其实并不是很好的效果，下面我们来看看如何改善它。

> 使用OpenCV的画线功能也可以画直方图，不过太麻烦了

### 直方图均衡化

一副效果好的图像通常在直方图上的分布比较均匀，直方图均衡化就是用来改善图像的全局亮度和对比度。其实从观感上就可以发现，前面那幅图对比度不高，偏灰白。对均衡化算法感兴趣的同学可参考：[维基百科：直方图均衡化](https://zh.wikipedia.org/wiki/直方图均衡化)

![img](http://cos.codec.wang/cv2_understand_histogram_equalization.jpg)

```python
equ = cv2.equalizeHist(img)
```

OpenCV中用`cv2.equalizeHist()`实现均衡化。我们把两张图片并排显示，对比一下：

```python
cv2.imshow('equalization', np.hstack((img, equ)))  # 并排显示
cv2.waitKey(0)
```

![img](http://cos.codec.wang/cv2_before_after_equalization.jpg)

![均衡化前后的直方图对比](http://cos.codec.wang/cv2_before_after_equalization_histogram.jpg)

可以看到均衡化后图片的亮度和对比度效果明显好于原图。

### 自适应均衡化

不难看出来，直方图均衡化是应用于整幅图片的，会有什么问题呢？看下图：

![img](http://cos.codec.wang/cv2_understand_adaptive_histogram.jpg)

很明显，因为全局调整亮度和对比度的原因，脸部太亮，大部分细节都丢失了。

自适应均衡化就是用来解决这一问题的：它在每一个小区域内（默认8×8）进行直方图均衡化。当然，如果有噪点的话，噪点会被放大，需要对小区域内的对比度进行了限制，所以这个算法全称叫：**对比度受限的自适应直方图均衡化**CLAHE([Contrast Limited Adaptive Histogram Equalization](https://en.wikipedia.org/wiki/Adaptive_histogram_equalization))。

```python
# 自适应均衡化，参数可选
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cl1 = clahe.apply(img)
```

![img](http://cos.codec.wang/cv2_adaptive_histogram.jpg)

## 模板匹配

[模板匹配]: https://baike.baidu.com/item/%E6%A8%A1%E6%9D%BF%E5%8C%B9%E9%85%8D	"模板匹配"

用来在大图中找小图，也就是说在一副图像中寻找另外一张模板图像的位置

<img src="http://cos.codec.wang/cv2_understand_template_matching.jpg" alt="img" style="zoom:50%;" />

用`cv2.matchTemplate()`实现模板匹配。首先我们来读入图片和模板：

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('lena.jpg', 0)
template = cv2.imread('face.jpg', 0)
h, w = template.shape[:2]  # rows->h, cols->
```

匹配函数返回的是一副灰度图，最白的地方表示最大的匹配。使用`cv2.minMaxLoc()`函数可以得到最大匹配值的坐标，以这个点为左上角角点，模板的宽和高画矩形就是匹配的位置了：

```python
# 相关系数匹配方法：cv2.TM_CCOEFF
res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

left_top = max_loc  # 左上角
right_bottom = (left_top[0] + w, left_top[1] + h)  # 右下角
cv2.rectangle(img, left_top, right_bottom, 255, 2)  # 画出矩形位置
```

![img](http://cos.codec.wang/cv2_ccoeff_matching_template.jpg)

### 原理

> 这部分可看可不看，不太理解也没关系，还记得前面的方法吗？不懂得就划掉(✿◕‿◕✿)

模板匹配的原理其实很简单，就是不断地在原图中移动模板图像去比较，有6种不同的比较方法，详情可参考：[TemplateMatchModes](https://docs.opencv.org/3.3.1/df/dfb/group__imgproc__object.html#ga3a7850640f1fe1f58fe91a2d7583695d)

- 平方差匹配CV_TM_SQDIFF：用两者的平方差来匹配，最好的匹配值为0
- 归一化平方差匹配CV_TM_SQDIFF_NORMED
- 相关匹配CV_TM_CCORR：用两者的乘积匹配，数值越大表明匹配程度越好
- 归一化相关匹配CV_TM_CCORR_NORMED
- 相关系数匹配CV_TM_CCOEFF：用两者的相关系数匹配，1表示完美的匹配，-1表示最差的匹配
- 归一化相关系数匹配CV_TM_CCOEFF_NORMED

归一化的意思就是将值统一到0~1，这些方法的对比代码可到[源码处](http://codec.wang/#/)查看。模板匹配也是应用卷积来实现的：假设原图大小为W×H，模板图大小为w×h，那么生成图大小是(W-w+1)×(H-h+1)，生成图中的每个像素值表示原图与模板的匹配程度。

### 匹配多个物体

前面我们是找最大匹配的点，所以只能匹配一次。我们可以设定一个匹配阈值来匹配多次：

```python
# 1.读入原图和模板
img_rgb = cv2.imread('mario.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('mario_coin.jpg', 0)
h, w = template.shape[:2]

# 2.标准相关模板匹配
res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.8 

# 3.这边是Python/Numpy的知识，后面解释
loc = np.where(res >= threshold)  # 匹配程度大于%80的坐标y,x
for pt in zip(*loc[::-1]):  # *号表示可选参数
    right_bottom = (pt[0] + w, pt[1] + h)
    cv2.rectangle(img_rgb, pt, right_bottom, (0, 0, 255), 2)
```

![img](http://cos.codec.wang/cv2_template_matching_multi.jpg)

第3步有几个Python/Numpy的重要知识，来大致看下：

- [np.where()](https://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html)在这里返回res中值大于0.8的所有坐标，如：

```python
x = np.arange(9.).reshape(3, 3)
print(np.where(x > 5))
# 结果(先y坐标，再x坐标)：(array([2, 2, 2]), array([0, 1, 2]))
```

![img](http://cos.codec.wang/cv2_np_where_function.jpg)

- [zip()](https://docs.python.org/3/library/functions.html#zip)函数，功能强大到难以解释，举个简单例子就知道了：

```python
x = [1, 2, 3]
y = [4, 5, 6]
print(list(zip(x, y)))  # [(1, 4), (2, 5), (3, 6)]
```

这样大家就能理解前面代码的用法了吧：因为loc是先y坐标再x坐标，所以用loc[::-1]翻转一下，然后再用zip函数拼接在一起。





## 霍夫变换

![img](http://cos.codec.wang/cv2_understand_hough_transform.jpg)

学习使用霍夫变换识别出图像中的直线和圆。图片等可到文末引用处下载。

#### 目标

- 理解霍夫变换的实现
- 分别使用霍夫线变换和圆变换检测图像中的直线和圆
- OpenCV函数：`cv2.HoughLines()`, `cv2.HoughLinesP()`, `cv2.HoughCircles()`

### 理解霍夫变换

霍夫变换常用来在图像中提取直线和圆等几何形状，我来做个简易的解释：

![img](http://cos.codec.wang/cv2_understand_hough_transform.jpg)

学过几何的都知道，直线可以分别用直角坐标系和极坐标系来表示：

![img](http://cos.codec.wang/cv2_line_expression_in_coordinate.jpg)

那么经过某个点(x0,y0)的所有直线都可以用这个式子来表示：

r_\theta=x_0\cdot\cos \theta+y_0\cdot\sin \thetarθ=x0⋅cosθ+y0⋅sinθ

也就是说每一个(r,θ)都表示一条经过(x0,y0)直线，那么同一条直线上的点必然会有同样的(r,θ)。如果将某个点所有的(r,θ)绘制成下面的曲线，那么同一条直线上的点的(r,θ)曲线会相交于一点：

![img](http://cos.codec.wang/cv2_curve_of_r_theta.jpg)

OpenCV中首先计算(r,θ) 累加数，累加数超过一定值后就认为在同一直线上。

### 霍夫直线变换

OpenCV中用`cv2.HoughLines()`在二值图上实现霍夫变换，函数返回的是一组直线的(r,θ)数据：

```python
import cv2
import numpy as np

# 1.加载图片，转为二值图
img = cv2.imread('shapes.jpg')
drawing = np.zeros(img.shape[:], dtype=np.uint8)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)

# 2.霍夫直线变换
lines = cv2.HoughLines(edges, 0.8, np.pi / 180, 90)
```

函数中：

- 参数1：要检测的二值图（一般是阈值分割或边缘检测后的图）
- 参数2：距离r的精度，值越大，考虑越多的线
- 参数3：角度θ的精度，值越小，考虑越多的线
- 参数4：累加数阈值，值越小，考虑越多的线

```python
# 3.将检测的线画出来（注意是极坐标噢）
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))

    cv2.line(drawing, (x1, y1), (x2, y2), (0, 0, 255))
```

![img](http://cos.codec.wang/cv2_hough_line_function.jpg)

### 统计概率霍夫直线变换

前面的方法又称为标准霍夫变换，它会计算图像中的每一个点，计算量比较大，另外它得到的是整一条线（r和θ），并不知道原图中直线的端点。所以提出了统计概率霍夫直线变换(Probabilistic Hough Transform)，是一种改进的霍夫变换：

```python
drawing = np.zeros(img.shape[:], dtype=np.uint8)
# 3.统计概率霍夫线变换
lines = cv2.HoughLinesP(edges, 0.8, np.pi / 180, 90,
                        minLineLength=50, maxLineGap=10)
```

前面几个参数跟之前的一样，有两个可选参数：

- `minLineLength`：最短长度阈值，比这个长度短的线会被排除
- `maxLineGap`：同一直线两点之间的最大距离

```python
# 3.将检测的线画出来
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(drawing, (x1, y1), (x2, y2), (0, 255, 0), 1, lineType=cv2.LINE_AA)
```

`cv2.LINE_AA`在之前绘图功能中讲解过，表示抗锯齿线型。

![img](http://cos.codec.wang/cv2_hough_lines_p_function.jpg)

### 霍夫圆变换

霍夫圆变换跟直线变换类似，只不过线是用(r,θ)表示，圆是用(x_center,y_center,r)来表示，从二维变成了三维，数据量变大了很多；所以一般使用霍夫梯度法减少计算量，对该算法感兴趣的同学可参考：[Circle Hough Transform](https://en.wikipedia.org/wiki/Circle_Hough_Transform)

```python
drawing = np.zeros(img.shape[:], dtype=np.uint8)
# 2.霍夫圆变换
circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 20, param2=30)
circles = np.int0(np.around(circles))
```



## 



# 接口：

