#!/usr/bin/env python
# coding: utf-8

# ## 毕加索
# 巴勃罗·毕加索（Pablo Picasso，1881年10月25日～1973年4月8日），西班牙画家、雕塑家，法国共产党党员。是现代艺术的创始人，西方现代派绘画的主要代表。毕加索是当代西方最有创造性和影响最深远的艺术家，是20世纪最伟大的艺术天才之一。、
# ## 风格介绍
# 毕加索的艺术生涯几乎贯穿其一生，作品风格丰富多样，后人用“毕加索永远是年轻的”的说法形容毕加索多变的艺术形式。史学上不得不把他浩繁的作品分为不同的时期——早年的“蓝色时期”、“粉红色时期”、盛年的“黑人时期”、“分析和综合立体主义时期”（又称“立体主义时期”）、后来的“超现实主义时期”等等。 他于1907年创作的《亚威农少女》是第一张被认为有立体主义倾向的作品，是一幅具有里程碑意义的著名杰作。它不仅标志着毕加索个人艺术历程中的重大转折，而且也是西方现代艺术史上的一次革命性突破，引发了立体主义运动的诞生。
# ##### 以上都不是我知道的，都现买现卖抄自百度百科里的O(∩_∩)O

# # 成品展示
# ## 毕加索画风的蒙娜丽莎

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from PIL import Image
import matplotlib.pyplot as plt
plt.figure(figsize=[20,6])
plt.subplot(1,3,1)
plt.imshow(Image.open('work/MonaLisa.jpg'))
plt.axis('off')
plt.xticks([])
plt.subplot(1,3,2)
plt.imshow(Image.open('work/style.jpg'))
plt.axis('off')
plt.xticks([])
plt.subplot(1,3,3)
plt.imshow(Image.open('transfer_result/ndarray_1590844241.6393652.jpg'))
plt.axis('off')
plt.xticks([])
plt.show()


# ## 毕加索画风的BadAppleMV

# In[ ]:


from IPython.display import HTML
HTML('<iframe style="width:98%;height: 450px;" src=//player.bilibili.com/player.html?aid=710442177&bvid=BV1QQ4y1P745&cid=184801392&page=1" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"> </iframe>" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"> </iframe>" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"> </iframe>" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"> </iframe>"   webkitallowfullscreen="true" mozallowfullscreen="true" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"> </iframe>')


# ### 获取相关图片

# In[ ]:


get_ipython().system('wget -O work/style.jpg https://bkimg.cdn.bcebos.com/pic/203fb80e7bec54e7ddbcbb56b8389b504fc26a22?x-bce-process=image/watermark,g_7,image_d2F0ZXIvYmFpa2U4MA==,xp_5,yp_5')
get_ipython().system('wget -O work/MonaLisa.jpg https://bkimg.cdn.bcebos.com/pic/4610b912c8fcc3ceb33fb3d19945d688d53f2050?x-bce-process=image/resize,m_lfit,w_500,h_500,limit_1')


# ## stylepro_artistic接口介绍
# 这里使用stylepro_artistic模型其模型接口说明如下：
# ```python
# def style_transfer(self,
#                    images=None,
#                    paths=None,
#                    alpha=1,
#                    use_gpu=False,
#                    visualization=False,
#                    output_dir='transfer_result'):
# ```
# 1. images (list[dict]): ndarray 格式的图片数据。每一个元素都为一个 dict，有关键字 content, styles, weights(可选)，相应取值为：
# 1. content (numpy.ndarray): 待转换的图片，shape 为 [H, W, C]，BGR格式；
# 1. styles (list[numpy.ndarray]) : 作为底色的风格图片组成的列表，各个图片数组的shape 都是 [H, W, C]，BGR格式；
# 1. weights (list[float], optioal) : 各个 style 对应的权重。当不设置 weights 时，默认各个 style 有着相同的权重；
# 1. paths (list[str]): 图片的路径。每一个元素都为一个 dict，有关键字 content, styles, weights(可选)，相应取值为：
# 1. content (str): 待转换的图片的路径；
# 1. styles (list[str]) : 作为底色的风格图片的路径；
# 1. weights (list[float], optioal) : 各个 style 对应的权重。当不设置 weights 时，各个 style 的权重相同；
# 1. alpha (float) : 转换的强度，[0, 1] 之间，默认值为1；
# 1. use_gpu (bool): 是否使用 GPU；
# 1. visualization (bool): 是否将结果保存为图片，默认为 False;
# 1. output_dir (str): 图片的保存路径，默认设为 transfer_result 。

# In[ ]:


# 安装stylepro_artistic
get_ipython().system('pip install --upgrade paddlehub -i https://pypi.tuna.tsinghua.edu.cn/simple')
get_ipython().system('hub install stylepro_artistic')


# ## 单个图像的融合
# ### 当蒙娜丽莎遇上毕加索的画风时

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import paddlehub as hub
import cv2

stylepro_artistic = hub.Module(name="stylepro_artistic")
result = stylepro_artistic.style_transfer(
    images=[{
        'content': cv2.imread('work/MonaLisa.jpg'),
        'styles': [cv2.imread('work/style.jpg')]
    }],
    alpha=1,
    visualization=True)
    
plt.imshow(result[0]['data'])


# In[1]:


import paddlehub as hub


stylepro_artistic = hub.Module(name="stylepro_artistic")
result = stylepro_artistic.style_transfer(
    paths=[{
        'content':'work/MonaLisa.jpg',
        'styles': ['work/style3.jpg']
    }],
    visualization=False)
# print(result)


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.imshow(result[0]['data'])


# ### 显示原图，style的图片以及，合成后的图片

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from PIL import Image
import matplotlib.pyplot as plt
plt.figure(figsize=[20,6])
plt.subplot(1,3,1)
plt.imshow(Image.open('work/MonaLisa.jpg'))
plt.axis('off')
plt.xticks([])
plt.subplot(1,3,2)
plt.imshow(Image.open('work/style.jpg'))
plt.axis('off')
plt.xticks([])
plt.subplot(1,3,3)
plt.imshow(Image.open('transfer_result/ndarray_1590844241.6393652.jpg'))
plt.axis('off')
plt.xticks([])
plt.show()


# ## 更大胆的想法
# 当BadApple MV与上毕加索画风时，
# ##### BadApple是网友特喜欢玩的一个动画，经常有技术宅大佬把这些MV放到各种能亮的设备上播放

# In[ ]:


# 这里将视频的每一帧都提取出来了
# 然后根据每一帧进行处理，这里保留源码，但后面代码与此无关，可以直接运行后面的
import cv2
from tqdm import tqdm
video = cv2.VideoCapture("work/badapple.mp4")
fps = video.get(cv2.CAP_PROP_FPS)
frameCount = video.get(cv2.CAP_PROP_FRAME_COUNT)
size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("总共的帧数为：",frameCount)
success, frame = video.read() 
index = 0
for i in tqdm(range(int(frameCount)),desc='处理进度'):
    if success:
        cv2.imwrite('work/target/'+str(index)+'.jpg', frame)
    success, frame = video.read()
    index += 1
    if index == 400:
        break


# ### 将视频的每一帧与图片进行融合，最后保存为图片
# ##### 这是一个漫长的过程CPU跑了16个小时

# In[ ]:


get_ipython().run_line_magic('env', 'CUDA_VISIBLE_DEVICES=0 # 指定GPU设备')
import cv2
import paddlehub as hub
from tqdm import tqdm
stylepro_artistic = hub.Module(name="stylepro_artistic")
video = cv2.VideoCapture("work/badapple.mp4")
fps = video.get(cv2.CAP_PROP_FPS)
frameCount = video.get(cv2.CAP_PROP_FRAME_COUNT)
size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("总共的帧数为：",frameCount)
success, frame = video.read() 
file_paths = []
index = 0
for i in tqdm(range(int(frameCount))):
    if success and index > 33:
            result = stylepro_artistic.style_transfer(
                images=[{
                    'content': frame,
                    'styles': [cv2.imread('work/style.jpg')]
                }],
                visualization=True,
                use_gpu=True,
                output_dir='transvideo_result')
            file_paths.append(result[0]['save_path'])
    elif success:
        filep = 'transvideo_result/'+str(index)+'.jpg'
        cv2.imwrite(filep, frame)
        file_paths.append(filep)
    success, frame = video.read()
    index += 1
videoWriter = cv2.VideoWriter('transbadaplle.avi', cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
flag = True
for i in tqdm(file_paths):
    videoWriter.write(cv2.imread(i))
videoWriter.release()


# ### 将图片合成为视频

# In[ ]:


import os
import cv2
import datetime
from tqdm import tqdm
file_dict = {}
video = cv2.VideoCapture("work/badapple.mp4")
fps = video.get(cv2.CAP_PROP_FPS)
frameCount = video.get(cv2.CAP_PROP_FRAME_COUNT)
size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
for i in os.listdir('transvideo_result/'):
    file_dict['transvideo_result/'+i] = float(i.replace('ndarray_','').replace('.jpg',''))
file_dict = sorted(file_dict.items(),key = lambda x:x[1])
videoWriter = cv2.VideoWriter('transbadaplle.avi', cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
flag = True
for i in tqdm(file_dict):
    if flag:
        for j in range(34):
            videoWriter.write(cv2.imread('work/target/0.jpg'))
        flag = False
    videoWriter.write(cv2.imread(i[0]))
videoWriter.release()
# cv2.destroyAllWindows()


# ## 成品展示

# In[ ]:


from IPython.display import HTML
HTML('<iframe style="width:98%;height: 450px;" src=//player.bilibili.com/player.html?aid=710442177&bvid=BV1QQ4y1P745&cid=184801392&page=1" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"> </iframe>" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"> </iframe>" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"> </iframe>" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"> </iframe>"   webkitallowfullscreen="true" mozallowfullscreen="true" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"> </iframe>')


# ### 网络一键部署

# In[6]:


# hub serving start -m stylepro_artistic


# In[6]:


import requests
import json
import cv2
import base64
import paddlehub as hub
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

def cv2_to_base64(image):
    data = cv2.imencode('.jpg', image)[1]
    return base64.b64encode(data.tostring()).decode('utf8')

def base64_to_cv2(b64str):
    data = base64.b64decode(b64str.encode('utf8'))
    data = np.fromstring(data, np.uint8)
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return data

data = {'images':[
    {
        'content':cv2_to_base64(cv2.imread('work/MonaLisa.jpg')),
        'styles':[cv2_to_base64(cv2.imread('work/style.jpg'))]
    }
]}

headers = {"Content-type": "application/json"}
url = "http://127.0.0.1:8866/predict/stylepro_artistic"
r = requests.post(url=url, headers=headers, data=json.dumps(data))

plt.imshow(base64_to_cv2(r.json()["results"][0]['data']))
# print(base64_to_cv2(r.json()["results"][0]['data']))


# In[7]:


r.json()


# 请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
# Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 
