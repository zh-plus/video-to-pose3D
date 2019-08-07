`针对于issue13 video pose 原理的掌握`



### Temporal dilated conv model 具体过程

**Input**:  The architecture takes the x and y coordinates of the J joints foe each frame as input.

---


#### Processes

用多个frames (respective field size) 预测一个 3D pose。

- Input layer

   input N * J joints(x and y coordinates ,  J = 17 )


   conv layer with filter size W and outputs C features 

- B ResNet-style blocks

  Each block 

  - performs a 1D conv (filter size W and dilation factor D = $`W^B`$ ) 
  - batch normalization , Relu , Dropout 

  - performs a linear projection(conv with filter size = 1,   D = 1)

    > 作用

  - batch normalization , Relu , Dropout 

- Output layer 

   the last layer outputs a 3D pose of the input sequence using both past and future data to exploit temporal info

   > 输出的是(1, 51).

---

##### Notes:

	每个block 不使用 padding， 所以输出的channel 比 输入的 channel少，在残差连接时需要 slice. 

Slice 的 数量为 dilation mun x 2 .



In order to evaluate our method for real-time scenarios, we also experiment with causal convolutions, rather than dilated convolution.

> add --causal parameter









# 真实3D世界、相机中3D、相机中2D pixel、真实2D图片之间的转化 

## 3D ——— 2D

1. World 3D ($`X_w, Y_w, Z_w`$)

2. Camera 3D ($`X_c, Y_c, Z_c`$)

3. Camera pixel 2D ($`x, y`$) 
4. World image 2D (u, v) 



### 1 -> 2 world_to_camera

![](https://raw.githubusercontent.com/lxy5513/Markdown_image_dateset/master/Xnip2019-01-16_15-16-39.png)



### 2 -> 3 

$`{x = f * X_c/Z_C}^{}`$    

$`y = f * Y_c / Z_c`$



### 3 -> 4

$`u = \frac{x}{dx} + C_x`$ 

$`v = \frac{y}{dy} + C_Y`$ 

> dx dy 实现物理坐标系到像素坐标系之间的转化 单位为像素/米

---

相关参数
- f为镜头的焦距 单位为米
- α β 的单位为 像素/米
- dx dy 为传感器x轴和y轴上单位像素的尺寸大小，单位为像素/米
- fx fy 为x y方向的焦距，单位为像素 
   根据 f,dx,dy 得到的
- fnormalx fnormaly 为 x, y方向归一化焦距
- (cx,cy) 为主点， 图像的中心， 单位为像素。





遗留问题： 如何构建半监督生成的3Dpose 和监督生成的3Dpose之间的bone length L2loss。原理是什么？  
为什么不同数据集生成得3Dpose之间可以比较它们的bone length。