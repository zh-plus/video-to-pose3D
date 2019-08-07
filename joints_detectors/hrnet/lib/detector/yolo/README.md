## use for image human bounding box obtain

 `python video_demo.py --video /home/xyliu/Videos/sports/dance.mp4 `

 产生一个文件夹 `/home/xyliu/Videos/sports/dance_images`
 里面包含dance.mp4的每一帧图片，和一个detection.json(文件名和人体bbox)

 `python detect.py --images person --det det`

 会处理person文件夹里面的所有image, 并生成包含文件名、人体bbox的json文件
