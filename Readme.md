# 风格文案生成系统Demo运行说明

## 环境说明
Django框架+HTML

数据库: sqlite

运行前：

保证Django环境 + sqlite3的驱动已安装

## 运行
- 初始化数据库
```python
python3 manage.py migrate  
```

- 运行
```python
python3 manage.py runserver
```
- 系统启动后，访问网页：
http://127.0.0.1:8000/image_caption_system/