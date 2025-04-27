### Руководство по загрузке моделей из зоопарка моделей OpenVINO ###

**Пример на модели "efficientdet-d1-tf"**

- Шаг 1. Загружаем веса модели.

- Шаг 2. Генерируем файл архитектуры:

	- Для генерации соответствующего файла *.pbtxt будем использовать скрипт tf_text_graph_efficientdet.py:
`python tf_text_graph_efficientdet.py --input efficientdet-d1_frozen.pb --config pipeline.config --output efficientdet-d1.pbtxt`


[Ссылка на скрипт](https://github.com/opencv/opencv/blob/4.x/samples/dnn/tf_text_graph_efficientdet.py) 

[Ссылка на зоопарк моделей OpenVINO]( https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public)
