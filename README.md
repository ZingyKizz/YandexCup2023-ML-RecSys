# YandexCup2023-ML-RecSys

## Шаги для запуска решения
* Склонировать репозиторий
  ~~~
  git clone https://github.com/ZingyKizz/YandexCup2023-ML-RecSys.git
  ~~~
* Перейти в директорию проекта
  ~~~
  cd YandexCup2023-ML-RecSys
  ~~~
* Установить зависимости
  ~~~
  pip3 install -r requirements.txt
  ~~~
* В директории configs выбрать, например, '42.yaml'. Отредактировать внутри ручками директорию data_path. Подразумевается, что в указанной директории лежат папки data и track_embeddings
~~~
.
├── data
│   ├── Baseline.ipynb
│   ├── sample_submission.csv
│   ├── test.csv
│   └── train.csv
└── track_embeddings
    ├── 1.npy
    ├── 2.npy
    ├── 3.npy
    └── 4.npy
~~~
* Из корня проекта запустить команду
  ~~~
  python3 -m main --cfg_path=configs/42.yaml
  ~~~

Итоговое решение - бленд множества моделей, обученных по конфигам из директории configs, процесс блендинга можно найти в notebooks/Submissions.ipynb. Но полностью воспроизвести вряд ли получится :)
