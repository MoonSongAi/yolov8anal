## Miniconda3 환경에서 가상환경 만든다
- terminal bash 명령어
conda create --name yolov8
conda activate yolov8

## Git init and remote connection
# PC에 git-scm.com에서 Git Download & install
>>> git init  
>>> git remote add origin https://github.com/MoonSongAi/yolov8anal.git
......
GUI> enter "message...."  and  click "commit button"
....
>>> git push -u origin master 


## yolov8 과 PyQT를 설치한다
- terminal bash 명령어
pip install PyQt6 ultralytics

#######################################################
    (yolov8) C:\Users\user\Desktop\Company\Yolo\yoloV8>pip install PyQt6 ultralytics
    Collecting PyQt6
    Downloading PyQt6-6.6.1-cp38-abi3-win_amd64.whl.metadata (2.2 kB)
    Collecting ultralytics
    Downloading ultralytics-8.1.24-py3-none-any.whl.metadata (40 kB)
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 40.4/40.4 kB 654.8 kB/s eta 0:00:00
    Collecting PyQt6-sip<14,>=13.6 (from PyQt6)
    Downloading PyQt6_sip-13.6.0-cp312-cp312-win_amd64.whl.metadata (524 bytes)
    Collecting PyQt6-Qt6>=6.6.0 (from PyQt6)
    Downloading PyQt6_Qt6-6.6.2-py3-none-win_amd64.whl.metadata (551 bytes)
    Collecting matplotlib>=3.3.0 (from ultralytics)
    Downloading matplotlib-3.8.3-cp312-cp312-win_amd64.whl.metadata (5.9 kB)
    Collecting opencv-python>=4.6.0 (from ultralytics)
    Using cached opencv_python-4.9.0.80-cp37-abi3-win_amd64.whl.metadata (20 kB)
    Collecting pillow>=7.1.2 (from ultralytics)
    Downloading pillow-10.2.0-cp312-cp312-win_amd64.whl.metadata (9.9 kB)
    Collecting pyyaml>=5.3.1 (from ultralytics)
    Using cached PyYAML-6.0.1-cp312-cp312-win_amd64.whl.metadata (2.1 kB)
    Collecting requests>=2.23.0 (from ultralytics)
    Using cached requests-2.31.0-py3-none-any.whl.metadata (4.6 kB)
    Collecting scipy>=1.4.1 (from ultralytics)
    Downloading scipy-1.12.0-cp312-cp312-win_amd64.whl.metadata (60 kB)
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 60.4/60.4 kB 3.1 MB/s eta 0:00:00
    Collecting torch>=1.8.0 (from ultralytics)
    Downloading torch-2.2.1-cp312-cp312-win_amd64.whl.metadata (26 kB)
    Collecting torchvision>=0.9.0 (from ultralytics)
    Downloading torchvision-0.17.1-cp312-cp312-win_amd64.whl.metadata (6.6 kB)
    Collecting tqdm>=4.64.0 (from ultralytics)
    Using cached tqdm-4.66.2-py3-none-any.whl.metadata (57 kB)
    Collecting psutil (from ultralytics)
    Using cached psutil-5.9.8-cp37-abi3-win_amd64.whl.metadata (22 kB)
    Collecting py-cpuinfo (from ultralytics)
    Using cached py_cpuinfo-9.0.0-py3-none-any.whl.metadata (794 bytes)
    Collecting thop>=0.1.1 (from ultralytics)
    Using cached thop-0.1.1.post2209072238-py3-none-any.whl.metadata (2.7 kB)
    Collecting pandas>=1.1.4 (from ultralytics)
    Downloading pandas-2.2.1-cp312-cp312-win_amd64.whl.metadata (19 kB)
    Collecting seaborn>=0.11.0 (from ultralytics)
    Using cached seaborn-0.13.2-py3-none-any.whl.metadata (5.4 kB)
    Collecting contourpy>=1.0.1 (from matplotlib>=3.3.0->ultralytics)
    Using cached contourpy-1.2.0-cp312-cp312-win_amd64.whl.metadata (5.8 kB)
    Collecting cycler>=0.10 (from matplotlib>=3.3.0->ultralytics)
    Using cached cycler-0.12.1-py3-none-any.whl.metadata (3.8 kB)
    Collecting fonttools>=4.22.0 (from matplotlib>=3.3.0->ultralytics)
    Downloading fonttools-4.49.0-cp312-cp312-win_amd64.whl.metadata (162 kB)
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 162.3/162.3 kB 9.5 MB/s eta 0:00:00
    Collecting kiwisolver>=1.3.1 (from matplotlib>=3.3.0->ultralytics)
    Using cached kiwisolver-1.4.5-cp312-cp312-win_amd64.whl.metadata (6.5 kB)
    Collecting numpy<2,>=1.21 (from matplotlib>=3.3.0->ultralytics)
    Downloading numpy-1.26.4-cp312-cp312-win_amd64.whl.metadata (61 kB)
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 61.0/61.0 kB ? eta 0:00:00
    Collecting packaging>=20.0 (from matplotlib>=3.3.0->ultralytics)
    Using cached packaging-23.2-py3-none-any.whl.metadata (3.2 kB)
    Collecting pyparsing>=2.3.1 (from matplotlib>=3.3.0->ultralytics)
    Using cached pyparsing-3.1.1-py3-none-any.whl.metadata (5.1 kB)
    Collecting python-dateutil>=2.7 (from matplotlib>=3.3.0->ultralytics)
    Downloading python_dateutil-2.9.0.post0-py2.py3-none-any.whl.metadata (8.4 kB)
    Collecting pytz>=2020.1 (from pandas>=1.1.4->ultralytics)
    Using cached pytz-2024.1-py2.py3-none-any.whl.metadata (22 kB)
    Collecting tzdata>=2022.7 (from pandas>=1.1.4->ultralytics)
    Using cached tzdata-2024.1-py2.py3-none-any.whl.metadata (1.4 kB)
    Collecting charset-normalizer<4,>=2 (from requests>=2.23.0->ultralytics)
    Using cached charset_normalizer-3.3.2-cp312-cp312-win_amd64.whl.metadata (34 kB)
    Collecting idna<4,>=2.5 (from requests>=2.23.0->ultralytics)
    Using cached idna-3.6-py3-none-any.whl.metadata (9.9 kB)
    Collecting urllib3<3,>=1.21.1 (from requests>=2.23.0->ultralytics)
    Using cached urllib3-2.2.1-py3-none-any.whl.metadata (6.4 kB)
    Collecting certifi>=2017.4.17 (from requests>=2.23.0->ultralytics)
    Using cached certifi-2024.2.2-py3-none-any.whl.metadata (2.2 kB)
    Collecting filelock (from torch>=1.8.0->ultralytics)
    Using cached filelock-3.13.1-py3-none-any.whl.metadata (2.8 kB)
    Collecting typing-extensions>=4.8.0 (from torch>=1.8.0->ultralytics)
    Using cached typing_extensions-4.10.0-py3-none-any.whl.metadata (3.0 kB)
    Collecting sympy (from torch>=1.8.0->ultralytics)
    Using cached sympy-1.12-py3-none-any.whl.metadata (12 kB)
    Collecting networkx (from torch>=1.8.0->ultralytics)
    Using cached networkx-3.2.1-py3-none-any.whl.metadata (5.2 kB)
    Collecting jinja2 (from torch>=1.8.0->ultralytics)
    Using cached Jinja2-3.1.3-py3-none-any.whl.metadata (3.3 kB)
    Collecting fsspec (from torch>=1.8.0->ultralytics)
    Using cached fsspec-2024.2.0-py3-none-any.whl.metadata (6.8 kB)
    Collecting colorama (from tqdm>=4.64.0->ultralytics)
    Using cached colorama-0.4.6-py2.py3-none-any.whl.metadata (17 kB)
    Collecting six>=1.5 (from python-dateutil>=2.7->matplotlib>=3.3.0->ultralytics)
    Using cached six-1.16.0-py2.py3-none-any.whl.metadata (1.8 kB)
    Collecting MarkupSafe>=2.0 (from jinja2->torch>=1.8.0->ultralytics)
    Downloading MarkupSafe-2.1.5-cp312-cp312-win_amd64.whl.metadata (3.1 kB)
    Collecting mpmath>=0.19 (from sympy->torch>=1.8.0->ultralytics)
    Using cached mpmath-1.3.0-py3-none-any.whl.metadata (8.6 kB)
    Downloading PyQt6-6.6.1-cp38-abi3-win_amd64.whl (6.5 MB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.5/6.5 MB 16.7 MB/s eta 0:00:00
    Downloading ultralytics-8.1.24-py3-none-any.whl (719 kB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 719.5/719.5 kB 22.9 MB/s eta 0:00:00
    Downloading matplotlib-3.8.3-cp312-cp312-win_amd64.whl (7.6 MB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7.6/7.6 MB 32.6 MB/s eta 0:00:00
    Using cached opencv_python-4.9.0.80-cp37-abi3-win_amd64.whl (38.6 MB)
    Downloading pandas-2.2.1-cp312-cp312-win_amd64.whl (11.5 MB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 11.5/11.5 MB 29.8 MB/s eta 0:00:00
    Downloading pillow-10.2.0-cp312-cp312-win_amd64.whl (2.6 MB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.6/2.6 MB 24.0 MB/s eta 0:00:00
    Downloading PyQt6_Qt6-6.6.2-py3-none-win_amd64.whl (62.4 MB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 62.4/62.4 MB 28.4 MB/s eta 0:00:00
    Downloading PyQt6_sip-13.6.0-cp312-cp312-win_amd64.whl (73 kB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 73.1/73.1 kB 3.9 MB/s eta 0:00:00
    Using cached PyYAML-6.0.1-cp312-cp312-win_amd64.whl (138 kB)
    Using cached requests-2.31.0-py3-none-any.whl (62 kB)
    Downloading scipy-1.12.0-cp312-cp312-win_amd64.whl (45.8 MB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 45.8/45.8 MB 18.7 MB/s eta 0:00:00
    Using cached seaborn-0.13.2-py3-none-any.whl (294 kB)
    Using cached thop-0.1.1.post2209072238-py3-none-any.whl (15 kB)
    Downloading torch-2.2.1-cp312-cp312-win_amd64.whl (198.5 MB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 198.5/198.5 MB 13.9 MB/s eta 0:00:00
    Downloading torchvision-0.17.1-cp312-cp312-win_amd64.whl (1.2 MB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.2/1.2 MB 18.6 MB/s eta 0:00:00
    Using cached tqdm-4.66.2-py3-none-any.whl (78 kB)
    Using cached psutil-5.9.8-cp37-abi3-win_amd64.whl (255 kB)
    Using cached py_cpuinfo-9.0.0-py3-none-any.whl (22 kB)
    Using cached certifi-2024.2.2-py3-none-any.whl (163 kB)
    Using cached charset_normalizer-3.3.2-cp312-cp312-win_amd64.whl (100 kB)
    Using cached contourpy-1.2.0-cp312-cp312-win_amd64.whl (187 kB)
    Using cached cycler-0.12.1-py3-none-any.whl (8.3 kB)
    Downloading fonttools-4.49.0-cp312-cp312-win_amd64.whl (2.2 MB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.2/2.2 MB 19.6 MB/s eta 0:00:00
    Using cached idna-3.6-py3-none-any.whl (61 kB)
    Using cached kiwisolver-1.4.5-cp312-cp312-win_amd64.whl (56 kB)
    Downloading numpy-1.26.4-cp312-cp312-win_amd64.whl (15.5 MB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 15.5/15.5 MB 14.2 MB/s eta 0:00:00
    Using cached packaging-23.2-py3-none-any.whl (53 kB)
    Using cached pyparsing-3.1.1-py3-none-any.whl (103 kB)
    Downloading python_dateutil-2.9.0.post0-py2.py3-none-any.whl (229 kB)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 229.9/229.9 kB 7.1 MB/s eta 0:00:00
    Using cached pytz-2024.1-py2.py3-none-any.whl (505 kB)
    Using cached typing_extensions-4.10.0-py3-none-any.whl (33 kB)
    Using cached tzdata-2024.1-py2.py3-none-any.whl (345 kB)
    Using cached urllib3-2.2.1-py3-none-any.whl (121 kB)
    Using cached colorama-0.4.6-py2.py3-none-any.whl (25 kB)
    Using cached filelock-3.13.1-py3-none-any.whl (11 kB)
    Using cached fsspec-2024.2.0-py3-none-any.whl (170 kB)
    Using cached Jinja2-3.1.3-py3-none-any.whl (133 kB)
    Using cached networkx-3.2.1-py3-none-any.whl (1.6 MB)
    Using cached sympy-1.12-py3-none-any.whl (5.7 MB)
    Downloading MarkupSafe-2.1.5-cp312-cp312-win_amd64.whl (17 kB)
    Using cached mpmath-1.3.0-py3-none-any.whl (536 kB)
    Using cached six-1.16.0-py2.py3-none-any.whl (11 kB)
    Installing collected packages: pytz, PyQt6-Qt6, py-cpuinfo, mpmath, urllib3, tzdata, typing-extensions, sympy, six, pyyaml, PyQt6-sip, pyparsing, psutil, pillow, packaging, numpy, networkx, MarkupSafe, kiwisolver, idna, fsspec, fonttools, filelock, cycler, colorama, charset-normalizer, certifi, tqdm, scipy, requests, python-dateutil, PyQt6, opencv-python, jinja2, contourpy, torch, pandas, matplotlib, torchvision, thop, seaborn, ultralytics
    Successfully installed MarkupSafe-2.1.5 PyQt6-6.6.1 PyQt6-Qt6-6.6.2 PyQt6-sip-13.6.0 certifi-2024.2.2 charset-normalizer-3.3.2 colorama-0.4.6 contourpy-1.2.0 cycler-0.12.1 filelock-3.13.1 fonttools-4.49.0 fsspec-2024.2.0 idna-3.6 jinja2-3.1.3 kiwisolver-1.4.5 matplotlib-3.8.3 mpmath-1.3.0 networkx-3.2.1 numpy-1.26.4 opencv-python-4.9.0.80 packaging-23.2 pandas-2.2.1 pillow-10.2.0 psutil-5.9.8 py-cpuinfo-9.0.0 pyparsing-3.1.1 python-dateutil-2.9.0.post0 pytz-2024.1 pyyaml-6.0.1 requests-2.31.0 scipy-1.12.0 seaborn-0.13.2 six-1.16.0 sympy-1.12 thop-0.1.1.post2209072238 torch-2.2.1 torchvision-0.17.1 tqdm-4.66.2 typing-extensions-4.10.0 tzdata-2024.1 ultralytics-8.1.24 urllib3-2.2.1

    (yolov8) C:\Users\user\Desktop\Company\Yolo\yoloV8>
#######################################################
    C:\Users\user\Desktop\Company\Yolo\yoloV8
    │
    ├───Bin
    │   │   main.py          # 메인 실행 Python 스크립트
    │   └───Classes         # 사용할 Python 클래스 모음
    │       │   model.py    # YOLO 모델을 다루는 클래스
    │       └───utils.py    # 유틸리티 함수를 담은 클래스
    │
    ├───Source              # 분석할 이미지 파일을 담는 폴더
    │
    ├───Result              # 분석되어 라벨링된 이미지를 저장하는 폴더
    │
    ├───Models              # YOLO 모델 파일 및 설정 파일을 저장하는 폴더
    │
    └───UI                  # PyQt를 통해 생성된 사용자 인터페이스 관련 파일
        │   mainwindow.ui   # PyQt 디자이너로 생성된 메인 윈도우 UI 파일
        └───resources.qrc   # UI 리소스 파일

# Object Detection & Image Segmentation using YOLO-NAS + SAM
Windows에서는 공식 프로토콜 버퍼 GitHub 릴리스 페이지(https://github.com/protocolbuffers/protobuf/releases)에서 
미리 빌드된 바이너리를 다운로드하여 사용할 수 있습니다. 
다운로드한 후에는 압축을 해제하고 해당 바이너리의 경로를 시스템 PATH에 추가해야 합니다.
pip install protobuf
pip install ultralytics install super-gradients 
pip install git+https://github.com/facebookresearch/segment-anything.git -q
# !wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

