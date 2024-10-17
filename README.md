# video_QA
Use OpenAI gpt-4o to classify cattle behaviour in video clips.

## Installation
create a conda environment and install requirements
```
conda create -n videoQA
conda activate videoQA
pip install -r requirements.txt
```

## Running the code
- Add the videos to a folder.
- create a csv file with following fields: `FileName`, `FilePath`, `ClassLabel`. The ClassLabel column can be empty.
  Supported classes: `Walking`, `Sitting/Sleeping`, `Standing`, `HeadButting`
- run the code a below
  ```
  python3 video_labelling_demo.py --input_csv='../Data4QA/input.csv'
  ```