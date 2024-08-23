# Dataset Preparation

## EGTEA Gaze+
1. The EGTEA Gaze+ Dataset could be downloaded via the [official website](https://cbs.ic.gatech.edu/fpv/).

2. After all the videos and gaze labels are downloaded, reorganize the video clips in this structure:

```
egtea
|_ cropped_clips
|  |_ OP01-R01-PastaSalad
|  |  |_ OP01-R01-PastaSalad-1002316-1004005-F024051-F024101.mp4
|  |  |_ OP01-R01-PastaSalad-1004110-1021110-F024057-F024548.mp4
|  |  |_ ...
|  |_ OP01-R02-TurkeySandwich
|  |  |_ OP01-R02-TurkeySandwich-102320-105110-F002449-F002529.mp4
|  |  |_ OP01-R02-TurkeySandwich-105440-106460-F002528-F002558.mp4
|  |  |_ ...
|  |_ ...
|
|_ gaze_data
   |_ OP01-R01-PastaSalad.txt
   |_ OP01-R02-TurkeySandwich.txt
   |_ OP01-R03-BaconAndEggs.txt
   |_ ...
```

## Ego4D

1. The Ego4D dataset could be downloaded following the [official instructions](https://ego4d-data.org/docs/start-here/).

2. We only need to download the videos with gaze annotations. The labels and video ids can be downloaded [here](https://ego4d-data.org/docs/data/gaze/).

3. Gaze annotations are organized in a bunch of csv files. Each file corresponds to a video. Unfortunately, Ego4D hasn't provided a command to download all of these videos yet. You need to download videos via the video ids (i.e. the name of each csv file) using the [CLI tool](https://ego4d-data.org/docs/CLI/) and `--video_uids`.

4. Please reorganize the video clips and annotations in this structure:

```
Ego4D
|_ full_scale.gaze
|  |_ 0d271871-c8ba-4249-9434-d39ce0060e58.mp4
|  |_ 1e83c2d1-ff03-4181-9ab5-a3e396f54a93.mp4
|  |_ 2bb31b69-fcda-4f54-8338-f590944df999.mp4
|  |_ ...
|
|_ gaze
   |_ 0d271871-c8ba-4249-9434-d39ce0060e58.csv
   |_ 1e83c2d1-ff03-4181-9ab5-a3e396f54a93.csv
   |_ 2bb31b69-fcda-4f54-8338-f590944df999.csv
   |_ ...
```

5. Enter `data` directory and replace the path of Ego4D dataset in `preprocessing.py` to the path on your computer. Then run `preprocessing.py` to get video clips and gaze annotation in each frame. They are saved in the folders `clips.gaze` and `gaze_frame_label`.
