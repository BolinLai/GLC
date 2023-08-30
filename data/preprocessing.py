import csv
import numpy as np
import av


from moviepy.editor import *
from tqdm import tqdm


def clip_videos(source_path, save_path, untrack_csv):
    """
    Clip long videos into clips.

    :param source_path: Long video path
    :param save_path:
    :param untrack_csv: Used to remove untracked frames.
    :return: None
    """
    os.makedirs(save_path, exist_ok=True)

    with open(untrack_csv, 'r') as f:
        lines = [item for item in csv.reader(f)]
    untracked = dict()
    for line in lines:
        start_hr, start_min, start_sec = line[1].split(':')
        end_hr, end_min, end_sec = line[2].split(':')
        start = int(start_hr) * 3600 + int(start_min) * 60 + int(start_sec)
        end = int(end_hr) * 3600 + int(end_min) * 60 + int(end_sec)
        if line[0] in untracked.keys():
            untracked[line[0]].append([start, end, int(line[-1])])
        else:
            untracked[line[0]] = [[start, end, int(line[-1])]]

    for item in tqdm(sorted(os.listdir(source_path))):
        if item in ['4e07da0c-450f-4c37-95e9-e793cb5d8f7f.mp4',
                    '5819e52c-4e12-4f86-ad69-76fc215dfbcb.mp4',
                    '83081c5a-8456-44d8-af67-280034f8f0a6.mp4',
                    'a77682da-cae7-4e68-8580-6cb47658b23f.mp4']:
            continue

        if os.path.splitext(item)[-1] == '.mp4':
            # loading video gfg
            video = VideoFileClip(os.path.join(source_path, item))
            duration = video.duration
            fps = video.fps

            vid = os.path.splitext(item)[0]
            os.makedirs(os.path.join(save_path, vid), exist_ok=True)

            for i in tqdm(range(0, int(duration), 5), leave=False):
                start, end = i, i + 5
                if end > duration:
                    break
                if os.path.splitext(item)[0] in untracked.keys():
                    skip = False
                    for interval in untracked[vid]:
                        if not (end < interval[0] or start > interval[1]):
                            skip = True
                            break
                    if skip:
                        continue

                clip = video.subclip(start, end)
                clip.write_videofile(os.path.join(save_path, vid, f'{vid}_t{start}_t{end}.mp4'))


def get_ego4d_frame_label(data_path, save_path):
    all_frames_num = 0
    all_saccade_num = 0
    all_trimmed_num = 0
    all_untracked_num = 0
    os.makedirs(save_path, exist_ok=True)
    for ann_file in os.listdir(os.path.join(data_path, 'gaze')):
        if ann_file == 'manifest.csv' or ann_file == 'manifest.ver':
            continue
        vid = ann_file.split('.')[0]
        with open(os.path.join(data_path, 'gaze', ann_file), 'r') as f:
            lines = [line for i, line in enumerate(csv.reader(f)) if i > 0]

        container = av.open(os.path.join(data_path, 'full_scale.gaze', f'{vid}.mp4'))
        fps = float(container.streams.video[0].average_rate)
        frames_length = container.streams.video[0].frames
        duration = container.streams.video[0].duration

        j = 0
        gaze_loc = list()
        for i in tqdm(range(frames_length), leave=False):
            time_stamp = i * 1 / fps  # find the accurate time stamp of each frame
            if j >= len(lines) - 2:
                break
            while float(lines[j][1]) < time_stamp:  # search the closest time of recorded location
                j += 1
            row = lines[j - 1] if abs(float(lines[j - 1][1]) - time_stamp) < abs(float(lines[j][1]) - time_stamp) else lines[j]
            x, y = float(row[5]), 1 - float(row[6])  # use bottom-left as origin

            if i == 0:
                gaze_type = 0
            else:
                movement = np.sqrt(((x - gaze_loc[-1][1]) * 1088) ** 2 + ((y - gaze_loc[-1][2]) * 1080) ** 2)
                gaze_type = 0 if movement <= 40 else 1  # for saccade

            if not (0 <= x <= 1 and 0 <= y <= 1):
                gaze_type = 2
                x = np.clip(x, 0, 1)
                y = np.clip(y, 0, 1)
            gaze_loc.append([i, x, y, gaze_type])

        if frames_length > len(gaze_loc):
            gaze_loc.extend([[k, 0, 0, 3] for k in range(gaze_loc[-1][0]+1, frames_length)])

        all_frames_num += len(gaze_loc)
        for item in gaze_loc:
            if item[3] == 1:
                all_saccade_num += 1
            elif item[3] == 2:
                all_trimmed_num += 1
            elif item[3] == 3:
                all_untracked_num += 1

        with open(os.path.join(save_path, f'{vid}_frame_label.csv'), 'w') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['frame', 'x', 'y', 'gaze_type'])
            csv_writer.writerows(gaze_loc)

    print('All saccade rate:', all_saccade_num / all_frames_num,
          'All trimmed rate:', all_trimmed_num / all_frames_num,
          'All untracked rate:', all_untracked_num / all_frames_num)


def main():
    path_to_ego4d = '/path/to/Ego4D'  # change this to your own path

    source_path = f'{path_to_ego4d}/full_scale.gaze'
    save_path = f'{path_to_ego4d}/clips.gaze'
    untracked_csv = f'ego4d_gaze_untracked.csv'
    clip_videos(source_path=source_path, save_path=save_path, untrack_csv=untracked_csv)

    data_path = path_to_ego4d
    save_path = f'{path_to_ego4d}/gaze_frame_label'
    get_ego4d_frame_label(data_path=data_path, save_path=save_path)


if __name__ == '__main__':
    main()


