from ai_badminton.hit_detector import read_hits
from ai_badminton.video_utils import read_video_assert, get_fps, get_total_num_frames
from ai_badminton.court import Court, read_court, court_points_to_corners
from ai_badminton.pose import get_player_poses_frame, read_player_poses

import cv2
import pandas as pd

import json
from pathlib import Path

PLAYER_ID_MAPPING = {
    1: "bottom",
    2: "top"
}

def create_match_summary(match_path, output_path):

    output_path.mkdir(parents=True, exist_ok=True)

    rally_video_path = match_path / "rally_video"

    for video_path in rally_video_path.iterdir():

        rally_name = video_path.stem
        print(f"Processing rally {rally_name} in match {match_path}")

        cap = read_video_assert(video_path)
        fps = get_fps(cap)
        N_frames = get_total_num_frames(cap)

        hit_path = match_path / "shot" / (rally_name + "_hit_predict.csv")
        assert hit_path.is_file()
        hits = read_hits(hit_path)
        shot_count = len(hits["frames"])

        court_path = match_path / "court" / (rally_name + ".out")
        court = Court(corners = court_points_to_corners(read_court(str(court_path))))

        all_poses = read_player_poses(str(match_path / "poses" / rally_name))

        trajectories_3d_path = match_path / "ball_trajectory_3d" / (rally_name + "_3d.csv")
        all_trajectories = pd.read_csv(trajectories_3d_path)

        duration = (hits["frames"][-1] - hits["frames"][0]) / fps

        shots = []
        for shot_idx in range(shot_count):
            start_frame_idx = int(hits["frames"][shot_idx])
            start_time = start_frame_idx / fps
            if shot_idx == shot_count-1:
                end_frame_idx = int(N_frames-1) # use the last frame in the rally to be the end of this shot
            else:
                end_frame_idx = max(0, int(hits["frames"][shot_idx+1]-1))
            end_time = end_frame_idx / fps

            start_poses = get_player_poses_frame(all_poses, start_frame_idx)
            end_poses = get_player_poses_frame(all_poses, end_frame_idx)
            if hits["player_ids"][shot_idx] == 1:
                start_position = start_poses["bottom"].get_base()
                end_position = end_poses["top"].get_base()
            elif hits["player_ids"][shot_idx] == 2:
                start_position = start_poses["top"].get_base()
                end_position = end_poses["bottom"].get_base()
            else:
                assert False, "Player id for hits not recognized"

            # convert pixel values to court coord
            start_position = court.unnormalize_court_position(court.pixel_to_court(start_position))
            end_position = court.unnormalize_court_position(court.pixel_to_court(end_position))

            # get trajectory slices
            trajectory = all_trajectories.loc[start_frame_idx:end_frame_idx,"ball_x":"ball_z"].values
            trajectory_y_relative = trajectory[:,1] - 6.71
            if len(trajectory_y_relative) >= 1 and trajectory_y_relative[0]*trajectory_y_relative[-1]:
                # if as expected trajectory ends on different sides, use the velocity vec when
                # crossing the net to determine offensive versus defensive
                idx = 0
                while trajectory_y_relative[idx]*trajectory_y_relative[idx+1] >= 0:
                    idx += 1
                    if idx >= len(trajectory_y_relative)-1:
                        break
                if idx == 0 or idx >= len(trajectory_y_relative)-1:
                    tendency = "N/A"
                else:
                    # computes velocity
                    velocity_z = (trajectory[idx+1,2] - trajectory[idx,2]) * fps
                    if velocity_z < 0:
                        tendency = "Offensive"
                    else:
                        tendency = "Defensive"
            else:
                tendency = "N/A"

            shot = {
                "index": shot_idx,
                "playerHit": PLAYER_ID_MAPPING[hits["player_ids"][shot_idx]],
                "startPlayerPosition": start_position.tolist(),
                "endPlayerPosition": end_position.tolist(),
                "startTime": start_time,
                "endTime": end_time,
                "tendency": tendency
            }
            shots.append(shot)

        rally = {
            "name": rally_name,
            "shotCount": shot_count,
            "duration": duration,
            "playerServe": PLAYER_ID_MAPPING[hits["player_ids"][0]],
            "shots": shots
        }

        summary_path = match_path / "summary" / (rally_name + "_summary.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as stream:
            print(f"Writing match summary to {summary_path}")
            json.dump(rally, stream, indent=2)

if __name__ == "__main__":
    dataset_base_path = Path("/sensei-fs/users/juiwang/ai-badminton/data/tracknetv2_042022/profession_dataset/")
    #for match_idx in range(23):
    #    match_path = dataset_base_path / f"match{match_idx}"
    #    if match_path.is_dir():
    #        create_match_summary(match_path, match_path / "summary")
    for match_idx in range(4):
        match_path = dataset_base_path / f"test_match{match_idx}"
        if match_path.is_dir():
            create_match_summary(match_path, match_path / "summary")
