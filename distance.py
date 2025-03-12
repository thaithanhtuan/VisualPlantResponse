import csv


def aggregate_speed(input_csv, output_csv, group_size=6, frame_interval=10.0):
    """
    Reads movingspeed.csv and aggregates the speed data over group_size frames.
    The output is written to output_csv. For each group, the average speed is computed.
    """
    with open(input_csv, 'r', newline='') as infile:
        reader = csv.reader(infile)
        header = next(reader)  # Expecting: Frame, Time (s), Label 1 Speed (pixels/sec), Label 2 Speed, ...
        rows = list(reader)

    aggregated_rows = []
    for i in range(0, len(rows), group_size):
        group = rows[i:i + group_size]
        # Use the last row's Frame and Time as the representative for this group (if available)
        frame_val = group[-1][0] if len(group[-1]) >= 1 else ''
        time_val = group[-1][1] if len(group[-1]) >= 2 else ''
        avg_speeds = []
        # Process each speed column from index 2 onward.
        for j in range(2, len(header)):
            speed_values = []
            for row in group:
                if len(row) > j and row[j] != '':
                    try:
                        speed_values.append(float(row[j]))
                    except ValueError:
                        continue
            avg_speed = sum(speed_values) / len(speed_values) if speed_values else 0.0
            avg_speeds.append(avg_speed)
        aggregated_rows.append([frame_val, time_val] + avg_speeds)

    with open(output_csv, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)  # Write the original header
        writer.writerows(aggregated_rows)
    print(f"Aggregated speed data over {group_size * frame_interval} seconds saved to {output_csv}.")


def compute_moving_distance(input_csv, output_csv, frame_interval=10.0):
    """
    Reads movingspeed.csv and computes moving distance over each frame (distance = speed * frame_interval).
    The result is written to output_csv.
    """
    with open(input_csv, 'r', newline='') as infile:
        reader = csv.reader(infile)
        header = next(reader)
        rows = list(reader)

    new_header = [header[0], header[1]] + [h.replace("Speed", "Distance") for h in header[2:]]
    distance_rows = []
    for row in rows:
        new_row = row[:2]  # Frame and Time
        for value in row[2:]:
            try:
                speed = float(value)
                distance = speed * frame_interval
            except ValueError:
                distance = value
            new_row.append(distance)
        distance_rows.append(new_row)

    with open(output_csv, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(new_header)
        writer.writerows(distance_rows)
    print(f"Moving distance over {frame_interval} seconds saved to {output_csv}.")





def aggregate_distance(input_csv, output_csv, group_size=6, frame_interval=10.0):
    """
    Reads movingdistance.csv and aggregates the moving distance data over group_size frames.
    The output is written to output_csv. For each group, the average moving distance is computed.
    """
    with open(input_csv, 'r', newline='') as infile:
        reader = csv.reader(infile)
        header = next(reader)  # Expecting: Frame, Time (s), Label 1 Distance (pixels), Label 2 Distance, ...
        rows = list(reader)

    aggregated_rows = []
    for i in range(0, len(rows), group_size):
        group = rows[i:i + group_size]
        # Use the last row's Frame and Time as the representative values for this group
        frame_val = group[-1][0] if len(group[-1]) >= 1 else ''
        time_val = group[-1][1] if len(group[-1]) >= 2 else ''
        avg_distances = []
        # Process each distance column (starting at index 2)
        for j in range(2, len(header)):
            distances = []
            for row in group:
                if len(row) > j and row[j] != '':
                    try:
                        distances.append(float(row[j]))
                    except ValueError:
                        continue
            avg_distance = sum(distances) / len(distances) if distances else 0.0
            avg_distances.append(avg_distance)
        aggregated_rows.append([frame_val, time_val] + avg_distances)

    with open(output_csv, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)  # Write the same header as the input file.
        writer.writerows(aggregated_rows)
    print(f"Aggregated moving distance data over {group_size * frame_interval} seconds saved to {output_csv}.")


# Example usage:
movingspeed_csv = "D:/Jeju/Thai/Dataset/Flower timelapse/Crop1/output/movingspeed.csv"
movingspeed_60_csv = "D:/Jeju/Thai/Dataset/Flower timelapse/Crop1/output/movingspeed_60.csv"
movingdistance_csv = "D:/Jeju/Thai/Dataset/Flower timelapse/Crop1/output/movingdistance.csv"

input_distance_csv = "D:/Jeju/Thai/Dataset/Flower timelapse/Crop1/output/movingdistance.csv"
output_distance_60_csv = "D:/Jeju/Thai/Dataset/Flower timelapse/Crop1/output/movingdistance_60.csv"



# Example usage:
movingspeed_csv = "D:/Jeju/Thai/Dataset/Flower timelapse/Crop2/output/movingspeed.csv"
movingspeed_60_csv = "D:/Jeju/Thai/Dataset/Flower timelapse/Crop2/output/movingspeed_60.csv"
movingdistance_csv = "D:/Jeju/Thai/Dataset/Flower timelapse/Crop2/output/movingdistance.csv"

input_distance_csv = "D:/Jeju/Thai/Dataset/Flower timelapse/Crop2/output/movingdistance.csv"
output_distance_60_csv = "D:/Jeju/Thai/Dataset/Flower timelapse/Crop2/output/movingdistance_60.csv"

"""# Example usage:
movingspeed_csv = "D:/Jeju/Thai/Dataset/Flower timelapse/Crop3/output/movingspeed.csv"
movingspeed_60_csv = "D:/Jeju/Thai/Dataset/Flower timelapse/Crop3/output/movingspeed_60.csv"
movingdistance_csv = "D:/Jeju/Thai/Dataset/Flower timelapse/Crop3/output/movingdistance.csv"

input_distance_csv = "D:/Jeju/Thai/Dataset/Flower timelapse/Crop3/output/movingdistance.csv"
output_distance_60_csv = "D:/Jeju/Thai/Dataset/Flower timelapse/Crop3/output/movingdistance_60.csv"
"""
aggregate_speed(movingspeed_csv, movingspeed_60_csv, group_size=6, frame_interval=10.0)
compute_moving_distance(movingspeed_csv, movingdistance_csv, frame_interval=10.0)

aggregate_distance(input_distance_csv, output_distance_60_csv, group_size=6, frame_interval=10.0)
