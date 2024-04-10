import numpy as np

all_n_data_points = {
    1: [(106, 81920), (167, 114688), (273, 188416), (389, 262144), (531, 344064)],
    2: [(321, 212992), (595, 376832), (1144, 696320), (2021, 1261568), (3486, 2129920)],
    3: [(450, 270336), (901, 516096), (1735, 991232), (3208, 1802240), (5880, 3219456)],
    4: [(520, 286720), (1072, 557056), (2027, 1040384), (3814, 1908736), (7132, 3489792)],
    5: [(565, 303104), (1178, 581632), (2191, 1064960), (4150, 1966080), (7827, 3645440)],
    6: [(591, 294912), (1244, 589824), (2287, 1064960), (4340, 1949696), (8230, 3637248)],
}

for n, data_points in all_n_data_points.items():
    x_values = np.array([x for x, _ in data_points])
    y_values = np.array([y for _, y in data_points])
    slope, intercept = np.polyfit(x_values, y_values, 1)
    r_squared = np.corrcoef(x_values, y_values)[0, 1]**2
    num_for_gigabyte = (1_000_000_000 - intercept) / slope
    print(f"n={n}, num_for_gigabyte={round(num_for_gigabyte)}, slope={slope:.2f}, intercept={intercept:.2f}, r_squared={r_squared:.6f}")