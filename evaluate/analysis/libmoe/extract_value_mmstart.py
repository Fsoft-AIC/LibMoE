import json
import glob

# Lấy tất cả các tệp trong thư mục logs
results = glob.glob("/cm/archive/anonymous/toolkitmoe/evaluate/logs/*")

for path in results:
    # Đường dẫn đến tệp results.json
    path_sub = f"{path}/results.json"
    try:
        # Mở tệp JSON
        with open(path_sub, 'r') as f:
            data = json.load(f)
    except:
        continue

    # Kiểm tra nếu có khóa "mmstar"
    if "mmstar" in data['results'].keys():
        # Lấy thông tin từ khóa mmstar
        mmstar_data = data['results']['mmstar']
        
        # Lấy các giá trị cần tính trung bình (bỏ qua các khóa stderr và alias)
        values = [
            mmstar_data["coarse perception,none"],
            mmstar_data["fine-grained perception,none"],
            mmstar_data["instance reasoning,none"],
            mmstar_data["logical reasoning,none"],
            mmstar_data["math,none"],
            mmstar_data["science & technology,none"]
        ]
        
        # Tính trung bình các giá trị
        average_score = str(sum(values) / len(values) *100).replace(".", ",")
        # print(average_score)
        # breakpoint()

        print(f"Path: {data['model_configs']['model_args']}")
        print(f"Average Score for mmstar: {average_score} \n")
