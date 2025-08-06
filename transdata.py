import csv

def txt_to_csv(txt_file_path, csv_file_path, encoding='utf-8'):
    """
    将文本文件转换为CSV格式

    参数:
    txt_file_path (str): 输入文本文件的路径
    csv_file_path (str): 输出CSV文件的路径
    encoding (str): 文件编码，默认为'utf-8'
    """
    try:
        with open(txt_file_path, 'r', encoding=encoding) as txt_file:
            # 读取所有行
            lines = txt_file.readlines()

            # 处理标题行
            headers = lines[0].strip().split('\t')

            # 处理数据行
            data = []
            for line in lines[1:]:
                if line.strip():  # 跳过空行
                    # 分割行数据并转换为适当的类型
                    values = line.strip().split('\t')
                    data.append(values)

        # 写入CSV文件
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            # 写入标题行
            writer.writerow(headers)
            # 写入数据行
            writer.writerows(data)

        print(f"成功将 {txt_file_path} 转换为 {csv_file_path}")

    except FileNotFoundError:
        print(f"错误: 文件 {txt_file_path} 未找到")
    except UnicodeDecodeError:
        print(f"错误: 文件 {txt_file_path} 不是 {encoding} 编码，请尝试其他编码")
    except Exception as e:
        print(f"错误: 发生了一个未知错误: {e}")


if __name__ == "__main__":
    # 输入和输出文件路径
    input_file = 'eval_new/farmer_evaluation_results.txt'
    output_file = 'eval_new/farmer_evaluation_results.csv'

    # 尝试使用gbk编码读取文件
    txt_to_csv(input_file, output_file, encoding='gbk')