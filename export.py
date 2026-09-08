#!/usr/bin/env python3
"""从 CSV 或 Excel 导入三元组到本地 Neo4j。

用法示例：
  python export.py "锡冶炼三元组（新）.csv"
  python export.py "锡冶炼三元组（新）.xlsx"
  python export.py "锡冶炼三元组（新）.csv" --yes

脚本会尝试从同目录下的 `config.py` 读取 NEO4J 配置，或使用环境变量覆盖。

CSV/Excel 列识别规则（默认）：
  列0: 实体1 (subject)
  列1: 关系 (predicate)
  列2: 实体2 (object)
  列3: 类别 (category，实体1和实体2共用的类别标签)
  列4: 属性名 (property_name，可选，属于实体2)
  列5: 属性值 (property_value，可选，属于实体2)
  
注意：
  - 实体1和实体2的类别都使用第3列（列3）的值
  - 属性名和属性值属于实体2（列2）
  - 导入顺序：先导入实体1 → 再导入实体2及其属性 → 最后创建关系

如果你的文件列顺序不同，可使用 --cols 参数指定逗号分隔的列索引，例如 --cols 0,1,2,3,4,5
"""
import os
import sys
import csv
import argparse
import logging
from datetime import datetime

try:
    this_dir = os.path.dirname(os.path.abspath(__file__))
    if this_dir not in sys.path:
        sys.path.insert(0, this_dir)
    import config
except Exception:
    config = None

from py2neo import Graph
import re

# 尝试导入 pandas 用于读取 Excel
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    logging.warning('未安装 pandas，无法读取 Excel 文件。请运行: pip install pandas openpyxl')


def get_neo4j_settings():
    uri = os.environ.get('NEO4J_URI') or (getattr(config, 'NEO4J_URI', None) if config else None)
    user = os.environ.get('NEO4J_USERNAME') or (getattr(config, 'NEO4J_USERNAME', None) if config else None)
    pwd = os.environ.get('NEO4J_PASSWORD') or (getattr(config, 'NEO4J_PASSWORD', None) if config else None)
    return uri, user, pwd


def normalize_label(s: str) -> str:
    s = (s or '').replace('`', '').strip()
    s = re.sub(r"[^\w\u4e00-\u9fff]+", "_", s)
    if not s:
        return 'Entity'
    return s[:64]


def normalize_relation(s: str) -> str:
    s = (s or '').replace('`', '').strip()
    s = re.sub(r"[^\w\u4e00-\u9fff]+", "_", s)
    if not s:
        return 'RELATED'
    return s[:64]


def try_open_csv(path: str):
    """尝试多种编码打开CSV文件"""
    encs = ['utf-8', 'utf-8-sig', 'gbk', 'gb18030']
    last_exc = None
    for e in encs:
        try:
            f = open(path, 'r', encoding=e, newline='')
            # peek first line
            _ = f.readline()
            f.seek(0)
            return f
        except Exception as ex:
            last_exc = ex
    raise last_exc


def read_file_rows(file_path: str):
    """
    读取文件并返回行迭代器
    支持 CSV 和 Excel (.xlsx) 格式
    对于 Excel 文件，会保留单元格的原始格式（如百分号）
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext in ['.xlsx', '.xls']:
        # Excel 文件 - 使用 openpyxl 直接读取以保留格式
        if not HAS_PANDAS:
            raise RuntimeError('读取 Excel 文件需要安装 pandas 和 openpyxl。请运行: pip install pandas openpyxl')
        
        logging.info(f'读取 Excel 文件: {file_path}（保留原始格式）')
        try:
            # 使用 openpyxl 直接读取，保留单元格格式
            try:
                from openpyxl import load_workbook
                HAS_OPENPYXL = True
            except ImportError:
                HAS_OPENPYXL = False
                logging.warning('未安装 openpyxl，将使用 pandas 读取（可能丢失格式）')
            
            if HAS_OPENPYXL:
                # 使用 openpyxl 直接读取，可以保留百分号等格式
                wb = load_workbook(file_path, data_only=False)
                ws = wb.active
                rows = []
                for row in ws.iter_rows(values_only=False):
                    row_data = []
                    for cell in row:
                        if cell.value is None:
                            row_data.append('')
                        elif cell.number_format and '%' in cell.number_format:
                            # 如果单元格格式包含百分号，手动转换并添加百分号
                            if isinstance(cell.value, (int, float)):
                                # Excel 中百分比存储为小数，需要乘以 100 并加上 %
                                # 使用 round 避免浮点数精度问题，保留合理的小数位数
                                percent_value = round(cell.value * 100, 10)  # 先 round 到 10 位小数
                                # 格式化：去除不必要的尾随零
                                if percent_value == int(percent_value):
                                    # 整数百分比
                                    row_data.append(f'{int(percent_value)}%')
                                else:
                                    # 小数百分比，去除尾随零
                                    formatted = f'{percent_value:.10f}'.rstrip('0').rstrip('.')
                                    row_data.append(f'{formatted}%')
                            else:
                                row_data.append(str(cell.value))
                        else:
                            row_data.append(str(cell.value))
                    rows.append(row_data)
                wb.close()
                return rows
            else:
                # 降级到 pandas
                df = pd.read_excel(file_path, sheet_name=0, header=None)
                return df.values.tolist()
        except Exception as e:
            logging.error(f'读取 Excel 文件失败: {e}')
            raise
    
    elif ext == '.csv':
        # CSV 文件
        logging.info(f'读取 CSV 文件: {file_path}')
        f = try_open_csv(file_path)
        reader = csv.reader(f)
        rows = list(reader)
        f.close()
        return rows
    
    else:
        raise ValueError(f'不支持的文件格式: {ext}。仅支持 .csv, .xlsx, .xls')


def import_file_to_neo4j(file_path: str, cols: list, dry_run: bool = False):
    """
    导入 CSV 或 Excel 文件到 Neo4j
    支持为第三列实体（object）添加属性
    列格式：实体1, 关系, 实体2, 类别, 属性名, 属性值
    
    数据按表格从前往后的顺序依次导入：
    - 每一行先导入第1列实体（使用第4列类别）
    - 再导入第3列实体及其属性（使用第4列类别）
    - 最后创建两个实体之间的关系
    """
    uri, user, pwd = get_neo4j_settings()
    if not uri or not user or not pwd:
        raise RuntimeError('找不到 Neo4j 配置，请设置环境变量或 tin/tin-smelte-backend/config.py 中的 NEO4J_*')

    logging.info(f'连接 Neo4j: {uri} (user={user})')
    g = Graph(uri, auth=(user, pwd))

    # 查询当前数据库中的节点数量
    try:
        node_count_query = "MATCH (n) RETURN count(n) AS total"
        total_nodes = g.run(node_count_query).evaluate() or 0
        logging.info(f'数据库中现有节点数量: {total_nodes}')
    except Exception as e:
        logging.warning(f'无法获取节点数量: {e}')
        total_nodes = 0

    # 查询当前最大的 sortIndex
    try:
        # 如果数据库为空，从 1 开始
        if total_nodes == 0:
            current_sort_index = 1
            logging.info('数据库为空，sortIndex 从 1 开始')
        else:
            # 查询最大的 sortIndex
            max_sort_index_query = "MATCH (n) WHERE n.sortIndex IS NOT NULL RETURN max(n.sortIndex) AS max_index"
            max_index_result = g.run(max_sort_index_query).evaluate()
            if max_index_result is None:
                current_sort_index = 1
            else:
                current_sort_index = int(max_index_result) + 1
            
            logging.info(f'当前 sortIndex 起始值: {current_sort_index}')
    except Exception as e:
        logging.warning(f'无法获取最大 sortIndex，使用默认值: {e}')
        current_sort_index = 1

    # 读取文件（按顺序读取所有行）
    rows = read_file_rows(file_path)
    logging.info(f'文件共有 {len(rows)} 行数据，将按顺序从第1行开始导入')

    total = 0
    success = 0
    errors = 0
    row_index = 0  # 记录当前处理的行号（从0开始）
    
    # 用于跟踪已分配 sortIndex 的节点，避免重复分配
    node_sort_index_map = {}  # {(label, name): sortIndex}

    # 按顺序遍历每一行
    for row in rows:
        row_index += 1
        # skip empty lines
        if not row or all([not (c and str(c).strip()) for c in row]):
            continue
        total += 1
        try:
            # 根据 cols 提取基本字段
            subj = str(row[cols[0]]).strip() if len(row) > cols[0] else ''
            pred = str(row[cols[1]]).strip() if len(row) > cols[1] else ''
            obj = str(row[cols[2]]).strip() if len(row) > cols[2] else ''
            
            # 类别（第4列）- 实体1和实体2共用
            category = str(row[cols[3]]).strip() if len(row) > cols[3] else ''
            
            # 提取属性名和属性值（属于实体2）
            # 属性值必须与表中属性值列一模一样，如果没有就不导入
            prop_name = ''
            prop_value = ''
            if len(cols) > 4 and len(row) > cols[4]:
                prop_name_raw = str(row[cols[4]]).strip()
                # 忽略 nan、NaN、空值
                if prop_name_raw and prop_name_raw.lower() not in ['nan', 'none', 'null', '']:
                    prop_name = prop_name_raw
            if len(cols) > 5 and len(row) > cols[5]:
                # 属性值完全保持原样，包括百分号、特殊字符等
                prop_value_raw = str(row[cols[5]]).strip()
                # 属性值必须存在且非空，才导入。保持与表格中一模一样的值（包括%等特殊字符）
                if prop_value_raw and prop_value_raw.lower() not in ['nan', 'none', 'null', '']:
                    prop_value = prop_value_raw  # 完全保留原始值，不做任何处理
            
            # 只有当属性名和属性值都存在时，才进行属性导入
            # 如果属性值列没有值，就不导入该属性
            if prop_name and not prop_value:
                prop_name = ''  # 如果只有属性名没有属性值，清空属性名

            if not subj or not pred or not obj:
                logging.warning(f'跳过第 {row_index} 行（缺少字段）：{row}')
                errors += 1
                continue

            # 实体1和实体2都使用第4列的类别
            label = normalize_label(category) if category else 'Entity'
            rel = normalize_relation(pred)
            
            # 日志显示两个实体共用同一个类别
            logging.debug(f'第{row_index}行: 实体1="{subj}"(类别:{label}), 实体2="{obj}"(类别:{label})')
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            today = datetime.now().strftime('%Y-%m-%d')

            if dry_run:
                prop_info = f", 属性: {prop_name}={prop_value}" if prop_name and prop_value else ""
                logging.info(f'[DRY] 第{row_index}行: MERGE ({label} {subj}) -[{rel}]-> ({label} {obj}){prop_info}')
                success += 1
                continue

            # 步骤1: 先导入实体1（第1列）
            # 为节点分配 sortIndex（如果该节点是第一次出现）
            node1_key = (label, subj)
            if node1_key not in node_sort_index_map:
                node_sort_index_map[node1_key] = current_sort_index
                node1_sort_index = current_sort_index
                current_sort_index += 1
            else:
                node1_sort_index = node_sort_index_map[node1_key]
            
            q1 = f"MERGE (a:`{label}` {{name: $subj}}) SET a.updatedAt=$today, a.sortIndex=$sortIndex RETURN id(a) as id"
            r1 = g.run(q1, subj=subj, today=today, sortIndex=node1_sort_index).data()
            logging.debug(f'第{row_index}行: 已导入实体1 "{subj}" (类别:{label}, sortIndex:{node1_sort_index})')

            # 步骤2: 再导入实体2（第3列）及其属性
            # 为节点分配 sortIndex（如果该节点是第一次出现）
            node2_key = (label, obj)
            if node2_key not in node_sort_index_map:
                node_sort_index_map[node2_key] = current_sort_index
                node2_sort_index = current_sort_index
                current_sort_index += 1
            else:
                node2_sort_index = node_sort_index_map[node2_key]
            
            if prop_name and prop_value:
                # 属性名和属性值都保持原样，不进行任何转换
                # 属性值要与表中"属性值"列的值一模一样（包括百分号%等特殊字符）
                # 使用参数化查询确保特殊字符（如%）不被转义
                # 注意：Neo4j 属性名不能包含某些特殊字符，这里做最小化处理
                # 将空格和特殊符号替换为下划线，但尽量保持原样
                safe_prop_name = prop_name.replace(' ', '_').replace('.', '_').replace('-', '_')
                # 如果属性名为纯中文，保持不变
                if all('\u4e00' <= c <= '\u9fff' or c.isalnum() or c == '_' for c in safe_prop_name):
                    pass  # 保持原样
                else:
                    # 移除其他特殊字符
                    safe_prop_name = ''.join(c if c.isalnum() or '\u4e00' <= c <= '\u9fff' or c == '_' else '_' for c in safe_prop_name)
                
                # 使用参数化查询，确保属性值中的特殊字符（%等）被完整保留
                q2 = (
                    f"MERGE (b:`{label}` {{name: $obj}}) "
                    f"SET b.updatedAt=$today, b.sortIndex=$sortIndex, b.`{safe_prop_name}`=$prop_value "
                    f"RETURN id(b) as id"
                )
                r2 = g.run(q2, obj=obj, today=today, sortIndex=node2_sort_index, prop_value=prop_value).data()
                # 日志中显示原始属性值，确认特殊字符被保留
                logging.info(f'第{row_index}行: 已导入实体2 "{obj}" (类别:{label}, sortIndex:{node2_sort_index})，并设置属性: {safe_prop_name}="{prop_value}" (原始值已保留)')
            else:
                # 没有属性或属性值为空/nan，只创建节点
                q2 = f"MERGE (b:`{label}` {{name: $obj}}) SET b.updatedAt=$today, b.sortIndex=$sortIndex RETURN id(b) as id"
                r2 = g.run(q2, obj=obj, today=today, sortIndex=node2_sort_index).data()
                logging.debug(f'第{row_index}行: 已导入实体2 "{obj}" (类别:{label}, sortIndex:{node2_sort_index})')

            # 步骤3: 最后创建关系
            qrel = (
                f"MATCH (a:`{label}` {{name: $subj}}),(b:`{label}` {{name: $obj}}) "
                f"MERGE (a)-[r:`{rel}`]->(b) SET r.createdAt=$now RETURN id(r) as id"
            )
            rr = g.run(qrel, subj=subj, obj=obj, now=now).data()
            logging.debug(f'第{row_index}行: 已创建关系 "{subj}" -[{rel}]-> "{obj}"')

            success += 1
            # 每处理10行显示一次进度
            if success % 10 == 0:
                logging.info(f'已成功导入 {success}/{total} 条记录...')
        except Exception as e:
            logging.exception(f'导入第 {row_index} 行失败: {e} -- 行内容: {row}')
            errors += 1

    logging.info(f'导入完成，总行数={total}，成功={success}，失败={errors}')
    logging.info(f'数据已按表格顺序（第1行→第{row_index}行）依次导入')
    return total, success, errors


def main():
    parser = argparse.ArgumentParser(description='CSV/Excel -> Neo4j 三元组导入（支持属性）')
    parser.add_argument('file', help='文件路径（支持 .csv, .xlsx, .xls）')
    parser.add_argument('--cols', default='0,1,2,3,4,5', help='列索引，逗号分隔，默认 0,1,2,3,4,5（实体1,关系,实体2,类别(共用),属性名,属性值）')
    parser.add_argument('--yes', action='store_true', help='无需交互直接导入')
    parser.add_argument('--dry', action='store_true', help='仅预览不写入')
    args = parser.parse_args()

    file_path = args.file
    if not os.path.exists(file_path):
        print(f'文件不存在: {file_path}')
        sys.exit(2)

    cols = [int(x) for x in args.cols.split(',')]

    if not args.yes:
        ans = input(f'准备将文件中的三元组导入 Neo4j，文件: {file_path}\n导入顺序：逐行处理，每行先导入实体1→再导入实体2及其属性→最后创建关系\n确认继续请输入 yes: ')
        if ans.strip().lower() != 'yes':
            print('已取消')
            sys.exit(0)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    try:
        import_file_to_neo4j(file_path, cols, dry_run=args.dry)
    except Exception as e:
        logging.exception('导入失败: %s', e)
        sys.exit(1)


if __name__ == '__main__':
    main()
